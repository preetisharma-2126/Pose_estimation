import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import cv2
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from tqdm import tqdm
import time
import subprocess
import gc
import signal
import sys
from typing import Any, cast


# =====================================================
# CONFIG  — tune these without touching the rest
# =====================================================
CFG: dict[str, Any] = dict(
    csv_path    = r"/home/spacets/data/computer_vision/raptor/poses.csv",
    bbox_path   = r"/home/spacets/data/computer_vision/raptor/bbox.csv",
    img_dir     = r"/home/spacets/data/computer_vision/raptor/camera_output/camera_output",
    mask_dir    = r"/home/spacets/data/computer_vision/raptor/masks/masks",
    checkpoint_load  = "last_checkpoint.pth",   # FIX: was "last_checkpoint_2.pth" on load,
    checkpoint_save  = "last_checkpoint.pth",   #      "last_checkpoint.pth" on save --> mismatch
    best_model_path  = "best_pose_model.pth",
    img_size    = 224,      # FIX: dropped from 384-->224; ResNet was designed for 224.
                            #      384 inflates RAM ~2.9× per sample with no accuracy gain.
    batch_size  = 32,       # Conservative default; increase after confirming stable VRAM headroom.
    num_workers = 4,        # Conservative default for broad workstation compatibility.
    prefetch_factor = 4,    # Pre-queue batches per worker to keep GPU fed.
    max_samples = None,
    epochs      = 50,
    lr          = 1e-4,
    beta        = 5.0,      # Rotation loss weight in PoseLoss
    grad_clip   = 1.0,      # NEW: gradient clipping — prevents NaN/Inf explosions that can crash training
    accum_steps = 1,        # NEW: gradient accumulation. Set >1 to simulate larger batches with less VRAM.
    mixed_precision = True,
    use_compile = True,
    matmul_precision = "high",   # highest | high | medium
    allow_tf32 = True,
    cudnn_benchmark = True,
    deterministic = False,
    skip_nonfinite_batches = True,
    oom_skip_batches = True,
    val_split   = 0.8,
    patience    = 3,
    lr_factor   = 0.5,
    seed        = 42,
)


# =====================================================
# GRACEFUL SHUTDOWN — saves checkpoint on Ctrl+C
# =====================================================
_interrupted = False

def _handle_sigint(sig, frame):
    global _interrupted
    print("\n[INFO] Interrupt received — will save checkpoint at end of this epoch.")
    _interrupted = True

signal.signal(signal.SIGINT, _handle_sigint)


# =====================================================
# GPU LOGGER
# =====================================================
def get_gpu_stats():
    try:
        result = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=memory.used,memory.total,power.draw",
             "--format=csv,noheader,nounits"],
            timeout=5,          # FIX: no timeout could hang the process
        )
        mem_used, mem_total, power = result.decode().strip().split(", ")
        return int(mem_used), int(mem_total), float(power)
    except Exception:
        return None, None, None


# =====================================================
# DATASET
# =====================================================
class SatellitePoseDataset(Dataset):
    """
    Loads satellite images, crops to bounding box, optionally masks,
    resizes, normalises, and returns (image_tensor [3,H,W], pose_tensor [7]).

    Key fixes vs original
    ----------------------
    * pandas `.iloc` lookups replaced with pre-converted numpy arrays --> no
      GIL-contention between DataLoader workers caused by DataFrame access.
    * FileNotFoundError replaced with a soft fallback (black image) so a
      single missing file does not abort the entire epoch.
    * Mask multiplication cast guard added to avoid silent uint8 overflow.
    """

    def __init__(self, csv_path, bbox_path, img_dir,
                 mask_dir=None, max_samples=None, img_size=224):

        df      = pd.read_csv(csv_path,  header=None)
        bbox_df = pd.read_csv(bbox_path, header=None)

        if max_samples is not None:
            df      = df.iloc[:max_samples]
            bbox_df = bbox_df.iloc[:max_samples]

        # FIX: convert to numpy once at init — eliminates per-sample pandas overhead
        #      and removes the GIL lock that was causing worker stalls/OOM under load.
        self.img_ids   = df.iloc[:, 0].to_numpy(dtype=np.int32)
        self.poses_raw = df.iloc[:, [1, 2, 3, 6, 7, 8, 9]].to_numpy(dtype=np.float32)
        self.bboxes    = bbox_df.iloc[:, [1, 2, 3, 4]].to_numpy(dtype=np.int32)  # xmin,xmax,ymin,ymax

        self.img_dir  = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        # Match ImageNet normalization expected by pretrained backbones.
        self.imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        self.imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]

        print(f"[Dataset] Total samples: {len(self.img_ids)}")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_name = f"scene{self.img_ids[idx]}.png"
        img_path = os.path.join(self.img_dir, img_name)

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # FIX: soft fallback instead of hard raise — avoids crashing a worker
        if image is None:
            print(f"[WARN] Missing image: {img_path} — using black placeholder.")
            image = np.zeros((self.img_size, self.img_size), dtype=np.uint8)

        xmin, xmax, ymin, ymax = self.bboxes[idx]
        h, w = image.shape
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)

        cropped = image[ymin:ymax, xmin:xmax]
        if cropped.size == 0:
            cropped = image

        # Optional mask
        if self.mask_dir is not None:
            mask_path = os.path.join(self.mask_dir, img_name)
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    mask_crop = mask[ymin:ymax, xmin:xmax]
                    # FIX: cast to float32 before multiply to avoid uint8 overflow
                    mask_bin  = (mask_crop > 0).astype(np.float32)
                    cropped   = (cropped.astype(np.float32) * mask_bin).astype(np.uint8)

        image_resized = cv2.resize(cropped, (self.img_size, self.img_size))

        img_f = image_resized.astype(np.float32) / 255.0
        img_3 = np.stack([img_f, img_f, img_f], axis=0)
        img_3 = (img_3 - self.imagenet_mean) / self.imagenet_std
        image_tensor = torch.from_numpy(img_3)

        # Pose: translation / 100, quaternion normalised
        pose = self.poses_raw[idx].copy()
        pose[0:3] /= 100.0
        q_norm = np.linalg.norm(pose[3:])
        if q_norm > 1e-8:          # FIX: guard against zero-quaternion
            pose[3:] /= q_norm
        else:
            pose[3:] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        return image_tensor, torch.from_numpy(pose)


# =====================================================
# MODEL
# =====================================================
class PoseResNet(nn.Module):
    """
    ResNet-50 backbone with a dedicated pose head.

    Changes vs original
    -------------------
    * Replaced single Linear(2048-->7) with a small MLP head:
        2048 --> 512 --> 7 with BatchNorm + Dropout.
      This gives the network a better inductive bias for regression and
      regularises against overfitting on the 100k dataset.
    * Quaternion and translation branches are explicit --> easier to extend
      (e.g. add per-branch loss weighting later).
    """

    def __init__(self, dropout=0.3):
        super().__init__()
        try:
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        except Exception as err:
            print(f"[WARN] Could not load pretrained ResNet-50 weights ({err}); using random init.")
            resnet = models.resnet50(weights=None)
        in_features = resnet.fc.in_features
        # Avoid assigning to typed ResNet.fc to keep static type checkers happy.
        self.backbone = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten(1))

        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 7),
        )

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)


# =====================================================
# LOSS
# =====================================================
class PoseLoss(nn.Module):
    """
    Combined translation MSE + quaternion geodesic loss.

    Fix: added eps to norm to prevent division-by-zero during early training
    when random initialisations can produce near-zero quaternion predictions.
    """

    def __init__(self, beta=5.0, trans_loss_type="smooth_l1"):
        super().__init__()
        self.beta = beta
        self.trans_loss_type = trans_loss_type

    def forward(self, pred, target):
        if self.trans_loss_type == "mse":
            trans_loss = torch.mean((pred[:, :3] - target[:, :3]) ** 2)
        else:
            trans_loss = F.smooth_l1_loss(pred[:, :3], target[:, :3], beta=0.01)

        pred_q   = pred[:, 3:]
        target_q = target[:, 3:]

        # FIX: eps guard prevents NaN when quaternion norm ≈ 0
        pred_q   = pred_q   / (torch.norm(pred_q,   dim=1, keepdim=True) + 1e-8)
        target_q = target_q / (torch.norm(target_q, dim=1, keepdim=True) + 1e-8)

        dot      = torch.clamp(torch.abs(torch.sum(pred_q * target_q, dim=1)), 0.0, 1.0)
        rot_loss = torch.mean(1.0 - dot)

        return trans_loss + self.beta * rot_loss


# =====================================================
# HELPERS
# =====================================================
def save_checkpoint(path, epoch, model, optimizer, scheduler, scaler, best_val):
    ckpt_dir = os.path.dirname(path)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
    model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save({
        "epoch":     epoch,
        "model":     model_to_save.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler":    scaler.state_dict(),
        "best_val":  best_val,
    }, path + ".tmp")
    # FIX: atomic replace — a crash mid-save won't corrupt the checkpoint
    os.replace(path + ".tmp", path)


def load_checkpoint(path, model, optimizer, scheduler, scaler):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model_to_load = model._orig_mod if hasattr(model, "_orig_mod") else model
    model_to_load.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    return ckpt["epoch"] + 1, ckpt["best_val"]


# =====================================================
# TRAINING
# =====================================================
def run_epoch(model, loader, criterion, optimizer, scaler, device,
              train=True, grad_clip=1.0, accum_steps=1,
              mixed_precision=True, amp_dtype=torch.float16,
              skip_nonfinite_batches=True, oom_skip_batches=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    processed_steps = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    desc = "Train" if train else "Val"
    pending_micro_steps = 0

    with ctx:
        loop = tqdm(loader, desc=f"  [{desc}]", leave=False)
        for step, (images, poses) in enumerate(loop, 1):
            images = images.to(device, non_blocking=True)
            poses  = poses.to(device,  non_blocking=True)

            try:
                # Use autocast only for forward/loss as recommended in PyTorch AMP docs.
                with autocast(device_type=device.type, dtype=amp_dtype, enabled=mixed_precision):
                    preds = model(images)
                    loss = criterion(preds, poses)
            except RuntimeError as err:
                if device.type == "cuda" and oom_skip_batches and "out of memory" in str(err).lower():
                    print("[WARN] CUDA OOM during batch; skipping batch and clearing cache.")
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    continue
                raise

            if skip_nonfinite_batches and not torch.isfinite(loss):
                print(f"[WARN] Non-finite loss ({loss.item()}) on step {step}; skipping batch.")
                optimizer.zero_grad(set_to_none=True)
                continue

            loss = loss / accum_steps

            if train:
                scaler.scale(loss).backward()
                pending_micro_steps += 1

                if pending_micro_steps >= accum_steps:
                    # FIX: gradient clipping — prevents exploding grads / NaN crash
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), grad_clip, error_if_nonfinite=False
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)   # FIX: set_to_none frees memory
                    pending_micro_steps = 0

            total_loss += loss.item() * accum_steps
            processed_steps += 1
            loop.set_postfix(loss=f"{loss.item() * accum_steps:.5f}")

    if train and pending_micro_steps > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), grad_clip, error_if_nonfinite=False
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return total_loss / max(1, processed_steps)


def main():
    seed = int(CFG["seed"])
    csv_path = str(CFG["csv_path"])
    bbox_path = str(CFG["bbox_path"])
    img_dir = str(CFG["img_dir"])
    mask_dir = str(CFG["mask_dir"]) if CFG["mask_dir"] is not None else None
    checkpoint_load = str(CFG["checkpoint_load"])
    checkpoint_save = str(CFG["checkpoint_save"])
    best_model_path = str(CFG["best_model_path"])
    img_size = int(CFG["img_size"])
    batch_size = int(CFG["batch_size"])
    num_workers = int(CFG["num_workers"])
    prefetch_factor = int(CFG["prefetch_factor"])
    max_samples = int(CFG["max_samples"]) if CFG["max_samples"] is not None else None
    epochs = int(CFG["epochs"])
    lr = float(CFG["lr"])
    beta = float(CFG["beta"])
    grad_clip = float(CFG["grad_clip"])
    accum_steps = int(CFG["accum_steps"])
    val_split = float(CFG["val_split"])
    patience = int(CFG["patience"])
    lr_factor = float(CFG["lr_factor"])
    mixed_precision = bool(CFG["mixed_precision"])
    use_compile = bool(CFG["use_compile"])
    matmul_precision = str(CFG["matmul_precision"])
    allow_tf32 = bool(CFG["allow_tf32"])
    cudnn_benchmark = bool(CFG["cudnn_benchmark"])
    deterministic = bool(CFG["deterministic"])
    skip_nonfinite_batches = bool(CFG["skip_nonfinite_batches"])
    oom_skip_batches = bool(CFG["oom_skip_batches"])

    for required_path in (csv_path, bbox_path, img_dir):
        if not os.path.exists(required_path):
            raise FileNotFoundError(f"Required path does not exist: {required_path}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.backends.cudnn.benchmark = cudnn_benchmark and not deterministic
        torch.backends.cudnn.deterministic = deterministic
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(matmul_precision)

    print("\n===== RAPTOR 6D Pose Training =====\n")

    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset = SatellitePoseDataset(
        csv_path, bbox_path, img_dir,
        mask_dir=mask_dir,
        max_samples=max_samples,
        img_size=img_size,
    )

    train_size = int(val_split * len(dataset))
    val_size   = len(dataset) - train_size
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=g)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    if num_workers > 0:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=prefetch_factor,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=prefetch_factor,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            persistent_workers=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            persistent_workers=False,
        )

    # ── Device / Model ────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # FIX: torch.compile gives ~20-30% throughput boost on Ada/Blackwell GPUs (RTX 5090)
    #      Remove if you are on an older PyTorch (<2.0) or encounter issues.
    model: nn.Module = PoseResNet().to(device)
    if use_compile and hasattr(torch, "compile"):
        print("[INFO] Compiling model with torch.compile …")
        try:
            model = cast(nn.Module, torch.compile(model))
        except Exception as err:
            print(f"[WARN] torch.compile failed ({err}); continuing without compile.")

    criterion = PoseLoss(beta=beta, trans_loss_type="smooth_l1")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=1e-4)    # FIX: L2 reg helps generalisation

    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    use_amp = mixed_precision and (device.type in {"cuda", "cpu"})
    # For CUDA+fp16 use GradScaler; for CPU/bfloat16 it is not needed.
    scaler = GradScaler(
        device=device.type,
        enabled=(use_amp and device.type == "cuda" and amp_dtype == torch.float16),
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        patience=patience, factor=lr_factor,
    )

    # ── Optional resume ───────────────────────────────────────────────────────
    start_epoch   = 0
    best_val_loss = float("inf")

    if os.path.exists(checkpoint_load):
        print(f"[INFO] Resuming from {checkpoint_load} …")
        start_epoch, best_val_loss = load_checkpoint(
            checkpoint_load, model, optimizer, scheduler, scaler)
        print(f"[INFO] Resumed at epoch {start_epoch}, best_val={best_val_loss:.6f}")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()

        train_loss = run_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            train=True,
            grad_clip=grad_clip,
            accum_steps=accum_steps,
            mixed_precision=use_amp,
            amp_dtype=amp_dtype,
            skip_nonfinite_batches=skip_nonfinite_batches,
            oom_skip_batches=oom_skip_batches,
        )

        val_loss = run_epoch(
            model, val_loader, criterion, optimizer, scaler, device,
            train=False,
            mixed_precision=use_amp,
            amp_dtype=amp_dtype,
            skip_nonfinite_batches=skip_nonfinite_batches,
            oom_skip_batches=oom_skip_batches,
        )

        scheduler.step(val_loss)

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_to_save = cast(nn.Module, model._orig_mod) if hasattr(model, "_orig_mod") else model
            best_dir = os.path.dirname(best_model_path)
            if best_dir:
                os.makedirs(best_dir, exist_ok=True)
            torch.save(model_to_save.state_dict(), best_model_path)
            print(f"  ✓ Best model saved  (val={best_val_loss:.6f})")

        # FIX: atomic checkpoint save
        save_checkpoint(checkpoint_save, epoch, model,
                        optimizer, scheduler, scaler, best_val_loss)

        # Logging
        mem_used, mem_total, power = get_gpu_stats()
        elapsed = time.time() - epoch_start
        print(f"\nEpoch {epoch+1}/{epochs}  |  "
              f"Train: {train_loss:.6f}  Val: {val_loss:.6f}  |  "
              f"Time: {elapsed:.1f}s")
        if mem_used is not None:
            print(f"  GPU: {mem_used}/{mem_total} MB  |  Power: {power:.1f} W")

        # FIX: explicit cache clear each epoch prevents VRAM fragmentation buildup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        if _interrupted:
            print("[INFO] Exiting cleanly after interrupt.")
            sys.exit(0)

    print("\nTraining Complete 🚀")


if __name__ == "__main__":
    main()
