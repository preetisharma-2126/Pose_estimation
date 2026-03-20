import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import cv2
import os
import numpy as np
import torch.nn as nn
import torchvision.models as models
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import time
import subprocess



# =====================================================
# GPU LOGGER
# =====================================================
def get_gpu_stats():
    try:
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits",
            ]
        )
        mem_used, mem_total, power = result.decode().strip().split(", ")
        return int(mem_used), int(mem_total), float(power)
    except:
        return None, None, None



# =====================================================
# DATASET
# =====================================================
class SatellitePoseDataset(Dataset):

    def __init__(
        self,
        csv_path,
        bbox_path,
        img_dir,
        mask_dir=None,
        max_samples=None,
        img_size=384,
    ):

        self.df = pd.read_csv(csv_path, header=None)
        self.bbox_df = pd.read_csv(bbox_path, header=None)

        if max_samples is not None:
            self.df = self.df.iloc[:max_samples]
            self.bbox_df = self.bbox_df.iloc[:max_samples]

        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size

        print("Total samples:", len(self.df))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]
        bbox_row = self.bbox_df.iloc[idx]

        img_id = int(row[0])
        img_name = f"scene{img_id}.png"
        img_path = os.path.join(self.img_dir, img_name)

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        xmin = int(bbox_row[1])
        xmax = int(bbox_row[2])
        ymin = int(bbox_row[3])
        ymax = int(bbox_row[4])

        h, w = image.shape
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)

        cropped = image[ymin:ymax, xmin:xmax]

        if cropped.size == 0:
            cropped = image.copy()

        if self.mask_dir is not None:

            mask_path = os.path.join(self.mask_dir, img_name)

            if os.path.exists(mask_path):

                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                if mask is not None:
                    mask_crop = mask[ymin:ymax, xmin:xmax]
                    mask_bin = (mask_crop > 0).astype(np.uint8)
                    cropped = cropped * mask_bin

        image_resized = cv2.resize(cropped, (self.img_size, self.img_size))
        image_resized = image_resized.astype(np.float32) / 255.0

        image_resized = np.stack(
            [image_resized, image_resized, image_resized], axis=0
        )

        image_resized = (image_resized - 0.5) / 0.5

        image_tensor = torch.from_numpy(image_resized).float()

        pose_values = row[[1, 2, 3, 6, 7, 8, 9]].values.astype(np.float32)

        pose_values[0:3] /= 100.0

        q = pose_values[3:]
        norm_q = np.linalg.norm(q)

        if norm_q > 0:
            q = q / norm_q

        pose_values[3:] = q

        pose_tensor = torch.from_numpy(pose_values).float()

        return image_tensor, pose_tensor



# =====================================================
# MODEL
# =====================================================
class PoseResNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1
        )

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 7)

    def forward(self, x):
        return self.backbone(x)



# =====================================================
# LOSS
# =====================================================
class PoseLoss(nn.Module):

    def __init__(self, beta=5.0):
        super().__init__()
        self.beta = beta

    def forward(self, pred, target):

        trans_loss = torch.mean((pred[:, :3] - target[:, :3]) ** 2)

        pred_q = pred[:, 3:]
        target_q = target[:, 3:]

        pred_q = pred_q / torch.norm(pred_q, dim=1, keepdim=True)
        target_q = target_q / torch.norm(target_q, dim=1, keepdim=True)

        dot = torch.abs(torch.sum(pred_q * target_q, dim=1))
        rot_loss = torch.mean(1 - dot)

        return trans_loss + self.beta * rot_loss



# =====================================================
# TRAINING
# =====================================================
if __name__ == "__main__":

    print("\n===== RAPTOR 6D Pose Training =====\n")

    dataset = SatellitePoseDataset(
        r"/home/spacets/data/computer_vision/raptor/poses.csv",
        r"/home/spacets/data/computer_vision/raptor/bbox.csv",
        r"/home/spacets/data/computer_vision/raptor/camera_output/camera_output",
        mask_dir=r"/home/spacets/data/computer_vision/raptor/masks/masks",
        max_samples=None,
        img_size=384,
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = PoseResNet().to(device)

    criterion = PoseLoss(beta=5.0)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4
    )

    scaler = GradScaler()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=3,
        factor=0.5,
    )

    start_epoch = 0
    best_val_loss = float("inf")

    if os.path.exists("last_checkpoint_2.pth"):

        print("Resuming from last checkpoint...")

        checkpoint = torch.load("last_checkpoint_2.pth")

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])

        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val"]

    EPOCHS =50

    for epoch in range(start_epoch, EPOCHS):

        epoch_start = time.time()

        model.train()
        train_loss = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")

        for images, poses in loop:

            images = images.to(device)
            poses = poses.to(device)

            optimizer.zero_grad()

            with autocast():

                preds = model(images)
                loss = criterion(preds, poses)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            loop.set_postfix(loss=loss.item())

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0

        with torch.no_grad():

            for images, poses in val_loader:

                images = images.to(device)
                poses = poses.to(device)

                with autocast():

                    preds = model(images)
                    loss = criterion(preds, poses)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                "best_pose_model_2.pth"
            )

            print("Best model saved!")

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_val": best_val_loss,
            },
            "last_checkpoint.pth",
        )

        mem_used, mem_total, power = get_gpu_stats()

        epoch_time = time.time() - epoch_start

        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss:   {val_loss:.6f}")
        print(f"Epoch Time: {epoch_time:.2f} sec")

        if mem_used is not None:
            print(f"GPU Memory: {mem_used}/{mem_total} MB")
            print(f"GPU Power:  {power:.2f} W")

    print("\nTraining Complete :rocket:")