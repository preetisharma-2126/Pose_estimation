# Change Logs

## A. Crash-Causing Bugs (Fixed)

1. Checkpoint name mismatch: The script loaded from `last_checkpoint_2.pth` but saved to `last_checkpoint.pth`. On resume, it could fail or silently restart. Both now use one consistent path in `CFG`.
2. `pandas.DataFrame.iloc` inside DataLoader workers: Calling `.iloc[idx]` inside `__getitem__` from multiple workers caused GIL contention and RAM pressure. Fixed by converting DataFrames to NumPy arrays once at init.
3. No gradient clipping: Early quaternion instability could produce `Inf`/`NaN` gradients and break AMP scaling. Added `clip_grad_norm_` with clip value `1.0`.
4. No epsilon guard on quaternion norm: Dividing by near-zero quaternion norms caused `NaN` loss and scaler overflow. Fixed with `+ 1e-8`.
5. Non-atomic checkpoint saves: Mid-save crashes could corrupt checkpoints. Now uses `.tmp` plus `os.replace()` for atomic writes.

## B. Memory / Stability Issues (Fixed)

6. Image size `384 -> 224`: ResNet-50 is pretrained for `224x224`. `384` significantly increases memory and compute. `224` is now the default stable setting.
7. `optimizer.zero_grad()` -> `zero_grad(set_to_none=True)`: Frees gradient tensor storage earlier and lowers peak VRAM.
8. `torch.cuda.empty_cache()` + `gc.collect()` per epoch: Reduces long-run fragmentation risk.
9. Deprecated `torch.cuda.amp` -> `torch.amp`: Updated to current API (`GradScaler`, `autocast`).

## C. Performance Improvements (Added)

10. `torch.compile(model)`: Enables higher throughput on supported PyTorch/GPU stacks, with fallback behavior.
11. `prefetch_factor=4`: Keeps data queue ahead of the GPU to reduce stalls.
12. `num_workers` increased: Better pipeline throughput on capable machines.
13. Improved pose head: Replaced single `Linear(2048->7)` with `2048->512->7` + BatchNorm + Dropout.
14. Graceful `Ctrl+C`: Saves a valid checkpoint before exit.

## D. Additional Optimizations (train_v2 package)

15. Augmentation pipeline: Brightness/contrast jitter, small rotations, and Gaussian blur with controlled behavior.
16. Cosine warmup scheduler option: Smoother LR dynamics than abrupt plateau drops in many runs.
17. Backbone selection: Supports alternatives like `efficientnet_b4` for higher-accuracy tradeoffs.
18. TensorBoard integration: Real-time loss/LR monitoring.
19. Early stopping: Stops when validation loss plateaus to reduce overfitting.
20. Gradient accumulation: Supports larger effective batch sizes without extra VRAM.
