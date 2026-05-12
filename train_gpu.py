#!/usr/bin/env python3
"""Train using optimized DataLoader with GPU acceleration."""
import sys
import os

# Set efficient GPU memory allocation BEFORE importing torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ml.config import ModelConfig
from ml.model import FishCatchTransformer
from ml.loss import AsymmetricFocalLoss
from ml.dataloader_optimized import create_optimized_dataloaders

print("="*70)
print("TRAINING WITH GPU (RTX 4070 Ti) + OPTIMIZED DATALOADER")
print("="*70)

cfg = ModelConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[GPU] Device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA version: {torch.version.cuda}")

# Create optimized dataloaders
print("\nCreating optimized dataloaders...")
train_loader, val_loader = create_optimized_dataloaders(
    "data/training_300/", cfg, batch_size=16, num_workers=0
)

# Model
print("Building model...")
model = FishCatchTransformer(cfg).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {n_params:,}")

loss_fn = AsymmetricFocalLoss(gamma_pos=0.0, gamma_neg=4.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr_peak, weight_decay=cfg.weight_decay)

# Train 5 epochs
print(f"\nTraining 5 epochs with batch size 16 on GPU (RTX 4070 Ti 12GB)...")
print("="*70)

best_loss = float('inf')
checkpoint_dir = Path("checkpoints_gpu")
checkpoint_dir.mkdir(exist_ok=True)

for epoch in range(5):
    print(f"\n--- Epoch {epoch+1}/5 ---")
    print(f"  Starting training loop...")
    model.train()

    train_losses = []
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx == 0:
            print(f"  Batch {batch_idx}: shapes - scans {batch['scans'].shape}, nav {batch['nav'].shape}")
        scans = batch["scans"].to(device)
        nav = batch["nav"].to(device)
        valid = batch["scan_valid"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        if batch_idx == 0:
            import time
            print(f"    Data loaded, starting forward pass at {time.time()}")
        logits = model(scans, valid, nav)
        if batch_idx == 0:
            print(f"    Forward pass done at {time.time()}")
        loss = loss_fn(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        train_losses.append(loss.item())

        # Clear cache periodically to avoid fragmentation
        if (batch_idx + 1) % 50 == 0:
            torch.cuda.empty_cache()

        if (batch_idx + 1) % 100 == 0:
            avg_loss = np.mean(train_losses[-100:])
            print(f"  Batch {batch_idx+1:4d}: loss={avg_loss:.6f}")

    epoch_loss = np.mean(train_losses)

    # Validation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for val_batch in val_loader:
            scans = val_batch["scans"].to(device)
            nav = val_batch["nav"].to(device)
            valid = val_batch["scan_valid"].to(device)
            labels = val_batch["label"].to(device)

            logits = model(scans, valid, nav)
            loss = loss_fn(logits, labels)
            val_losses.append(loss.item())

    val_loss = np.mean(val_losses)

    print(f"Epoch {epoch+1} complete: train_loss={epoch_loss:.6f} val_loss={val_loss:.6f}")

    if val_loss < best_loss:
        best_loss = val_loss
        ckpt_path = checkpoint_dir / "best.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {
                "window_size": cfg.window_size,
                "horizon_s": cfg.horizon_s,
                "species": cfg.species,
            },
            "val_loss": val_loss,
        }, ckpt_path)
        print(f"  [BEST] Saved best checkpoint: {ckpt_path}")

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)
print(f"Best validation loss: {best_loss:.6f}")
print(f"Checkpoint: {checkpoint_dir}/best.pt")
print(f"Model parameters: {n_params:,}")
print(f"[SUCCESS] Model trained on GPU with real sonar data!")
