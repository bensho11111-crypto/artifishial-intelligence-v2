#!/usr/bin/env python3
"""Train with data pre-loaded to RAM (fast GPU training)."""
import sys
import os
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ml.config import ModelConfig
from ml.model import FishCatchTransformer
from ml.loss import AsymmetricFocalLoss
from ml.dataloader_optimized import create_optimized_dataloaders

print("="*70)
print("TRAINING: DATA CACHED IN RAM + GPU")
print("="*70)

cfg = ModelConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")

# Create optimized dataloaders
print("\nLoading and caching data to RAM...")
train_loader, val_loader = create_optimized_dataloaders(
    "data/training_300/", cfg, batch_size=16, num_workers=0
)

# Pre-load ALL training data to CPU RAM (not GPU - too large)
print("Caching training data to CPU RAM...")
train_data = []
for batch_idx, batch in enumerate(train_loader):
    if (batch_idx + 1) % 100 == 0:
        print(f"  Cached {batch_idx + 1}/{len(train_loader)} batches...")
    cached_batch = {
        "scans": batch["scans"],  # Keep on CPU
        "nav": batch["nav"],
        "valid": batch["scan_valid"],
        "labels": batch["label"],
    }
    train_data.append(cached_batch)

print(f"Cached {len(train_data)} batches to {device}")

# Model
print("\nBuilding model...")
model = FishCatchTransformer(cfg).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {n_params:,}")

loss_fn = AsymmetricFocalLoss(gamma_pos=0.0, gamma_neg=4.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr_peak, weight_decay=cfg.weight_decay)

# Train 2 epochs with cached data (fast!)
print(f"\nTraining 2 epochs with cached data...")
print("="*70)

best_loss = float('inf')
checkpoint_dir = Path("checkpoints_gpu")
checkpoint_dir.mkdir(exist_ok=True)

for epoch in range(2):
    print(f"\n--- Epoch {epoch+1}/2 ---")
    model.train()

    train_losses = []
    for batch_idx, batch in enumerate(train_data):
        # Move batch to device (from CPU RAM to GPU)
        scans = batch["scans"].to(device)
        nav = batch["nav"].to(device)
        valid = batch["valid"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(scans, valid, nav)
        loss = loss_fn(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        train_losses.append(loss.item())

        if (batch_idx + 1) % 100 == 0:
            avg_loss = np.mean(train_losses[-100:])
            print(f"  Batch {batch_idx+1:4d}: loss={avg_loss:.6f}")

    epoch_loss = np.mean(train_losses)
    print(f"Epoch {epoch+1} complete: avg_loss={epoch_loss:.6f}")

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        ckpt_path = checkpoint_dir / "best.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "loss": epoch_loss,
        }, ckpt_path)
        print(f"  Saved checkpoint: {ckpt_path}")

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)
print(f"Best loss: {best_loss:.6f}")
print(f"Checkpoint: {checkpoint_dir}/best.pt")
print(f"\nWith data cached in RAM, training is FAST!")
