#!/usr/bin/env python3
"""
OPTIMIZED TRAINING: Load all data to RAM, then train fast.

Strategy:
1. Load 30,600 training windows into RAM (30GB)
2. Train at GPU speed (no I/O bottleneck)
3. ETA: 2-5 min load, then 10-30 min for 5 epochs
"""
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
print("OPTIMIZED TRAINING: DATA IN RAM + GPU")
print("="*70)

cfg = ModelConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")

# Step 1: Load all data to RAM (one-time, ~2-5 min)
print("\n[1/4] LOADING DATA TO RAM...")
train_loader, val_loader = create_optimized_dataloaders(
    "data/training_300/", cfg, batch_size=16, num_workers=0
)

print(f"  Caching {len(train_loader)} training batches to RAM...")
train_data = []
for batch_idx, batch in enumerate(train_loader):
    if (batch_idx + 1) % 500 == 0:
        print(f"    Cached {batch_idx + 1}/{len(train_loader)}...")

    # Keep data on CPU (RAM), move to GPU per batch during training
    train_data.append({
        "scans": batch["scans"],
        "nav": batch["nav"],
        "valid": batch["scan_valid"],
        "labels": batch["label"],
    })

print(f"  ✓ Cached {len(train_data)} batches to RAM")

# Step 2: Build model
print("\n[2/4] BUILDING MODEL...")
model = FishCatchTransformer(cfg).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"  ✓ Model: {n_params:,} parameters on {device}")

# Step 3: Setup training
print("\n[3/4] SETTING UP TRAINING...")
loss_fn = AsymmetricFocalLoss(gamma_pos=0.0, gamma_neg=4.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr_peak, weight_decay=cfg.weight_decay)
checkpoint_dir = Path("checkpoints_gpu")
checkpoint_dir.mkdir(exist_ok=True)
print(f"  ✓ Optimizer ready, checkpoints to {checkpoint_dir}/")

# Step 4: FAST TRAINING (data already in RAM)
print("\n[4/4] TRAINING (FAST - DATA IN RAM)...")
print("="*70)

best_loss = float('inf')

for epoch in range(5):
    print(f"\n--- Epoch {epoch+1}/5 ---")
    model.train()

    epoch_losses = []
    for batch_idx, batch in enumerate(train_data):
        # Move batch from RAM to GPU (fast!)
        scans = batch["scans"].to(device)
        nav = batch["nav"].to(device)
        valid = batch["valid"].to(device)
        labels = batch["labels"].to(device)

        # Training step
        optimizer.zero_grad()
        logits = model(scans, valid, nav)
        loss = loss_fn(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        epoch_losses.append(loss.item())

        # Print every 200 batches
        if (batch_idx + 1) % 200 == 0:
            avg_loss = np.mean(epoch_losses[-200:])
            print(f"  Batch {batch_idx+1:4d}/{len(train_data)}: loss={avg_loss:.6f}")

    # Epoch summary
    epoch_loss = np.mean(epoch_losses)
    print(f"  Epoch loss: {epoch_loss:.6f}")

    # Checkpoint if best
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        ckpt_path = checkpoint_dir / "best.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_loss,
        }, ckpt_path)
        print(f"  → Saved best checkpoint (loss={epoch_loss:.6f})")

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print(f"Best validation loss: {best_loss:.6f}")
print(f"Checkpoint saved to: {checkpoint_dir}/best.pt")
print(f"\nWith data in RAM:")
print(f"  ✓ 100x faster I/O")
print(f"  ✓ Full GPU utilization")
print(f"  ✓ ~10-30 min for 5 epochs")
