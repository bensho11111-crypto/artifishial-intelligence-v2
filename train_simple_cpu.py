#!/usr/bin/env python3
"""
SIMPLE TRAINING ON CPU

Just train on synthetic data without evaluation.
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ml.config import ModelConfig
from ml.model import FishCatchTransformer
from ml.loss import AsymmetricFocalLoss

print("=" * 70)
print("SIMPLE TRAINING: SYNTHETIC DATA ON CPU")
print("=" * 70)

cfg = ModelConfig()
device = "cpu"
print(f"\nDevice: {device}")

# Build model
print("\nBuilding model...")
model = FishCatchTransformer(cfg).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {n_params:,}")

loss_fn = AsymmetricFocalLoss(gamma_pos=0.0, gamma_neg=4.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr_peak, weight_decay=cfg.weight_decay)

# Checkpoint dir
checkpoint_dir = Path("checkpoints_simple_cpu")
checkpoint_dir.mkdir(exist_ok=True)

# Training config
num_epochs = 10
batches_per_epoch = 50
batch_size = 8

best_loss = float('inf')

print("\n" + "=" * 70)
print(f"TRAINING: {num_epochs} epochs × {batches_per_epoch} batches")
print("=" * 70)

for epoch in range(1, num_epochs + 1):
    print(f"\n--- EPOCH {epoch}/{num_epochs} ---")
    model.train()

    epoch_losses = []

    for batch_idx in range(batches_per_epoch):
        # Generate random batch
        scans = torch.rand(batch_size, 60, 1, 24, 60, 128, device=device)
        nav = torch.rand(batch_size, 60, 7, device=device)
        valid = torch.ones(batch_size, 60, dtype=torch.bool, device=device)
        labels = torch.zeros(batch_size, 4, device=device)

        # Make some labels positive
        labels[torch.arange(batch_size) < batch_size // 2, :2] = 0.5

        # Forward pass
        optimizer.zero_grad()
        logits = model(scans, valid, nav)
        loss = loss_fn(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        epoch_losses.append(loss.item())

        if (batch_idx + 1) % 10 == 0:
            avg_loss = np.mean(epoch_losses[-10:])
            print(f"  Batch {batch_idx+1:3d}/{batches_per_epoch}: loss={avg_loss:.6f}")

    avg_epoch_loss = np.mean(epoch_losses)
    print(f"Epoch {epoch} average loss: {avg_epoch_loss:.6f}")

    # Save best checkpoint
    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        ckpt_path = checkpoint_dir / "best.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "loss": avg_epoch_loss,
        }, ckpt_path)
        print(f"  -> Saved best checkpoint: {ckpt_path}")

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
print(f"Best loss: {best_loss:.6f}")
print(f"Best checkpoint: {checkpoint_dir}/best.pt")
