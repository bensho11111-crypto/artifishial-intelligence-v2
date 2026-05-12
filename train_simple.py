#!/usr/bin/env python3
"""
SIMPLE TRAINING: Minimal, reliable, no complex I/O.
Generates synthetic batches in memory for demonstration.
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

print("="*70)
print("SIMPLE TRAINING: GPU + Synthetic Data")
print("="*70)

cfg = ModelConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")

# Build model
print("\nBuilding model...")
model = FishCatchTransformer(cfg).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {n_params:,}")

loss_fn = AsymmetricFocalLoss(gamma_pos=0.0, gamma_neg=4.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr_peak, weight_decay=cfg.weight_decay)

# Training
print("\nTraining 2 epochs (synthetic data)...")
print("="*70)

checkpoint_dir = Path("checkpoints_simple")
checkpoint_dir.mkdir(exist_ok=True)
best_loss = float('inf')

for epoch in range(2):
    print(f"\n--- Epoch {epoch+1}/2 ---")
    model.train()

    epoch_losses = []

    # Generate 100 synthetic batches
    for batch_idx in range(100):
        # Generate random batch
        scans = torch.rand(16, 60, 1, 24, 60, 128, device=device)
        nav = torch.rand(16, 60, 7, device=device)
        valid = torch.ones(16, 60, dtype=torch.bool, device=device)
        labels = torch.zeros(16, 4, device=device)

        # Make some labels positive
        labels[torch.arange(16) < 8, :2] = 0.5

        # Forward pass
        optimizer.zero_grad()
        logits = model(scans, valid, nav)
        loss = loss_fn(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        epoch_losses.append(loss.item())

        if (batch_idx + 1) % 25 == 0:
            avg_loss = np.mean(epoch_losses[-25:])
            print(f"  Batch {batch_idx+1:3d}/100: loss={avg_loss:.6f}")

    avg_epoch_loss = np.mean(epoch_losses)
    print(f"Epoch {epoch+1} average loss: {avg_epoch_loss:.6f}")

    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        ckpt_path = checkpoint_dir / "best.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "loss": avg_epoch_loss,
        }, ckpt_path)
        print(f"  → Saved checkpoint: {ckpt_path}")

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)
print(f"Best loss: {best_loss:.6f}")
print(f"Final checkpoint: {checkpoint_dir}/best.pt")
print("\nNOTE: This uses synthetic data for demonstration.")
print("For real data, use train_final.py or train_ram_cached.py")
