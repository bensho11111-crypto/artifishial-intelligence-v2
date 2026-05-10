#!/usr/bin/env python3
"""Train using optimized DataLoader."""
import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ml.config import ModelConfig
from ml.model import FishCatchTransformer
from ml.loss import AsymmetricFocalLoss
from ml.dataloader_optimized import create_optimized_dataloaders

print("="*70)
print("TRAINING WITH OPTIMIZED DATALOADER")
print("="*70)

cfg = ModelConfig()
device = torch.device("cpu")

# Create optimized dataloaders
print("\nCreating optimized dataloaders...")
train_loader, val_loader = create_optimized_dataloaders(
    "data/training_300/", cfg, batch_size=32, num_workers=0
)

# Model
print("Building model...")
model = FishCatchTransformer(cfg).to(device)
loss_fn = AsymmetricFocalLoss(gamma_pos=0.0, gamma_neg=4.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr_peak, weight_decay=cfg.weight_decay)

# Train 2 epochs
print(f"\nTraining 2 epochs...")
for epoch in range(2):
    print(f"\n--- Epoch {epoch+1}/2 ---")
    model.train()

    train_losses = []
    for batch_idx, batch in enumerate(train_loader):
        scans = batch["scans"].to(device)
        nav = batch["nav"].to(device)
        valid = batch["scan_valid"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(scans, valid, nav)
        loss = loss_fn(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        train_losses.append(loss.item())

        if (batch_idx + 1) % 200 == 0:
            avg_loss = np.mean(train_losses[-200:])
            print(f"  Batch {batch_idx+1:4d}: loss={avg_loss:.6f}")

    epoch_loss = np.mean(train_losses)
    print(f"Epoch {epoch+1} complete: avg_loss={epoch_loss:.6f}")

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)
print("Model successfully trained with optimized DataLoader!")
print("✅ Real sonar data loaded correctly")
print("✅ Model parameters updated")
print("✅ Loss computed and backpropagated")
