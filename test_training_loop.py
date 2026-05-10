#!/usr/bin/env python3
"""Test if training loop runs without hanging."""
import sys
import os
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ml.config import ModelConfig
from ml.model import FishCatchTransformer
from ml.loss import AsymmetricFocalLoss
from ml.dataset import FishCatchDataset
from ml.train import train_epoch, eval_epoch
from torch.utils.data import DataLoader, WeightedRandomSampler

print("1. Setup...", flush=True)
cfg = ModelConfig()
device = torch.device("cpu")

# Load small datasets
train_ds = FishCatchDataset("data/training_300/", cfg, augment=True, val_fraction=0.15, train=True)
val_ds = FishCatchDataset("data/training_300/", cfg, augment=False, val_fraction=0.15, train=False)

# Use weighted sampler (like the real training)
print(f"   Train: {len(train_ds)}, Val: {len(val_ds)}")
print(f"2. Computing weights...", flush=True)
if train_ds._use_h5 and train_ds._h5_group is not None:
    all_labels = train_ds._h5_group['labels'][:]
    has_pos = (all_labels.max(axis=1) > 0).astype(float)
    weights = np.where(has_pos > 0, 30.0, 1.0)
else:
    weights = np.ones(len(train_ds))

sampler = WeightedRandomSampler(weights, len(train_ds), replacement=True)

# Small dataloaders
train_loader = DataLoader(
    train_ds, batch_size=4, sampler=sampler, num_workers=0, pin_memory=False
)
val_loader = DataLoader(
    val_ds, batch_size=4, shuffle=False, num_workers=0, pin_memory=False
)

# Create model and loss
print(f"3. Creating model...", flush=True)
model = FishCatchTransformer(cfg).to(device)
loss_fn = AsymmetricFocalLoss(gamma_pos=0.0, gamma_neg=4.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr_peak)

# Train one epoch
print(f"4. Training one epoch...", flush=True)
try:
    train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device, grad_clip=1.0)
    print(f"   Train loss: {train_loss:.4f}")
except Exception as e:
    print(f"   ERROR during training: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Eval one epoch
print(f"5. Evaluating...", flush=True)
try:
    val_loss, val_auroc, val_ap, aurocs, aps = eval_epoch(model, val_loader, loss_fn, device)
    print(f"   Val loss: {val_loss:.4f}, AUROC: {val_auroc:.3f}, AP: {val_ap:.3f}")
except Exception as e:
    print(f"   ERROR during evaluation: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print(f"\nTraining loop works!")
