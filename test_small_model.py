#!/usr/bin/env python3
"""Test smaller model with improved hyperparameters on larger dataset."""
import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ml.config import ModelConfig
from ml.model import FishCatchTransformer
from ml.loss import AsymmetricFocalLoss
from ml.dataset import FishCatchDataset
from torch.utils.data import DataLoader, WeightedRandomSampler

print("=" * 70)
print("TESTING SMALLER MODEL WITH IMPROVED HYPERPARAMETERS")
print("=" * 70)

cfg = ModelConfig()
device = torch.device("cpu")

print(f"\nModel dimensions:")
print(f"  d_sonar={cfg.d_sonar}, d_nav={cfg.d_nav}, d_model={cfg.d_model}")
print(f"  n_heads={cfg.n_heads}, n_layers={cfg.n_layers}, dropout={cfg.dropout}")
print(f"Hyperparameters:")
print(f"  lr_peak={cfg.lr_peak}, weight_decay={cfg.weight_decay}, grad_clip={cfg.grad_clip}")

# Load dataset (larger one)
print(f"\nLoading dataset...")
train_ds = FishCatchDataset("data/training_300/", cfg, augment=False, val_fraction=0.15, train=True)
print(f"  Train samples: {len(train_ds)}")

# Weights
all_labels = train_ds._h5_group['labels'][:]
has_pos = (all_labels.max(axis=1) > 0).astype(float)
weights = np.where(has_pos > 0, 30.0, 1.0)
sampler = WeightedRandomSampler(weights, len(train_ds), replacement=True)

train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=0, pin_memory=False)

# Model
model = FishCatchTransformer(cfg).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"\nModel parameters: {n_params:,}")
print(f"  Ratio (params/samples): {n_params / len(train_ds):.1f}")

loss_fn = AsymmetricFocalLoss(gamma_pos=0.0, gamma_neg=4.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr_peak, weight_decay=cfg.weight_decay)

# Train 10 batches
print(f"\n" + "=" * 70)
print("TRAINING 10 BATCHES")
print("=" * 70)

losses = []
for batch_idx, batch in enumerate(train_loader):
    if batch_idx >= 10:
        break

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

    losses.append(loss.item())
    print(f"Batch {batch_idx:2d}: loss={loss.item():.6f}")

# Analyze
print(f"\n" + "=" * 70)
print("ANALYSIS")
print("=" * 70)
initial_loss = losses[0]
final_loss = losses[-1]
improvement = (initial_loss - final_loss) / initial_loss * 100

print(f"Initial loss: {initial_loss:.6f}")
print(f"Final loss:   {final_loss:.6f}")
print(f"Improvement:  {improvement:.2f}%")

if improvement > 5:
    print(f"[OK] Model is learning! Loss decreased by {improvement:.1f}%")
elif improvement > 0:
    print(f"[OK] Model is learning but slowly ({improvement:.1f}% improvement)")
else:
    print(f"[FAIL] Model is NOT learning (loss increased by {-improvement:.1f}%)")

# Check min/max
min_loss = min(losses)
max_loss = max(losses)
print(f"\nMin loss: {min_loss:.6f}, Max loss: {max_loss:.6f}")
print(f"Volatility: {max_loss - min_loss:.6f}")
