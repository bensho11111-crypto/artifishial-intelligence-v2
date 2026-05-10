#!/usr/bin/env python3
"""Debug if weights are actually updating during training."""
import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ml.config import ModelConfig
from ml.model import FishCatchTransformer
from ml.loss import AsymmetricFocalLoss
from ml.dataset import FishCatchDataset
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

print("Testing weight updates during training...\n")

# Setup
cfg = ModelConfig()
device = torch.device("cpu")

# Load dataset
train_ds = FishCatchDataset("data/training_300/", cfg, augment=False, val_fraction=0.15, train=True)

# Compute weights
all_labels = train_ds._h5_group['labels'][:]
has_pos = (all_labels.max(axis=1) > 0).astype(float)
weights = np.where(has_pos > 0, 30.0, 1.0)
sampler = WeightedRandomSampler(weights, len(train_ds), replacement=True)

# Small loader for testing
train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=0, pin_memory=False)

# Model and training
model = FishCatchTransformer(cfg).to(device)
loss_fn = AsymmetricFocalLoss(gamma_pos=0.0, gamma_neg=4.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr_peak, weight_decay=cfg.weight_decay)

# Store initial weights
initial_weight = model.local_transformer.layers[0].self_attn.in_proj_weight.data.clone()

print(f"Initial weight sum: {initial_weight.sum():.6f}")
print(f"Learning rate: {cfg.lr_peak}")
print(f"Batch size: 32, Batches per epoch: {len(train_loader)}\n")

# Train one batch
batch = next(iter(train_loader))
scans = batch["scans"].to(device)
nav = batch["nav"].to(device)
valid = batch["scan_valid"].to(device)
labels = batch["label"].to(device)

print("Before backward pass:")
print(f"  Model output range: [{model(scans, valid, nav).min():.4f}, {model(scans, valid, nav).max():.4f}]")

optimizer.zero_grad()
logits = model(scans, valid, nav)
loss = loss_fn(logits, labels)

print(f"  Loss: {loss.item():.6f}")
print(f"  Loss requires_grad: {loss.requires_grad}")

# Check gradients before backward
print("\nComputing gradients...")
loss.backward()

# Check if gradients exist
has_grads = 0
total_params = 0
for name, param in model.named_parameters():
    if param.grad is not None:
        has_grads += 1
    total_params += 1

print(f"  Parameters with gradients: {has_grads}/{total_params}")

# Check gradient magnitudes
grad_sum = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
print(f"  Total gradient magnitude: {grad_sum:.6f}")

# Optimizer step
print("\nApplying optimizer step...")
optimizer.step()

# Check if weights changed
updated_weight = model.local_transformer.layers[0].self_attn.in_proj_weight.data
weight_change = (updated_weight - initial_weight).abs().sum().item()

print(f"\nAfter optimizer step:")
print(f"  Updated weight sum: {updated_weight.sum():.6f}")
print(f"  Weight change magnitude: {weight_change:.6f}")

if weight_change > 1e-6:
    print(f"  [OK] WEIGHTS UPDATED (change: {weight_change:.2e})")
else:
    print(f"  [FAIL] WEIGHTS NOT UPDATED (change < 1e-6)")

# Train 5 full batches
print("\n" + "="*60)
print("Training 5 batches to see loss trend...")
print("="*60)

losses = []
for batch_idx, batch in enumerate(train_loader):
    if batch_idx >= 5:
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
    print(f"Batch {batch_idx}: loss={loss.item():.6f}")

if len(losses) > 1:
    improvement = (losses[0] - losses[-1]) / losses[0] * 100
    print(f"\nLoss change: {losses[0]:.6f} → {losses[-1]:.6f} ({improvement:.2f}%)")
    if improvement > 0:
        print("[OK] Loss is decreasing")
    else:
        print("[FAIL] Loss is NOT decreasing")
