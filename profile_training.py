#!/usr/bin/env python3
"""Profile training bottlenecks."""
import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.config import ModelConfig
from ml.model import FishCatchTransformer
from ml.loss import AsymmetricFocalLoss
from ml.dataset import FishCatchDataset

print("=" * 60)
print("TRAINING PROFILING")
print("=" * 60)

cfg = ModelConfig()
device = torch.device("cpu")

# Load dataset
print("\n1. Dataset loading")
start = time.time()
ds = FishCatchDataset("data/training/", cfg, augment=True)
ds_time = time.time() - start
print(f"   Time: {ds_time:.2f}s")
print(f"   Samples: {len(ds)}")

# Create DataLoader
print("\n2. DataLoader creation")
loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)
batch_count = len(loader)
print(f"   Batches: {batch_count}")

# Profile one batch
print("\n3. Single batch (forward + backward)")
model = FishCatchTransformer(cfg).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
loss_fn = AsymmetricFocalLoss(gamma_pos=0.0, gamma_neg=4.0)

batch = next(iter(loader))
scans = batch["scans"].to(device)
scan_valid = batch["scan_valid"].to(device)
nav = batch["nav"].to(device)
labels = batch["label"].to(device)

# Forward
start = time.time()
logits = model(scans, scan_valid, nav)
forward_time = time.time() - start
print(f"   Forward: {forward_time*1000:.1f}ms ({batch[0].shape[0]} batch size)")

# Loss
start = time.time()
loss = loss_fn(logits, labels)
loss_time = time.time() - start
print(f"   Loss: {loss_time*1000:.1f}ms")

# Backward
start = time.time()
loss.backward()
back_time = time.time() - start
print(f"   Backward: {back_time*1000:.1f}ms")

# Optimizer step
start = time.time()
optimizer.step()
optimizer.zero_grad()
step_time = time.time() - start
print(f"   Step: {step_time*1000:.1f}ms")

# Summary
print("\n" + "=" * 60)
print("TRAINING ESTIMATE (10 epochs)")
print("=" * 60)

batch_time = forward_time + loss_time + back_time + step_time
epoch_time = batch_time * batch_count
total_time = epoch_time * 10

print(f"\nPer batch: {batch_time*1000:.1f}ms")
print(f"Per epoch: {epoch_time:.1f}s ({epoch_time/60:.1f}min)")
print(f"Total (10 epochs): {total_time:.0f}s ({total_time/60:.1f}min)")

print("\n" + "=" * 60)
print("OPTIMIZATION OPPORTUNITIES")
print("=" * 60)
print("\n✓ Use GPU (if available) - 10-20x speedup")
print("✓ Use mixed precision - 2-4x speedup")
print("✓ Increase batch size - Better GPU utilization")
print("✓ Num_workers > 0 - Faster DataLoader")
