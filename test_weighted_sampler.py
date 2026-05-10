#!/usr/bin/env python3
"""Test WeightedRandomSampler with HDF5 dataset."""
import sys
import os
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ml.config import ModelConfig
from ml.dataset import FishCatchDataset
from torch.utils.data import DataLoader, WeightedRandomSampler

print("1. Loading dataset...", flush=True)
cfg = ModelConfig()
train_ds = FishCatchDataset("data/training_300/", cfg, augment=False, val_fraction=0.15, train=True)
print(f"   OK - {len(train_ds)} windows")

print("2. Computing weights...", flush=True)
if train_ds._use_h5 and train_ds._h5_group is not None:
    all_labels = train_ds._h5_group['labels'][:]
    has_pos = (all_labels.max(axis=1) > 0).astype(float)
    weights = np.where(has_pos > 0, 30.0, 1.0)
else:
    weights = np.ones(len(train_ds))
print(f"   OK - {(weights > 1).sum()} positive samples")

print("3. Creating WeightedRandomSampler...", flush=True)
sampler = WeightedRandomSampler(weights, len(train_ds), replacement=True)
print(f"   OK")

print("4. Creating DataLoader...", flush=True)
loader = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=0, pin_memory=False)
print(f"   OK - {len(loader)} batches")

print("5. Iterating first 3 batches...", flush=True)
for i, batch in enumerate(loader):
    print(f"   Batch {i}: shapes {batch['scans'].shape}, {batch['nav'].shape}, {batch['label'].shape}", flush=True)
    if i >= 2:
        break

print("\nWeightedRandomSampler works!")
