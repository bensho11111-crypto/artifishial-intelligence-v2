#!/usr/bin/env python3
"""Minimal test to identify training bottleneck."""
import sys
import os
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ml.config import ModelConfig
from ml.model import FishCatchTransformer
from ml.dataset import FishCatchDataset
from torch.utils.data import DataLoader

print("1. Loading config...", flush=True)
cfg = ModelConfig()
print("   OK")

print("2. Loading dataset...", flush=True)
train_ds = FishCatchDataset("data/training_300/", cfg, augment=False, val_fraction=0.15, train=True)
print(f"   OK - {len(train_ds)} windows")

print("3. Creating DataLoader...", flush=True)
loader = DataLoader(train_ds, batch_size=4, shuffle=False, num_workers=0, pin_memory=False)
print("   OK")

print("4. Getting first batch...", flush=True)
batch = next(iter(loader))
print(f"   OK - batch keys: {list(batch.keys())}")
print(f"   scans shape: {batch['scans'].shape}")
print(f"   nav shape: {batch['nav'].shape}")
print(f"   label shape: {batch['label'].shape}")

print("5. Creating model...", flush=True)
device = torch.device("cpu")
model = FishCatchTransformer(cfg).to(device)
print(f"   OK - {sum(p.numel() for p in model.parameters()):,} params")

print("6. Forward pass...", flush=True)
with torch.no_grad():
    scans = batch["scans"].to(device)
    nav = batch["nav"].to(device)
    valid = batch["scan_valid"].to(device)
    logits = model(scans, valid, nav)
    print(f"   OK - output shape: {logits.shape}")

print("\nAll steps completed successfully!")
