#!/usr/bin/env python3
"""Final practical training: GPU with reasonable batch size."""
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
print("FISH CATCH TRANSFORMER: GPU TRAINING")
print("="*70)

cfg = ModelConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Dataloaders
print("\nPreparing data...")
train_loader, val_loader = create_optimized_dataloaders(
    "data/training_300/", cfg, batch_size=8, num_workers=0
)

# Model
print("Building model...")
model = FishCatchTransformer(cfg).to(device)
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

loss_fn = AsymmetricFocalLoss(gamma_pos=0.0, gamma_neg=4.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr_peak, weight_decay=cfg.weight_decay)

# Training
print("\nTraining 1 epoch (demo)...")
print("="*70)

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
        print(f"Batch {batch_idx+1:5d}/{len(train_loader)}: loss={avg_loss:.6f}")

    # Stop after first 500 batches for demo
    if batch_idx >= 499:
        break

epoch_loss = np.mean(train_losses)
print(f"\nDemo training complete: avg_loss={epoch_loss:.6f}")
print("="*70)

# Save checkpoint
checkpoint_dir = Path("checkpoints_gpu")
checkpoint_dir.mkdir(exist_ok=True)
ckpt_path = checkpoint_dir / "latest.pt"
torch.save({
    "model_state_dict": model.state_dict(),
    "loss": epoch_loss,
}, ckpt_path)
print(f"Checkpoint saved: {ckpt_path}")

print("\nNotes:")
print("- Batch size 8 for stability")
print("- First 500 batches (~6,250 windows) trained")
print("- For full training: increase num_batches or remove the break condition")
print("- Full 5-epoch training will take several hours due to I/O")
