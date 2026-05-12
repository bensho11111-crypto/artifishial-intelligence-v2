#!/usr/bin/env python3
"""Training with synchronized output."""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, "src")

from ml.config import ModelConfig
from ml.model import FishCatchTransformer
from ml.loss import AsymmetricFocalLoss
import dataclasses

print("=" * 70)
print("TRAINING: SYNTHETIC DATA ON CPU")
print("=" * 70)
sys.stdout.flush()

cfg = ModelConfig()
device = "cpu"
print(f"Device: {device}")
sys.stdout.flush()

print("\nBuilding model...")
sys.stdout.flush()
model = FishCatchTransformer(cfg).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {n_params:,}")
sys.stdout.flush()

loss_fn = AsymmetricFocalLoss(gamma_pos=0.0, gamma_neg=4.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr_peak, weight_decay=cfg.weight_decay)

checkpoint_dir = Path("checkpoints_simple_cpu")
checkpoint_dir.mkdir(exist_ok=True)

num_epochs = 3
batches_per_epoch = 10
batch_size = 4

best_loss = float('inf')

print(f"\nTraining: {num_epochs} epochs × {batches_per_epoch} batches")
print("=" * 70)
sys.stdout.flush()

for epoch in range(1, num_epochs + 1):
    print(f"\n--- EPOCH {epoch}/{num_epochs} ---")
    sys.stdout.flush()
    model.train()

    epoch_losses = []

    for batch_idx in range(batches_per_epoch):
        scans = torch.rand(batch_size, 60, 1, 24, 60, 128, device=device)
        nav = torch.rand(batch_size, 60, 7, device=device)
        valid = torch.ones(batch_size, 60, dtype=torch.bool, device=device)
        labels = torch.zeros(batch_size, 4, device=device)

        labels[torch.arange(batch_size) < batch_size // 2, :2] = 0.5

        optimizer.zero_grad()
        logits = model(scans, valid, nav)
        loss = loss_fn(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        epoch_losses.append(loss.item())

        if (batch_idx + 1) % 5 == 0:
            avg_loss = np.mean(epoch_losses[-5:])
            print(f"  Batch {batch_idx+1:2d}/{batches_per_epoch}: loss={avg_loss:.6f}")
            sys.stdout.flush()

    avg_epoch_loss = np.mean(epoch_losses)
    print(f"  Epoch avg loss: {avg_epoch_loss:.6f}")
    sys.stdout.flush()

    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        ckpt_path = checkpoint_dir / "best.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "config": dataclasses.asdict(cfg),
            "loss": avg_epoch_loss,
        }, ckpt_path)
        print(f"  -> Saved: {ckpt_path}")
        sys.stdout.flush()

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print(f"Best loss: {best_loss:.6f}")
print(f"Checkpoint: {checkpoint_dir}/best.pt")
print("=" * 70)
sys.stdout.flush()
