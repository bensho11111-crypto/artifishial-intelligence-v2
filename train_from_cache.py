#!/usr/bin/env python3
"""
Fast training from cached synthetic dataset.

Usage:
    # First generate cache (one time, ~2 hours)
    python tools/generate_synthetic_dataset.py --output data/synthetic_cache.h5 --n-samples 24000 --workers 4

    # Then train (fast iterations, ~3-5 min per epoch)
    python train_from_cache.py --cache data/synthetic_cache.h5 --epochs 30
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import dataclasses

sys.path.insert(0, "src")

from ml.config import ModelConfig
from ml.model import FishCatchTransformer
from ml.loss import AsymmetricFocalLoss
from ml.cached_dataset import get_cached_dataloader
import argparse


def main():
    parser = argparse.ArgumentParser(description="Train from cached synthetic dataset")
    parser.add_argument("--cache", default="data/synthetic_cache.h5", help="Path to HDF5 cache")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--out", default="checkpoints_cached", help="Output checkpoint directory")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    DEVICE = args.device
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LR = args.lr
    CHECKPOINT_DIR = Path(args.out)
    CACHE_PATH = args.cache

    print("=" * 70)
    print("TRAINING FROM CACHED SYNTHETIC DATASET")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Cache: {CACHE_PATH}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LR}")
    print()
    sys.stdout.flush()

    # Build model
    print("Building model...")
    sys.stdout.flush()
    cfg = ModelConfig()
    model = FishCatchTransformer(cfg).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    sys.stdout.flush()

    # Loss and optimizer
    loss_fn = AsymmetricFocalLoss(gamma_pos=0.0, gamma_neg=4.0, clip=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=cfg.weight_decay)

    # Checkpoint directory
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    # Load cached dataset
    print(f"\nLoading cached dataset from {CACHE_PATH}...")
    sys.stdout.flush()
    train_loader = get_cached_dataloader(
        CACHE_PATH,
        batch_size=BATCH_SIZE,
        num_workers=2,
        shuffle=True,
        augment=True,
    )
    print(f"  Loaded {len(train_loader)} batches")
    sys.stdout.flush()

    best_loss = float("inf")

    print(f"\nTraining...")
    print("=" * 70)
    sys.stdout.flush()

    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- EPOCH {epoch}/{EPOCHS} ---", flush=True)
        model.train()

        epoch_losses = []

        for batch_idx, batch in enumerate(train_loader):
            scans = batch["scans"].to(DEVICE)
            valids = batch["valids"].to(DEVICE)
            navs = batch["navs"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            # Forward pass
            optimizer.zero_grad()
            logits = model(scans, valids, navs)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            epoch_losses.append(loss.item())

            if (batch_idx + 1) % 10 == 0:
                avg_loss = np.mean(epoch_losses[-10:])
                print(
                    f"  Batch {batch_idx + 1:3d}/{len(train_loader)}: loss={avg_loss:.6f}",
                    flush=True,
                )

        avg_epoch_loss = np.mean(epoch_losses)
        print(f"  Epoch avg loss: {avg_epoch_loss:.6f}", flush=True)

        # Save checkpoint if best
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            ckpt_path = CHECKPOINT_DIR / "best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "config": dataclasses.asdict(cfg),
                    "species": list(cfg.species),
                    "horizon_s": cfg.horizon_s,
                    "loss": avg_epoch_loss,
                },
                ckpt_path,
            )
            print(f"  -> Saved: {ckpt_path}", flush=True)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Checkpoint: {CHECKPOINT_DIR}/best.pt")
    print("=" * 70)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
