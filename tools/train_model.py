"""
tools/train_model.py

CLI wrapper for training FishCatchTransformer on synthetic data.

Usage:
    python tools/train_model.py \\
        --data data/training/ \\
        --out checkpoints/ \\
        --epochs 30 \\
        --batch 32
"""
import argparse
import sys
import os
import dataclasses
import random
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ml.config import ModelConfig
from ml.model import FishCatchTransformer
from ml.loss import AsymmetricFocalLoss
from ml.dataset import FishCatchDataset
from ml.train import train_epoch, eval_epoch


def main():
    parser = argparse.ArgumentParser(
        description="Train FishCatchTransformer on synthetic dataset"
    )
    parser.add_argument("--data", required=True, help="Directory with .ticks + _catches.json pairs")
    parser.add_argument("--out", required=True, help="Output checkpoint directory")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--device", default=None, help="Device (cpu or cuda, default auto)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Device selection
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")

    # Load config
    cfg = ModelConfig()

    # Create dataset and dataloaders
    print(f"Loading dataset from {args.data}...")
    train_ds = FishCatchDataset(args.data, cfg, augment=True, val_fraction=cfg.val_fraction, train=True)
    val_ds = FishCatchDataset(args.data, cfg, augment=False, val_fraction=cfg.val_fraction, train=False)

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # Compute sample weights for balanced training
    # Positive windows get 30x weight to increase their frequency in batches
    weights = []
    for i in range(len(train_ds)):
        label = train_ds[i]["label"]
        if label.max() > 0:
            weights.append(30.0)  # Positive window — boost frequency
        else:
            weights.append(1.0)   # Negative window

    sampler = WeightedRandomSampler(weights, len(train_ds), replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        sampler=sampler,
        num_workers=0,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=0,  # Windows multiprocessing issue; use serial loading
        pin_memory=False
    )

    # Create model
    model = FishCatchTransformer(cfg).to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Loss function
    loss_fn = AsymmetricFocalLoss(gamma_pos=0.0, gamma_neg=4.0)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=cfg.lr_peak, weight_decay=cfg.weight_decay)

    # Learning rate scheduler
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    scheduler = OneCycleLR(
        optimizer,
        max_lr=cfg.lr_peak,
        total_steps=total_steps,
        pct_start=0.3
    )

    # Training loop
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    best_auroc = 0.0

    for epoch in range(args.epochs):
        # Training
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device, grad_clip=cfg.grad_clip)

        # Validation
        val_loss, val_auroc, val_ap, aurocs, aps = eval_epoch(model, val_loader, loss_fn, device)

        # Step scheduler
        scheduler.step()

        # Print progress
        print(f"Epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.3f}  val_auroc={val_auroc:.3f}  val_ap={val_ap:.3f}")

        # Save best checkpoint
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": dataclasses.asdict(cfg),
                "species": list(cfg.species),
                "val_auroc": val_auroc,
                "val_ap": val_ap,
            }
            checkpoint_path = out_path / "best.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"  -> Saved best checkpoint (auroc={val_auroc:.3f})")

    print(f"\nTraining complete. Best AUROC: {best_auroc:.3f}")

    # Save final model if no best checkpoint was saved
    if best_auroc <= 0.0:
        print("  -> No valid AUROC found in validation. Saving final model...")
        checkpoint = {
            "epoch": args.epochs - 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": dataclasses.asdict(cfg),
            "species": list(cfg.species),
            "val_auroc": 0.0,
            "val_ap": 0.0,
            "note": "Saved as final model (validation metrics unavailable)",
        }
        checkpoint_path = out_path / "best.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"  -> Saved final checkpoint to {checkpoint_path}")
    else:
        print(f"Checkpoint saved to {out_path / 'best.pt'}")


if __name__ == "__main__":
    main()
