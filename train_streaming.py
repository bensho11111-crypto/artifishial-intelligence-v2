#!/usr/bin/env python3
"""
Streaming training: generate batches on-the-fly while training without pre-generating dataset.
Minimal disk footprint (~1 GB max queue), 4× faster than single-process.
"""

import argparse
import dataclasses
import multiprocessing
import numpy as np
import sys
import time
import torch
import torch.nn as nn
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, "src")

from ml.config import ModelConfig
from ml.model import FishCatchTransformer
from ml.loss import AsymmetricFocalLoss
from ml.train import train_epoch, eval_epoch
from ml.streaming_generator import (
    StreamingDataLoader,
    FixedNpzLoader,
    generate_sample_numpy,
    worker_main,
)


def generate_validation_set(n_samples, batch_size=32, device="cpu"):
    """Generate and save a fixed validation set (synchronous)."""
    val_path = Path("data/val_streaming.npz")
    if val_path.exists():
        print(f"Validation set already exists: {val_path}")
        return val_path

    print(f"Generating {n_samples} validation samples...")
    start = time.time()

    scans_list = []
    valids_list = []
    navs_list = []
    labels_list = []

    for i in range(n_samples):
        if i % 10 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  {i+1:4d}/{n_samples}: {rate:.2f} samples/sec")

        sample = generate_sample_numpy(i)
        scans_list.append(sample["scans"])
        valids_list.append(sample["valids"])
        navs_list.append(sample["navs"])
        labels_list.append(sample["labels"])

    # Stack into batch arrays and save
    val_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(val_path),
        scans=np.stack(scans_list, axis=0),
        valids=np.stack(valids_list, axis=0),
        navs=np.stack(navs_list, axis=0),
        labels=np.stack(labels_list, axis=0),
    )

    elapsed = time.time() - start
    print(f"Validation set saved: {val_path} ({elapsed:.0f}s)")
    return val_path


def main():
    parser = argparse.ArgumentParser(
        description="Stream training: on-the-fly generation without pre-generation"
    )
    parser.add_argument("--workers", type=int, default=4, help="Number of generator workers")
    parser.add_argument(
        "--epochs", type=int, default=30, help="Training epochs"
    )
    parser.add_argument(
        "--steps", type=int, default=50, help="Training steps per epoch"
    )
    parser.add_argument("--batch", type=int, default=32, help="Training batch size")
    parser.add_argument(
        "--queue-depth", type=int, default=16, help="Max batch files in queue"
    )
    parser.add_argument(
        "--val-samples", type=int, default=64, help="Validation set size"
    )
    parser.add_argument(
        "--out", type=str, default="checkpoints_streaming", help="Output checkpoint dir"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint",
    )
    parser.add_argument("--device", type=str, default="cpu", help="torch device")
    args = parser.parse_args()

    # Setup
    multiprocessing.freeze_support()
    device = torch.device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    queue_dir = Path("data/gen_queue")

    # Clean queue dir (ignore locked files from previous runs)
    queue_dir.mkdir(parents=True, exist_ok=True)
    for f in queue_dir.glob("*.npz"):
        try:
            f.unlink()
        except (OSError, PermissionError):
            pass  # File locked by previous worker, will be overwritten
    for f in queue_dir.glob("*.tmp"):
        try:
            f.unlink()
        except (OSError, PermissionError):
            pass

    print("=" * 70)
    print("STREAMING TRAINING")
    print("=" * 70)
    print(f"Workers: {args.workers}")
    print(f"Epochs: {args.epochs}, Steps/epoch: {args.steps}, Batch: {args.batch}")
    print(f"Val samples: {args.val_samples}, Queue depth: {args.queue_depth}")
    print(f"Output: {out_dir}")
    print()

    # Generate validation set (synchronous)
    val_path = generate_validation_set(args.val_samples, device=args.device)

    # Build model
    cfg = ModelConfig()
    model = FishCatchTransformer(cfg).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    loss_fn = AsymmetricFocalLoss(gamma_pos=0.0, gamma_neg=4.0, clip=0.05)
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    total_steps = args.epochs * args.steps
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)

    # Load checkpoint if resuming
    start_epoch = 1
    best_val_auroc = -1.0
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_auroc = ckpt.get("val_mean_auroc", -1.0)

    # Start worker processes
    stop_event = multiprocessing.Event()
    workers = []
    print(f"Starting {args.workers} generator workers...")
    for worker_id in range(args.workers):
        p = multiprocessing.Process(
            target=worker_main,
            args=(
                worker_id,
                args.workers,
                queue_dir,
                8,  # batch_size per worker output file
                args.queue_depth,
                stop_event,
            ),
        )
        p.start()
        workers.append(p)

    # Give workers a moment to start
    time.sleep(1)

    print()

    # Training loop
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            # Training
            train_loader = StreamingDataLoader(
                queue_dir,
                n_steps=args.steps,
                batch_size=args.batch,
                device=args.device,
            )
            train_loss = train_epoch(
                model,
                train_loader,
                loss_fn,
                optimizer,
                device,
                grad_clip=1.0,
            )

            # Validation
            val_loader = FixedNpzLoader(
                val_path,
                n_steps=2,  # Just 2 batches for quick eval
                batch_size=args.batch,
                device=args.device,
            )
            val_loss, val_auroc, val_ap, _, _ = eval_epoch(
                model, val_loader, loss_fn, device
            )

            print(
                f"Epoch {epoch:2d}/{args.epochs}  "
                f"loss={train_loss:.4f}  val_auroc={val_auroc:.4f}  "
                f"queue={len(list(queue_dir.glob('*.npz')))} files"
            )

            # Checkpoint if best (or first epoch, or nan val_auroc)
            is_best = (np.isnan(val_auroc) and np.isnan(best_val_auroc)) or (
                not np.isnan(val_auroc) and val_auroc > best_val_auroc
            )
            if is_best or epoch == start_epoch:
                if not np.isnan(val_auroc):
                    best_val_auroc = val_auroc
                ckpt_path = out_dir / "best.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": dataclasses.asdict(cfg),
                        "species": list(cfg.species),
                        "horizon_s": cfg.horizon_s,
                        "val_mean_auroc": val_auroc,
                    },
                    ckpt_path,
                )

            scheduler.step()

    except KeyboardInterrupt:
        print("\nInterrupted, shutting down...")
    finally:
        # Cleanup
        stop_event.set()
        for p in workers:
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()
        print("Done.")


if __name__ == "__main__":
    main()
