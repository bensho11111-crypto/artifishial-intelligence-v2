#!/usr/bin/env python3
"""
CONTINUOUS TRAINING + CAPTAIN EVALUATION (CPU MODE)

Uses CPU instead of CUDA to avoid Windows TransformerEncoder hanging issues.
Trains model on synthetic data and evaluates performance via captain agents.
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ml.config import ModelConfig
from ml.model import FishCatchTransformer
from ml.loss import AsymmetricFocalLoss
from eval.metrics import evaluate_model

print("=" * 70)
print("CONTINUOUS TRAINING + CAPTAIN EVALUATION (CPU MODE)")
print("=" * 70)

cfg = ModelConfig()
device = "cpu"  # Force CPU to avoid CUDA transformer hanging
print(f"\nDevice: {device}")
print("(Using CPU mode to avoid Windows CUDA transformer hangs)")

# Build model
print("\nBuilding model...")
model = FishCatchTransformer(cfg).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {n_params:,}")

loss_fn = AsymmetricFocalLoss(gamma_pos=0.0, gamma_neg=4.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr_peak, weight_decay=cfg.weight_decay)

# Checkpoint dir
checkpoint_dir = Path("checkpoints_eval_cpu")
checkpoint_dir.mkdir(exist_ok=True)
eval_report_dir = Path("eval_reports_cpu")
eval_report_dir.mkdir(exist_ok=True)

# Training config
num_epochs = 5  # Shorter for faster iteration
eval_interval = 1  # Evaluate every epoch on CPU (faster)
batches_per_epoch = 50  # Smaller for CPU speed
batch_size = 8  # Smaller batch for CPU

# Tracking
best_captain_score = 0.0
training_history = []

print("\n" + "=" * 70)
print(f"TRAINING SCHEDULE: {num_epochs} epochs, eval every {eval_interval} epochs")
print(f"Batches/epoch: {batches_per_epoch}, Batch size: {batch_size}")
print("=" * 70)

for epoch in range(1, num_epochs + 1):
    print(f"\n--- EPOCH {epoch}/{num_epochs} ---")
    model.train()

    epoch_losses = []

    # Generate and train on synthetic batches
    for batch_idx in range(batches_per_epoch):
        # Generate random batch
        scans = torch.rand(batch_size, 60, 1, 24, 60, 128, device=device)
        nav = torch.rand(batch_size, 60, 7, device=device)
        valid = torch.ones(batch_size, 60, dtype=torch.bool, device=device)
        labels = torch.zeros(batch_size, 4, device=device)

        # Make some labels positive
        labels[torch.arange(batch_size) < batch_size // 2, :2] = 0.5

        # Forward pass
        optimizer.zero_grad()
        logits = model(scans, valid, nav)
        loss = loss_fn(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        epoch_losses.append(loss.item())

        if (batch_idx + 1) % 10 == 0:
            avg_loss = np.mean(epoch_losses[-10:])
            print(f"  Batch {batch_idx+1:3d}/{batches_per_epoch}: loss={avg_loss:.6f}")

    avg_epoch_loss = np.mean(epoch_losses)
    print(f"Epoch {epoch} average loss: {avg_epoch_loss:.6f}")

    # Evaluate every N epochs
    if epoch % eval_interval == 0:
        print(f"\n[EVALUATING] Running captain agent evaluation...")
        ckpt_path = checkpoint_dir / f"epoch_{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "loss": avg_epoch_loss,
        }, ckpt_path)
        print(f"  Saved checkpoint: {ckpt_path}")

        # Run evaluation (shorter episodes for CPU speed)
        print(f"  Running 3 episodes per captain (30s each)...")
        try:
            report = evaluate_model(
                model_path=str(ckpt_path),
                n_episodes=3,
                duration_s=30,
                seeds=[epoch, epoch+1, epoch+2],
            )

            # Extract metrics
            captain_score = report.captain_score
            oracle_frac = report.oracle_fraction
            model_guided_mean = report.mean_catches.get("ModelGuidedCaptain", 0.0)
            random_mean = report.mean_catches.get("RandomCaptain", 0.0)
            oracle_mean = report.mean_catches.get("OracleCaptain", 0.0)

            history_entry = {
                "epoch": epoch,
                "train_loss": avg_epoch_loss,
                "captain_score": float(captain_score) if captain_score != float("inf") else "inf",
                "oracle_fraction": float(oracle_frac) if oracle_frac != float("inf") else "inf",
                "model_guided_mean": model_guided_mean,
                "random_mean": random_mean,
                "oracle_mean": oracle_mean,
            }
            training_history.append(history_entry)

            # Print summary
            print(f"\n  === EVAL SUMMARY (Epoch {epoch}) ===")
            print(f"  Train loss:        {avg_epoch_loss:.6f}")
            print(f"  Random baseline:   {random_mean:.2f} catches/episode")
            print(f"  Straight baseline: {report.mean_catches.get('StraightCaptain', 0.0):.2f} catches/episode")
            print(f"  Model-guided:      {model_guided_mean:.2f} catches/episode")
            print(f"  Oracle (perfect):  {oracle_mean:.2f} catches/episode")
            print(f"  ")
            if captain_score == float("inf"):
                print(f"  Captain Score:     inf (Random caught 0, model caught > 0)")
            else:
                print(f"  Captain Score:     {captain_score:.3f} (ModelGuided / Random)")
            print(f"  Oracle Fraction:   {oracle_frac:.3f} (ModelGuided / Oracle)")

            # Save report
            report_path = eval_report_dir / f"epoch_{epoch:03d}.json"
            with open(report_path, "w") as f:
                json.dump({
                    "epoch": epoch,
                    "train_loss": avg_epoch_loss,
                    "captain_score": float(captain_score) if captain_score != float("inf") else "inf",
                    "oracle_fraction": float(oracle_frac) if oracle_frac != float("inf") else "inf",
                    "mean_catches": report.mean_catches,
                }, f, indent=2)

            # Track best model by captain_score
            if captain_score != float("inf") and captain_score > best_captain_score:
                best_captain_score = captain_score
                best_ckpt = checkpoint_dir / "best_by_captain_score.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "loss": avg_epoch_loss,
                    "captain_score": captain_score,
                    "oracle_fraction": oracle_frac,
                }, best_ckpt)
                print(f"  -> NEW BEST: saved to {best_ckpt}")
        except Exception as e:
            print(f"  ERROR during evaluation: {e}")
            import traceback
            traceback.print_exc()

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

# Save training history
history_path = eval_report_dir / "training_history.json"
with open(history_path, "w") as f:
    json.dump(training_history, f, indent=2)
print(f"Training history: {history_path}")

if checkpoint_dir.exists():
    best_files = list(checkpoint_dir.glob("best_by_captain_score.pt"))
    if best_files:
        print(f"Best checkpoint: {best_files[0]}")
        print(f"Best captain_score: {best_captain_score:.3f}")

# Print final summary table
if training_history:
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"{'Epoch':<8} {'Train Loss':<15} {'Captain Score':<18} {'Oracle Frac':<15}")
    print("-" * 80)
    for entry in training_history:
        cs = entry["captain_score"]
        of = entry["oracle_fraction"]
        cs_str = "inf" if cs == "inf" else f"{cs:.3f}"
        of_str = "inf" if of == "inf" else f"{of:.3f}"
        print(f"{entry['epoch']:<8} {entry['train_loss']:<15.6f} {cs_str:<18} {of_str:<15}")
