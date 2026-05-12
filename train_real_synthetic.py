#!/usr/bin/env python3
"""
Real synthetic training: uses SteerableSimulator to generate proper training batches
with realistic sonar scans and real catch events.

This replaces train_sync.py with a proper training loop that:
- Generates batches from SteerableSimulator (real sonar + real catches)
- Uses 45s horizon (appropriate for 40m sonar range at 1.8 m/s)
- Trains for 20 epochs with checkpoint saving
"""
import torch
import torch.nn as nn
import numpy as np
import random
from pathlib import Path
import sys

sys.path.insert(0, "src")

from eval.environment import SteerableSimulator
from ml.config import ModelConfig
from ml.model import FishCatchTransformer
from ml.loss import AsymmetricFocalLoss
from ml.dataset import encode_nav
import dataclasses
import math

# Configuration
WINDOW_SIZE = 60  # observation window (ticks/seconds)
HORIZON_S = 45.0  # prediction horizon (seconds)
BATCH_SIZE = 8
EPOCHS = 20  # Full training: 20 epochs
BATCHES_PER_EPOCH = 50  # 50 batches per epoch (8K training samples total)
DEVICE = "cpu"
LR = 3e-4
CHECKPOINT_DIR = Path("checkpoints_real_synthetic_full")

# Species names (must match model config)
SPECIES_NAMES = ["largemouth bass", "rainbow trout", "common carp", "bluegill bream"]
SPECIES_TO_IDX = {name: idx for idx, name in enumerate(SPECIES_NAMES)}


def generate_real_batch(batch_size, window_size=WINDOW_SIZE, horizon_s=HORIZON_S, device=DEVICE):
    """
    Generate a real training batch from SteerableSimulator.

    For each batch item:
    1. Create a fresh simulator with random seed
    2. Use seeking captain that steers toward nearest school (to generate catches)
    3. Collect observations and catch events
    4. Create label: 1.0 if species was caught in (last_ts, last_ts + horizon_s]
    5. Encode scans and nav, return as tensors

    Returns:
        (scans, scan_valid, nav, labels) tensors
    """
    scans_list = []
    nav_list = []
    valid_list = []
    labels_list = []

    for batch_idx in range(batch_size):
        # Fresh simulator with random seed
        seed = random.randint(0, 9999)
        sim = SteerableSimulator(seed=seed)

        # Run simulation
        duration = int(window_size + horizon_s)
        obs_history = []
        catch_history = []

        obs = sim.reset()
        for t in range(duration):
            obs_history.append(obs)

            # Seeking captain: steer toward nearest school to generate catch events
            nearest_school = None
            nearest_dist = float("inf")
            for school in sim.fish_schools:
                s = school.at(obs.ts)
                dist = np.sqrt((obs.east_m - s.east_m) ** 2 + (obs.north_m - s.north_m) ** 2)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_school = s

            if nearest_school is not None:
                # Steer toward nearest school
                target_heading = np.degrees(np.arctan2(
                    nearest_school.east_m - obs.east_m,
                    nearest_school.north_m - obs.north_m
                ))
                heading_delta = np.clip(target_heading - obs.heading_deg, -30, 30)
                speed_kts = 3.5 if nearest_dist > 30 else 1.5  # Slow down when close
            else:
                # Fallback: random walk
                heading_delta = random.uniform(-30, 30)
                speed_kts = random.uniform(2.0, 5.0)

            obs, catches = sim.step(heading_delta, speed_kts, dt=1.0)

            # Record catches with timestamp and species
            for catch in catches:
                catch_history.append((catch["ts"], catch["species"]))

        # Extract window and compute label
        # Window = last window_size observations
        if len(obs_history) < window_size:
            # Not enough observations, pad with zeros
            obs_window = [None] * (window_size - len(obs_history)) + obs_history
        else:
            obs_window = obs_history[-window_size:]

        # Get first observation timestamp of the window
        first_obs_in_window = obs_window[0] if obs_window[0] is not None else obs_window[-1]
        first_ts = first_obs_in_window.ts
        # Get last observation timestamp
        last_obs = obs_history[-1]
        last_ts = last_obs.ts

        # Label: was there a catch of each species within [first_ts, last_ts + horizon_s]?
        # This allows catches that happened during the window or in the prediction horizon after
        labels = np.zeros(4, dtype=np.float32)
        for catch_ts, species_name in catch_history:
            # Accept catches during the observation window or shortly after
            if first_ts <= catch_ts <= last_ts + horizon_s:
                if species_name in SPECIES_TO_IDX:
                    idx = SPECIES_TO_IDX[species_name]
                    labels[idx] = 1.0

        # Encode observations
        scans = []
        navs = []
        valids = []

        for obs in obs_window:
            if obs is None:
                # Padding: add zeros
                scans.append(np.zeros((1, 24, 60, 128), dtype=np.float32))
                navs.append(np.zeros(7, dtype=np.float32))
                valids.append(False)
            else:
                # Encode sonar scan (forward_scan is bytes, reshape and normalize)
                try:
                    if obs.forward_scan is not None:
                        scan_bytes = obs.forward_scan
                        scan_uint8 = np.frombuffer(scan_bytes, dtype=np.uint8).reshape(24, 60, 128)
                        scan_float = scan_uint8.astype(np.float32) / 255.0
                        scans.append(scan_float[np.newaxis, :, :, :])  # (1, 24, 60, 128)
                    else:
                        scans.append(np.zeros((1, 24, 60, 128), dtype=np.float32))
                except Exception:
                    scans.append(np.zeros((1, 24, 60, 128), dtype=np.float32))

                # Encode navigation
                nav_vec = encode_nav(
                    obs.east_m,
                    obs.north_m,
                    obs.depth_m,
                    obs.speed_kts,
                    obs.heading_deg,
                    obs.confidence,
                )
                navs.append(nav_vec)
                valids.append(True)

        scans_batch = np.stack(scans)  # (60, 1, 24, 60, 128)
        navs_batch = np.stack(navs)  # (60, 7)
        valids_batch = np.array(valids, dtype=bool)  # (60,)

        scans_list.append(scans_batch)
        nav_list.append(navs_batch)
        valid_list.append(valids_batch)
        labels_list.append(labels)

    # Stack and convert to tensors
    scans_tensor = torch.from_numpy(np.stack(scans_list)).to(device)  # (B, 60, 1, 24, 60, 128)
    nav_tensor = torch.from_numpy(np.stack(nav_list)).to(device)  # (B, 60, 7)
    valid_tensor = torch.from_numpy(np.stack(valid_list)).to(device)  # (B, 60)
    labels_tensor = torch.from_numpy(np.stack(labels_list)).to(device)  # (B, 4)

    return scans_tensor, valid_tensor, nav_tensor, labels_tensor


def main():
    print("=" * 70)
    print("REAL SYNTHETIC TRAINING")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Window size: {WINDOW_SIZE}s")
    print(f"Horizon: {HORIZON_S}s")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}, Batches/epoch: {BATCHES_PER_EPOCH}")
    print()
    sys.stdout.flush()

    # Build model
    print("Building model...")
    sys.stdout.flush()
    cfg = ModelConfig()
    cfg.horizon_s = HORIZON_S  # Override config with actual horizon
    model = FishCatchTransformer(cfg).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    sys.stdout.flush()

    # Loss and optimizer
    loss_fn = AsymmetricFocalLoss(gamma_pos=0.0, gamma_neg=4.0, clip=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=cfg.weight_decay)

    # Checkpoint directory
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    best_loss = float("inf")

    print(f"\nTraining on real synthetic data...")
    print("=" * 70)
    sys.stdout.flush()

    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- EPOCH {epoch}/{EPOCHS} ---", flush=True)
        model.train()

        epoch_losses = []

        for batch_idx in range(BATCHES_PER_EPOCH):
            # Generate real batch
            try:
                scans, valid, nav, labels = generate_real_batch(BATCH_SIZE, WINDOW_SIZE, HORIZON_S, DEVICE)
            except Exception as e:
                print(f"ERROR generating batch {batch_idx}: {e}", flush=True)
                raise

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
                print(
                    f"  Batch {batch_idx + 1:2d}/{BATCHES_PER_EPOCH}: loss={avg_loss:.6f}",
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
                    "species": SPECIES_NAMES,
                    "horizon_s": HORIZON_S,
                    "loss": avg_epoch_loss,
                },
                ckpt_path,
            )
            print(f"  -> Saved: {ckpt_path}", flush=True)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Checkpoint: {CHECKPOINT_DIR}/best.pt")
    print(f"Horizon: {HORIZON_S}s")
    print("=" * 70)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
