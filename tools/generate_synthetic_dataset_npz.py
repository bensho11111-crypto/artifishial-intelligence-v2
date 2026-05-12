#!/usr/bin/env python3
"""
Generate synthetic dataset and save as compressed NPZ (no pre-allocation issues).
Much simpler than HDF5 for our use case.
"""
import numpy as np
import random
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

from eval.environment import SteerableSimulator
from ml.dataset import encode_nav


WINDOW_SIZE = 60
HORIZON_S = 45.0
SPECIES_NAMES = ["largemouth bass", "rainbow trout", "common carp", "bluegill bream"]
SPECIES_TO_IDX = {name: idx for idx, name in enumerate(SPECIES_NAMES)}


def generate_single_sample(seed: int):
    """Generate a single training sample."""
    sim = SteerableSimulator(seed=seed)
    duration = int(WINDOW_SIZE + HORIZON_S)
    obs_history = []
    catch_history = []

    obs = sim.reset()
    for t in range(duration):
        obs_history.append(obs)

        nearest_school = None
        nearest_dist = float("inf")
        for school in sim.fish_schools:
            s = school.at(obs.ts)
            dist = np.sqrt((obs.east_m - s.east_m) ** 2 + (obs.north_m - s.north_m) ** 2)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_school = s

        if nearest_school is not None:
            target_heading = np.degrees(np.arctan2(
                nearest_school.east_m - obs.east_m,
                nearest_school.north_m - obs.north_m
            ))
            heading_delta = np.clip(target_heading - obs.heading_deg, -30, 30)
            if nearest_dist > 30:
                speed_kts = 3.5
            elif nearest_dist > 8:
                speed_kts = 1.5
            else:
                speed_kts = 0.0
        else:
            heading_delta = random.uniform(-30, 30)
            speed_kts = random.uniform(0.0, 5.0)

        obs, catches = sim.step(heading_delta, speed_kts, dt=1.0)
        for catch in catches:
            catch_history.append((catch["ts"], catch["species"]))

    # Extract window and label
    if len(obs_history) < WINDOW_SIZE:
        obs_window = [None] * (WINDOW_SIZE - len(obs_history)) + obs_history
    else:
        obs_window = obs_history[-WINDOW_SIZE:]

    first_obs = obs_window[0] if obs_window[0] is not None else obs_window[-1]
    first_ts = first_obs.ts
    last_ts = obs_history[-1].ts

    labels = np.zeros(4, dtype=np.float32)
    for catch_ts, species_name in catch_history:
        if first_ts <= catch_ts <= last_ts + HORIZON_S:
            if species_name in SPECIES_TO_IDX:
                idx = SPECIES_TO_IDX[species_name]
                labels[idx] = 1.0

    # Encode
    scans = []
    navs = []
    valids = []

    for obs in obs_window:
        if obs is None:
            scans.append(np.zeros((1, 24, 60, 128), dtype=np.float32))
            navs.append(np.zeros(7, dtype=np.float32))
            valids.append(False)
        else:
            try:
                if obs.forward_scan is not None:
                    scan_bytes = obs.forward_scan
                    scan_uint8 = np.frombuffer(scan_bytes, dtype=np.uint8).reshape(24, 60, 128)
                    scan_float = scan_uint8.astype(np.float32) / 255.0
                    scans.append(scan_float[np.newaxis, :, :, :])
                else:
                    scans.append(np.zeros((1, 24, 60, 128), dtype=np.float32))
            except Exception:
                scans.append(np.zeros((1, 24, 60, 128), dtype=np.float32))

            nav_vec = encode_nav(
                obs.east_m, obs.north_m, obs.depth_m,
                obs.speed_kts, obs.heading_deg, obs.confidence,
            )
            navs.append(nav_vec)
            valids.append(True)

    scans_batch = np.stack(scans)
    navs_batch = np.stack(navs)
    valids_batch = np.array(valids, dtype=bool)

    return scans_batch, valids_batch, navs_batch, labels


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic dataset as NPZ")
    parser.add_argument("--output", default="data/synthetic_cache.npz", help="Output NPZ file")
    parser.add_argument("--n-samples", type=int, default=24000, help="Total samples to generate")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GENERATING SYNTHETIC DATASET (NPZ format)")
    print("=" * 70)
    print(f"Output: {output_path}")
    print(f"Total samples: {args.n_samples}")
    print()
    sys.stdout.flush()

    # Generate all samples in memory first, then save
    print(f"Generating {args.n_samples} samples...")
    start_time = time.time()

    scans_list = []
    valids_list = []
    navs_list = []
    labels_list = []

    for i in range(args.n_samples):
        try:
            seed = i
            scan, valid, nav, label = generate_single_sample(seed)

            scans_list.append(scan)
            valids_list.append(valid)
            navs_list.append(nav)
            labels_list.append(label)

            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (args.n_samples - i - 1) / rate
                print(
                    f"  {i+1:5d}/{args.n_samples}: {rate:.2f} samples/sec, "
                    f"ETA {remaining/60:.1f} min",
                    flush=True,
                )
        except Exception as e:
            print(f"  ERROR at sample {i}: {e}", flush=True)
            continue

    elapsed_gen = time.time() - start_time

    # Stack arrays
    print(f"\nStacking arrays...")
    scans = np.stack(scans_list)
    valids = np.stack(valids_list)
    navs = np.stack(navs_list)
    labels = np.stack(labels_list)

    print(f"  Scans: {scans.shape}")
    print(f"  Navs: {navs.shape}")
    print(f"  Labels: {labels.shape}")

    # Save as NPZ
    print(f"\nSaving to NPZ (compressed)...")
    start_save = time.time()
    np.savez_compressed(
        output_path,
        scans=scans,
        valids=valids,
        navs=navs,
        labels=labels,
        window_size=WINDOW_SIZE,
        horizon_s=HORIZON_S,
        species=SPECIES_NAMES,
    )
    elapsed_save = time.time() - start_save

    # Verify
    print(f"\nVerifying...")
    data = np.load(output_path, allow_pickle=True)
    print(f"  Keys: {list(data.keys())}")
    print(f"  Scans: {data['scans'].shape}")
    print(f"  Labels shape: {data['labels'].shape}")
    pos_rate = data['labels'].mean(axis=0)
    print(f"  Positive rates: {pos_rate}")

    file_size_mb = output_path.stat().st_size / (1024 * 1024)

    print("\n" + "=" * 70)
    print(f"COMPLETE in {elapsed_gen/60:.1f} min (generation) + {elapsed_save/60:.1f} min (save)")
    print(f"File size: {file_size_mb:.1f} MB")
    print(f"Cache file: {output_path}")
    print("=" * 70)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
