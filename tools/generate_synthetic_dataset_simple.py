#!/usr/bin/env python3
"""
Simple sequential dataset generator (no multiprocessing).
Writes to HDF5 incrementally to avoid memory issues.
"""
import h5py
import numpy as np
import random
import argparse
import sys
import time

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
    parser = argparse.ArgumentParser(description="Generate synthetic dataset cache (simple sequential)")
    parser.add_argument("--output", default="data/synthetic_cache.h5", help="Output HDF5 file")
    parser.add_argument("--n-samples", type=int, default=24000, help="Total samples to generate")
    args = parser.parse_args()

    output_path = args.output

    print("=" * 70)
    print("GENERATING SYNTHETIC DATASET CACHE (Sequential)")
    print("=" * 70)
    print(f"Output: {output_path}")
    print(f"Total samples: {args.n_samples}")
    print()
    sys.stdout.flush()

    # Create HDF5 file with datasets
    print("Creating HDF5 file...")
    with h5py.File(output_path, "w") as f:
        f.create_dataset("scans", shape=(args.n_samples, 60, 1, 24, 60, 128), dtype=np.float32)
        f.create_dataset("valids", shape=(args.n_samples, 60), dtype=bool)
        f.create_dataset("navs", shape=(args.n_samples, 60, 7), dtype=np.float32)
        f.create_dataset("labels", shape=(args.n_samples, 4), dtype=np.float32)
        f.attrs["n_samples"] = args.n_samples
        f.attrs["window_size"] = WINDOW_SIZE
        f.attrs["horizon_s"] = HORIZON_S
        f.attrs["species"] = SPECIES_NAMES

    # Generate and write samples
    print(f"Generating {args.n_samples} samples...")
    start_time = time.time()

    with h5py.File(output_path, "r+") as f:
        for i in range(args.n_samples):
            try:
                seed = i
                scan, valid, nav, label = generate_single_sample(seed)

                f["scans"][i] = scan
                f["valids"][i] = valid
                f["navs"][i] = nav
                f["labels"][i] = label

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

    elapsed_total = time.time() - start_time

    # Verify
    print(f"\nVerifying...")
    with h5py.File(output_path, "r") as f:
        print(f"  Scans: {f['scans'].shape}")
        print(f"  Navs: {f['navs'].shape}")
        print(f"  Labels: {f['labels'].shape}")

        labels = f["labels"][:]
        pos_rate = labels.mean(axis=0)
        print(f"  Positive rates: {pos_rate}")

    print("\n" + "=" * 70)
    print(f"COMPLETE in {elapsed_total/60:.1f} minutes")
    print(f"Cache file: {output_path}")
    print("=" * 70)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
