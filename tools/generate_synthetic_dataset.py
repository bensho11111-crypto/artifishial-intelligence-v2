#!/usr/bin/env python3
"""
Generate and cache synthetic training dataset to HDF5.

This script parallelizes batch generation and caches to HDF5 for fast training.
One-time ~2 hour setup, then training becomes minutes instead of hours.

Usage:
    python tools/generate_synthetic_dataset.py \
        --output data/synthetic_cache.h5 \
        --n-samples 24000 \
        --batch-size 16 \
        --workers 4
"""
import h5py
import numpy as np
import random
import argparse
import sys
from pathlib import Path
from multiprocessing import Pool, Process, Queue
import time

sys.path.insert(0, "src")

from eval.environment import SteerableSimulator
from ml.dataset import encode_nav


WINDOW_SIZE = 60
HORIZON_S = 45.0
SPECIES_NAMES = ["largemouth bass", "rainbow trout", "common carp", "bluegill bream"]
SPECIES_TO_IDX = {name: idx for idx, name in enumerate(SPECIES_NAMES)}


def generate_single_sample(seed: int, window_size=WINDOW_SIZE, horizon_s=HORIZON_S):
    """
    Generate a single training sample.

    Returns:
        (scan_60, valid_60, nav_60, label_4) as numpy arrays
    """
    # Fresh simulator
    sim = SteerableSimulator(seed=seed)

    # Run simulation
    duration = int(window_size + horizon_s)
    obs_history = []
    catch_history = []

    obs = sim.reset()
    for t in range(duration):
        obs_history.append(obs)

        # Seeking captain
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

    # Extract window
    if len(obs_history) < window_size:
        obs_window = [None] * (window_size - len(obs_history)) + obs_history
    else:
        obs_window = obs_history[-window_size:]

    first_obs_in_window = obs_window[0] if obs_window[0] is not None else obs_window[-1]
    first_ts = first_obs_in_window.ts
    last_obs = obs_history[-1]
    last_ts = last_obs.ts

    # Label
    labels = np.zeros(4, dtype=np.float32)
    for catch_ts, species_name in catch_history:
        if first_ts <= catch_ts <= last_ts + horizon_s:
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
    navs_batch = np.stack(navs)    # (60, 7)
    valids_batch = np.array(valids, dtype=bool)  # (60,)

    return scans_batch, valids_batch, navs_batch, labels


def worker_process(worker_id: int, n_samples: int, output_queue: Queue):
    """
    Worker process: generate samples and put them on queue.
    """
    samples = []
    for i in range(n_samples):
        seed = worker_id * 100000 + i
        try:
            sample = generate_single_sample(seed)
            samples.append(sample)

            if (i + 1) % 10 == 0:
                print(f"  Worker {worker_id}: {i+1}/{n_samples} samples", flush=True)
        except Exception as e:
            print(f"  Worker {worker_id} error at sample {i}: {e}", flush=True)
            # Skip failed sample

    output_queue.put((worker_id, samples))


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic training dataset cache")
    parser.add_argument("--output", default="data/synthetic_cache.h5", help="Output HDF5 file")
    parser.add_argument("--n-samples", type=int, default=24000, help="Total samples to generate")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (for reference)")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"GENERATING SYNTHETIC DATASET CACHE")
    print("=" * 70)
    print(f"Output: {output_path}")
    print(f"Total samples: {args.n_samples}")
    print(f"Workers: {args.workers}")
    print(f"Samples per worker: {args.n_samples // args.workers}")
    print()

    # Distribute samples across workers
    samples_per_worker = args.n_samples // args.workers

    print(f"Starting {args.workers} worker processes...")
    start_time = time.time()

    # Create work queue and output queue
    output_queue = Queue()
    processes = []

    for worker_id in range(args.workers):
        p = Process(target=worker_process, args=(worker_id, samples_per_worker, output_queue))
        p.start()
        processes.append(p)

    # Collect results
    all_samples = {}
    for _ in range(args.workers):
        worker_id, samples = output_queue.get()
        all_samples[worker_id] = samples
        print(f"Worker {worker_id} completed: {len(samples)} samples")

    # Wait for all processes
    for p in processes:
        p.join()

    elapsed_gen = time.time() - start_time
    print(f"\nGeneration took {elapsed_gen/60:.1f} minutes")

    # Flatten samples in order
    flat_samples = []
    for worker_id in range(args.workers):
        flat_samples.extend(all_samples[worker_id])

    print(f"\nTotal samples collected: {len(flat_samples)}")

    # Write to HDF5
    print(f"\nWriting to HDF5...")
    start_write = time.time()

    with h5py.File(output_path, "w") as f:
        # Create datasets
        f.create_dataset("scans", shape=(len(flat_samples), 60, 1, 24, 60, 128), dtype=np.float32)
        f.create_dataset("valids", shape=(len(flat_samples), 60), dtype=bool)
        f.create_dataset("navs", shape=(len(flat_samples), 60, 7), dtype=np.float32)
        f.create_dataset("labels", shape=(len(flat_samples), 4), dtype=np.float32)

        # Write data in chunks (to avoid memory explosion)
        chunk_size = 100
        for i in range(0, len(flat_samples), chunk_size):
            chunk = flat_samples[i : i + chunk_size]

            scans_chunk = np.stack([s[0] for s in chunk])
            valids_chunk = np.stack([s[1] for s in chunk])
            navs_chunk = np.stack([s[2] for s in chunk])
            labels_chunk = np.stack([s[3] for s in chunk])

            f["scans"][i : i + len(chunk)] = scans_chunk
            f["valids"][i : i + len(chunk)] = valids_chunk
            f["navs"][i : i + len(chunk)] = navs_chunk
            f["labels"][i : i + len(chunk)] = labels_chunk

            if (i + len(chunk)) % 500 == 0:
                print(f"  Written {i + len(chunk)}/{len(flat_samples)} samples")

        # Store metadata
        f.attrs["n_samples"] = len(flat_samples)
        f.attrs["window_size"] = WINDOW_SIZE
        f.attrs["horizon_s"] = HORIZON_S
        f.attrs["species"] = SPECIES_NAMES

    elapsed_write = time.time() - start_write

    # Verify
    print(f"\nVerifying HDF5...")
    with h5py.File(output_path, "r") as f:
        print(f"  Scans: {f['scans'].shape}")
        print(f"  Valids: {f['valids'].shape}")
        print(f"  Navs: {f['navs'].shape}")
        print(f"  Labels: {f['labels'].shape}")

        # Check positive rate
        labels = f["labels"][:]
        pos_rate = labels.mean(axis=0)
        print(f"  Positive rates: {pos_rate}")

    elapsed_total = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"COMPLETE in {elapsed_total/60:.1f} minutes")
    print(f"  Generation: {elapsed_gen/60:.1f} min")
    print(f"  HDF5 write: {elapsed_write/60:.1f} min")
    print(f"Cache file: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
