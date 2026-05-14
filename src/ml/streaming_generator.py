"""Streaming batch generation for on-the-fly training without pre-generation."""

import logging
import multiprocessing
import numpy as np
import os
import random
import sys
import tempfile
import time
import torch
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eval.environment import SteerableSimulator
from ml.dataset import encode_nav

WINDOW_SIZE = 60
HORIZON_S = 45.0
SPECIES_NAMES = ["largemouth bass", "rainbow trout", "common carp", "bluegill bream"]
SPECIES_TO_IDX = {name: idx for idx, name in enumerate(SPECIES_NAMES)}
WORKER_BATCH_SIZE = 8


def generate_sample_numpy(seed: int) -> dict:
    """Generate a single training sample as numpy dict."""
    sim = SteerableSimulator(seed=seed)
    duration = 250
    obs_history = []
    catch_history = []

    rng = random.Random(seed)
    obs = sim.reset()
    for t in range(duration):
        obs_history.append(obs)
        heading_delta = rng.uniform(-30, 30)
        speed_kts = rng.uniform(0.0, 5.0)
        obs, catches = sim.step(heading_delta, speed_kts, dt=1.0)
        for catch in catches:
            catch_history.append((catch["ts"], catch["species"]))

    # Extract window from middle of trajectory
    window_start_idx = len(obs_history) - WINDOW_SIZE - int(HORIZON_S)
    if window_start_idx < 0:
        window_start_idx = 0
    window_end_idx = window_start_idx + WINDOW_SIZE

    obs_window = obs_history[window_start_idx:window_end_idx]
    if len(obs_window) < WINDOW_SIZE:
        obs_window = [None] * (WINDOW_SIZE - len(obs_window)) + obs_window

    first_obs = obs_window[0] if obs_window[0] is not None else obs_window[-1]
    first_ts = first_obs.ts
    last_ts = obs_window[-1].ts

    # Label: catches in (last_ts, last_ts + HORIZON_S]
    labels = np.zeros(4, dtype=np.float32)
    for catch_ts, species_name in catch_history:
        if last_ts < catch_ts <= last_ts + HORIZON_S:
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

    return {
        "scans": scans_batch,           # (60, 1, 24, 60, 128)
        "valids": valids_batch,         # (60,) bool
        "navs": navs_batch,             # (60, 7)
        "labels": labels,               # (4,)
    }


def worker_main(worker_id, n_workers, queue_dir, batch_size, max_queue_depth, stop_event):
    """Worker process: generates batches and writes to queue directory.

    Args:
        worker_id: [0, n_workers) identifier
        n_workers: total number of workers
        queue_dir: Path to directory for batch .npz files
        batch_size: samples per batch file
        max_queue_depth: max .npz files before back-pressuring
        stop_event: multiprocessing.Event() to signal shutdown
    """
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(f"worker_{worker_id}")

    seed = worker_id
    counter = 0
    log.info(f"Worker started, initial seed={seed}")

    while not stop_event.is_set():
        try:
            # Back-pressure: limit queue depth
            while len(list(queue_dir.glob("*.npz"))) >= max_queue_depth:
                if stop_event.is_set():
                    return
                time.sleep(0.5)

            # Generate batch of samples
            samples = []
            for i in range(batch_size):
                sample = generate_sample_numpy(seed + i * n_workers)
                samples.append(sample)

            seed += batch_size * n_workers

            # Stack samples into batch arrays
            batch_scans = np.stack([s["scans"] for s in samples], axis=0)      # (B, T, 1, H, W, D)
            batch_valids = np.stack([s["valids"] for s in samples], axis=0)    # (B, T)
            batch_navs = np.stack([s["navs"] for s in samples], axis=0)        # (B, T, 7)
            batch_labels = np.stack([s["labels"] for s in samples], axis=0)    # (B, 4)

            # Write to temp file with explicit fsync before atomic rename
            out_path = queue_dir / f"batch_{worker_id:02d}_{counter:06d}.npz"

            # Create temporary file in same directory (ensures same filesystem for atomic rename)
            fd, tmp_path = tempfile.mkstemp(suffix='.npz', dir=str(queue_dir))
            try:
                os.close(fd)  # Close the FD, numpy will open it
                np.savez_compressed(
                    tmp_path,
                    scans=batch_scans,
                    valids=batch_valids,
                    navs=batch_navs,
                    labels=batch_labels,
                )

                # Force flush to disk to ensure file is complete
                try:
                    fd = os.open(tmp_path, os.O_RDONLY | os.O_BINARY)
                    try:
                        os.fsync(fd)
                    finally:
                        os.close(fd)
                except Exception:
                    pass  # If fsync fails, still try rename

                # Atomic rename
                os.rename(tmp_path, str(out_path))
                log.info(f"Batch {counter} written ({out_path.name})")
            except Exception as e:
                log.error(f"Error writing batch {counter}: {e}")
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                raise

            counter += 1
        except Exception as e:
            log.error(f"Error in worker loop: {e}", exc_info=True)
            break


class StreamingDataLoader:
    """Reads batch files from queue directory, yields training-compatible dicts.

    Accumulates samples from queue files until batch_size is reached, then yields
    a training batch. Files are deleted after reading. This is compatible with
    train_epoch() and eval_epoch() from src.ml.train.
    """

    def __init__(self, queue_dir, n_steps, batch_size=32, device="cpu"):
        """
        Args:
            queue_dir: Path to directory where workers write .npz files
            n_steps: number of batches to yield (one epoch = n_steps)
            batch_size: samples per training batch
            device: torch device for tensors
        """
        self.queue_dir = Path(queue_dir)
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.device = device

    def __len__(self):
        """Return number of batches (for compatibility with eval_epoch)."""
        return self.n_steps

    def _wait_for_batch_file(self, timeout=300):
        """Poll for next batch file, delete after returning."""
        start_time = time.time()
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"No batch file after {timeout}s")

            files = sorted(self.queue_dir.glob("*.npz"))
            if files:
                # Return oldest file (first in sorted order)
                # Brief stability check: worker uses fsync, so just verify file exists and stable
                f = files[0]
                try:
                    size1 = f.stat().st_size
                    if size1 > 0:
                        time.sleep(0.5)  # Brief wait for atomic rename to complete
                        size2 = f.stat().st_size
                        if size1 == size2:  # File is stable
                            return f
                except OSError:
                    pass  # File deleted concurrently
            time.sleep(0.1)

    def __iter__(self):
        """Yield n_steps batches of batch_size samples each."""
        pending_samples = []

        for step_idx in range(self.n_steps):
            # Accumulate samples until we have batch_size
            while len(pending_samples) < self.batch_size:
                batch_file = self._wait_for_batch_file()
                try:
                    data = np.load(batch_file, allow_pickle=False)
                except (EOFError, ValueError, OSError, zipfile.BadZipFile) as e:
                    # File was incomplete, corrupted, or deleted, try next
                    try:
                        batch_file.unlink()
                    except OSError:
                        pass
                    time.sleep(0.1)
                    continue

                # Extract individual samples from batch file BEFORE deleting
                n_samples_in_file = len(data["labels"])
                for i in range(n_samples_in_file):
                    pending_samples.append({
                        "scans": data["scans"][i].copy(),  # Make copies to release file handles
                        "valids": data["valids"][i].copy(),
                        "navs": data["navs"][i].copy(),
                        "labels": data["labels"][i].copy(),
                    })

                # Now delete file with retry
                for attempt in range(5):
                    try:
                        batch_file.unlink()
                        break
                    except (OSError, PermissionError):
                        if attempt < 4:
                            time.sleep(0.1)
                        else:
                            pass  # Ignore if we can't delete, it's not critical

            # Take batch_size samples
            batch_list = pending_samples[:self.batch_size]
            pending_samples = pending_samples[self.batch_size:]

            # Stack into tensors
            batch = {
                "scans": torch.from_numpy(
                    np.stack([s["scans"] for s in batch_list], axis=0)
                ).to(self.device),
                "scan_valid": torch.from_numpy(
                    np.stack([s["valids"] for s in batch_list], axis=0)
                ).to(self.device),
                "nav": torch.from_numpy(
                    np.stack([s["navs"] for s in batch_list], axis=0)
                ).to(self.device),
                "label": torch.from_numpy(
                    np.stack([s["labels"] for s in batch_list], axis=0)
                ).to(self.device),
            }

            yield batch


class FixedNpzLoader:
    """Reads a fixed NPZ file (e.g., validation set) in batches.

    Compatible with eval_epoch() — does not delete the file.
    """

    def __init__(self, npz_path, n_steps, batch_size=32, device="cpu"):
        """
        Args:
            npz_path: Path to .npz file with scans, valids, navs, labels
            n_steps: number of batches to yield
            batch_size: samples per batch
            device: torch device
        """
        self.npz_path = Path(npz_path)
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.device = device

        # Load all data upfront
        data = np.load(self.npz_path, allow_pickle=False)
        self.scans = data["scans"]      # (N, 60, 1, 24, 60, 128)
        self.valids = data["valids"]    # (N, 60)
        self.navs = data["navs"]        # (N, 60, 7)
        self.labels = data["labels"]    # (N, 4)

        self.n_samples = len(self.labels)

    def __len__(self):
        """Return number of batches (for compatibility with eval_epoch)."""
        return self.n_steps

    def __iter__(self):
        """Yield n_steps batches of batch_size samples each."""
        idx = 0
        for _ in range(self.n_steps):
            # Get next batch_size samples (cycle if needed)
            batch_indices = [(idx + i) % self.n_samples for i in range(self.batch_size)]
            idx = (idx + self.batch_size) % self.n_samples

            batch = {
                "scans": torch.from_numpy(self.scans[batch_indices]).to(self.device),
                "scan_valid": torch.from_numpy(self.valids[batch_indices]).to(self.device),
                "nav": torch.from_numpy(self.navs[batch_indices]).to(self.device),
                "label": torch.from_numpy(self.labels[batch_indices]).to(self.device),
            }

            yield batch
