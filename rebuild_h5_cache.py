#!/usr/bin/env python3
"""Rebuild HDF5 cache with correct sonar data from .ticks files."""
import sys
import os
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

try:
    import h5py
except ImportError:
    print("h5py not installed. Install with: pip install h5py")
    sys.exit(1)

from ml.config import ModelConfig
from ml.dataset import SPECIES
from ticks.replayer import Replayer

def build_cache(data_dir, output_h5):
    """Build HDF5 cache from .ticks files with correct sonar loading."""

    cfg = ModelConfig()
    data_path = Path(data_dir)

    # Find all sessions
    ticks_files = sorted(data_path.glob("*.ticks"))
    if not ticks_files:
        print(f"No .ticks files found in {data_dir}")
        return

    print(f"Found {len(ticks_files)} sessions")

    # Count total windows
    total_windows = 0
    windows_per_session = {}

    for ticks_file in ticks_files:
        catch_file = ticks_file.parent / (ticks_file.stem + "_catches.json")
        if not catch_file.exists():
            continue

        # Load windows from session
        windows = []
        try:
            with Replayer(str(ticks_file)) as rep:
                # Maintain latest sonar data (same as dataset.py)
                latest_fwd_bytes = None
                latest_depth = 0.0
                obs_count = 0

                for tick in rep.iter_all():
                    # Update latest sonar
                    if tick.sonar:
                        latest_fwd_bytes = tick.sonar.forward_scan
                        latest_depth = tick.sonar.depth_m

                    # Only count GPS ticks
                    if tick.gps:
                        obs_count += 1
                        windows.append({
                            "ts": tick.gps.ts,
                            "fwd_bytes": latest_fwd_bytes,
                            "depth_m": latest_depth,
                        })

            # All windows with 60-second lookback
            n_windows = max(0, obs_count - cfg.window_size)
            windows_per_session[str(ticks_file)] = n_windows
            total_windows += n_windows
            print(f"  {ticks_file.name}: {obs_count} obs, {n_windows} windows")
        except Exception as e:
            print(f"  ERROR {ticks_file.name}: {e}")

    print(f"\nTotal windows: {total_windows}")
    print(f"Building HDF5: {output_h5}")

    # Create HDF5 file
    with h5py.File(output_h5, 'w') as h5:
        # Pre-allocate arrays
        scans_train = h5.create_dataset('train/scans',
                                        shape=(total_windows, cfg.window_size, 1, 24, 60, 128),
                                        dtype=np.uint8)
        scan_valid_train = h5.create_dataset('train/scan_valid',
                                             shape=(total_windows, cfg.window_size),
                                             dtype=bool)
        nav_train = h5.create_dataset('train/nav',
                                      shape=(total_windows, cfg.window_size, 7),
                                      dtype=np.float32)
        labels_train = h5.create_dataset('train/labels',
                                         shape=(total_windows, len(SPECIES)),
                                         dtype=np.float32)

        # Fill datasets
        global_idx = 0

        for ticks_file in ticks_files:
            catch_file = ticks_file.parent / (ticks_file.stem + "_catches.json")
            if not catch_file.exists():
                continue

            print(f"\nProcessing {ticks_file.name}...")

            # Load observations and catches
            obs_list = []
            latest_fwd_bytes = None
            latest_depth = 0.0

            try:
                with Replayer(str(ticks_file)) as rep:
                    for tick in rep.iter_all():
                        if tick.sonar:
                            latest_fwd_bytes = tick.sonar.forward_scan
                            latest_depth = tick.sonar.depth_m

                        if tick.gps:
                            obs_list.append({
                                "ts": tick.gps.ts,
                                "fwd_bytes": latest_fwd_bytes,
                                "depth_m": latest_depth,
                            })
            except Exception as e:
                print(f"  ERROR loading {ticks_file.name}: {e}")
                continue

            # Load catches
            try:
                with open(catch_file) as f:
                    catch_data = json.load(f)
                    catches = catch_data.get("catches", [])
            except:
                catches = []

            # Build windows
            for end_idx in range(cfg.window_size, len(obs_list)):
                window = obs_list[end_idx - cfg.window_size:end_idx]

                # Scans
                scans = np.zeros((cfg.window_size, 1, 24, 60, 128), dtype=np.uint8)
                scan_valid = np.zeros(cfg.window_size, dtype=bool)

                for t, obs in enumerate(window):
                    if obs["fwd_bytes"] is not None:
                        try:
                            arr = np.frombuffer(obs["fwd_bytes"], dtype=np.uint8).reshape(24, 60, 128)
                            scans[t, 0] = arr
                            scan_valid[t] = True
                        except:
                            pass

                # Labels
                last_ts = window[-1]["ts"]
                label = np.zeros(len(SPECIES), dtype=np.float32)
                for catch in catches:
                    catch_ts = catch.get("ts", 0)
                    if last_ts < catch_ts <= last_ts + 300:  # horizon_s=300
                        try:
                            s_idx = SPECIES.index(catch["species"])
                            label[s_idx] = 1.0
                        except:
                            pass

                # Write to HDF5
                scans_train[global_idx] = scans
                scan_valid_train[global_idx] = scan_valid
                nav_train[global_idx] = 0  # Placeholder nav data
                labels_train[global_idx] = label
                global_idx += 1

                if global_idx % 1000 == 0:
                    print(f"  Written {global_idx}/{total_windows} windows")

    print(f"\nDone! HDF5 cache created: {output_h5}")
    print(f"Total windows written: {global_idx}")

if __name__ == "__main__":
    build_cache("data/training_300/", "data/training_300_cache.h5")
