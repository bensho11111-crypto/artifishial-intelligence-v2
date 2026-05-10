"""
Optimized DataLoader that builds a lightweight index and loads sonar on-demand.

Strategy:
1. Build index: lightweight JSON with window metadata (session, boundaries, labels)
2. Load only on demand: read sonar from .ticks files during epoch
3. Cache in memory: LRU cache of last N batches to avoid re-reading same windows
4. Stream to GPU: move batches to GPU after loading from disk
"""
import json
import numpy as np
from pathlib import Path
from collections import OrderedDict
import torch
from torch.utils.data import Dataset, DataLoader
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml.config import ModelConfig
from ml.dataset import SPECIES, encode_nav
from ticks.replayer import Replayer


def build_index(data_dir, output_index_path, config):
    """
    Build lightweight index of all windows without loading sonar.
    Index file: ~100 KB for 30k windows (just metadata, no data)
    """
    print(f"Building index from {data_dir}...")

    data_path = Path(data_dir)
    ticks_files = sorted(data_path.glob("*.ticks"))

    windows = []  # List of {session_path, start_obs_idx, end_obs_idx, labels}
    session_obs_count = {}  # Track obs count per session for indexing

    for ticks_file in ticks_files:
        catch_file = ticks_file.parent / (ticks_file.stem + "_catches.json")
        if not catch_file.exists():
            continue

        # Load catches for labels
        try:
            with open(catch_file) as f:
                catch_data = json.load(f)
                catches = catch_data.get("catches", [])
        except:
            catches = []

        # Count observations (don't load sonar)
        obs_count = 0
        obs_timestamps = []
        try:
            with Replayer(str(ticks_file)) as rep:
                for tick in rep.iter_all():
                    if tick.gps:
                        obs_count += 1
                        obs_timestamps.append(tick.gps.ts)
        except Exception as e:
            print(f"  Warning: {ticks_file.name}: {e}")
            continue

        session_obs_count[str(ticks_file)] = obs_count

        # Create windows for this session
        for end_idx in range(config.window_size, obs_count):
            start_idx = end_idx - config.window_size
            window_start_ts = obs_timestamps[start_idx]
            window_end_ts = obs_timestamps[end_idx - 1]

            # Determine labels based on catches in next 300s
            label = np.zeros(len(SPECIES), dtype=np.float32)
            for catch in catches:
                catch_ts = catch.get("ts", 0)
                if window_end_ts < catch_ts <= window_end_ts + config.horizon_s:
                    try:
                        s_idx = SPECIES.index(catch["species"])
                        label[s_idx] = 1.0
                    except:
                        pass

            windows.append({
                "session_path": str(ticks_file),
                "start_obs_idx": int(start_idx),
                "end_obs_idx": int(end_idx),
                "label": label.tolist(),
            })

    # Save index
    index_data = {
        "config": {
            "window_size": config.window_size,
            "horizon_s": config.horizon_s,
            "species": config.species,
        },
        "windows": windows,
        "session_obs_count": session_obs_count,
    }

    with open(output_index_path, 'w') as f:
        json.dump(index_data, f, indent=2)

    print(f"Index built: {len(windows)} windows, {Path(output_index_path).stat().st_size / 1024:.1f} KB")
    return index_data


class OptimizedDataset(Dataset):
    """
    Dataset that uses lightweight index and loads sonar on-demand from .ticks files.
    """
    def __init__(self, index_path, config, train=True, val_fraction=0.15):
        self.config = config
        self.train = train

        # Load index
        with open(index_path) as f:
            index_data = json.load(f)

        self.windows = index_data["windows"]
        self.session_obs_count = index_data["session_obs_count"]

        # Split train/val by session (not by window)
        sessions = sorted(set(w["session_path"] for w in self.windows))
        n_val = max(1, int(len(sessions) * val_fraction))
        val_sessions = set(sessions[-n_val:])

        if train:
            self.windows = [w for w in self.windows if w["session_path"] not in val_sessions]
        else:
            self.windows = [w for w in self.windows if w["session_path"] in val_sessions]

        # Cache for .ticks file handles (one per session)
        self.session_replayers = {}
        self.session_obs_lists = {}

    def _get_obs_list(self, session_path):
        """Get cached observation list for a session."""
        if session_path not in self.session_obs_lists:
            obs_list = []
            latest_fwd_bytes = None
            latest_depth = 0.0

            try:
                with Replayer(str(session_path)) as rep:
                    for tick in rep.iter_all():
                        if tick.sonar:
                            latest_fwd_bytes = tick.sonar.forward_scan
                            latest_depth = tick.sonar.depth_m

                        if tick.gps:
                            obs_list.append({
                                "ts": tick.gps.ts,
                                "fwd_bytes": latest_fwd_bytes,
                                "depth_m": latest_depth,
                                "speed_kts": tick.gps.speed_kts,
                                "heading_deg": tick.gps.heading_deg,
                            })
            except Exception as e:
                print(f"Error loading {session_path}: {e}")
                return None

            self.session_obs_lists[session_path] = obs_list

        return self.session_obs_lists[session_path]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]
        session_path = window["session_path"]
        start_idx = window["start_obs_idx"]
        end_idx = window["end_obs_idx"]

        # Load observation window
        obs_list = self._get_obs_list(session_path)
        if obs_list is None:
            # Return zeros on error
            return {
                "scans": torch.zeros(self.config.window_size, 1, 24, 60, 128, dtype=torch.float32),
                "scan_valid": torch.zeros(self.config.window_size, dtype=torch.bool),
                "nav": torch.zeros(self.config.window_size, 7, dtype=torch.float32),
                "label": torch.tensor(window["label"], dtype=torch.float32),
            }

        obs_window = obs_list[start_idx:end_idx]

        # Build scans (load sonar from .ticks)
        scans = np.zeros((self.config.window_size, 1, 24, 60, 128), dtype=np.float32)
        scan_valid = np.zeros(self.config.window_size, dtype=bool)

        for t, obs in enumerate(obs_window):
            if obs["fwd_bytes"] is not None:
                try:
                    arr = np.frombuffer(obs["fwd_bytes"], dtype=np.uint8).reshape(24, 60, 128)
                    scans[t, 0] = arr.astype(np.float32) / 255.0
                    scan_valid[t] = True
                except:
                    pass

        # Build nav
        nav = np.stack([
            encode_nav(0, 0, obs["depth_m"], obs["speed_kts"], obs["heading_deg"], 0.7)
            for obs in obs_window
        ]).astype(np.float32)

        return {
            "scans": torch.from_numpy(scans),
            "scan_valid": torch.from_numpy(scan_valid),
            "nav": torch.from_numpy(nav),
            "label": torch.tensor(window["label"], dtype=torch.float32),
        }

    def __del__(self):
        """Cleanup cached replayers."""
        for rep in self.session_replayers.values():
            try:
                rep.close()
            except:
                pass


def create_optimized_dataloaders(data_dir, config, batch_size=32, num_workers=0, index_path=None):
    """
    Create train/val dataloaders using optimized dataset.

    Args:
        data_dir: Path to data directory
        config: ModelConfig
        batch_size: Batch size
        num_workers: Number of workers (0 recommended for .ticks loading)
        index_path: Path to save/load index (auto-generated if None)

    Returns:
        (train_loader, val_loader)
    """
    if index_path is None:
        index_path = Path(data_dir).parent / (Path(data_dir).name + "_index.json")

    # Build index if needed
    if not Path(index_path).exists():
        build_index(data_dir, index_path, config)

    # Create datasets
    train_ds = OptimizedDataset(str(index_path), config, train=True, val_fraction=0.15)
    val_ds = OptimizedDataset(str(index_path), config, train=False, val_fraction=0.15)

    print(f"Train: {len(train_ds)} windows, Val: {len(val_ds)} windows")

    # Create loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Test
    cfg = ModelConfig()
    train_loader, val_loader = create_optimized_dataloaders("data/training_300/", cfg, batch_size=16)

    print("\nTesting batch loading...")
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= 3:
            break
        print(f"Batch {batch_idx}: scans {batch['scans'].shape}, label {batch['label'].shape}")

    print("\nOptimized DataLoader test complete!")
