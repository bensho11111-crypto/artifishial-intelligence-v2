"""
src/ml/dataset.py

Windowed dataset for fish catch prediction from paired .ticks + _catches.json files.
"""
import numpy as np
import torch
import json
import math
from pathlib import Path
from torch.utils.data import Dataset
from ml.config import ModelConfig
from ticks.replayer import Replayer

SPECIES = ["largemouth bass", "rainbow trout", "common carp", "bluegill bream"]


def encode_nav(east_m: float, north_m: float, depth_m: float, speed_kts: float,
               heading_deg: float, confidence: float) -> np.ndarray:
    """
    Encode 7-dimensional navigation state vector.

    Normalizes position, depth, and speed; converts heading to sin/cos components;
    includes confidence score.

    Args:
        east_m: East position in meters
        north_m: North position in meters
        depth_m: Depth in meters
        speed_kts: Speed in knots
        heading_deg: Heading in degrees [0, 360)
        confidence: Confidence score [0, 1]

    Returns:
        (7,) float32 array [east_norm, north_norm, depth_norm, speed_norm, sin_head, cos_head, conf]
    """
    rad = math.radians(heading_deg)
    return np.array([
        east_m / 250.0,
        north_m / 250.0,
        depth_m / 32.0,
        speed_kts / 15.0,
        math.sin(rad),
        math.cos(rad),
        confidence
    ], dtype=np.float32)


class FishCatchDataset(Dataset):
    """
    Windowed dataset from paired .ticks + _catches.json files.

    Loads sessions from data_dir, splits by session (not window) to prevent leakage.
    Each __getitem__ returns one window (60 ticks at 1 Hz) with multi-label target
    indicating if any catch occurred in the lookahead horizon_s.

    Args:
        data_dir: Directory containing *.ticks and *_catches.json files
        cfg: ModelConfig with window_size, horizon_s, etc.
        augment: If True, apply data augmentation (heading rotation, speed jitter, temporal flip)
        val_fraction: Fraction of sessions to hold out for validation [0, 1]
        train: If True, use training split; else use validation split
    """

    def __init__(self, data_dir: str, cfg: ModelConfig, augment: bool = False,
                 val_fraction: float = 0.15, train: bool = True):
        self.data_dir = Path(data_dir)
        self.cfg = cfg
        self.augment = augment
        self.window_size = cfg.window_size  # typically 60 ticks at 1 Hz
        self.horizon_s = cfg.horizon_s      # typically 300 seconds

        # Find all .ticks files and their catch pairs
        ticks_files = sorted(self.data_dir.glob("*.ticks"))
        assert len(ticks_files) > 0, f"No .ticks files in {data_dir}"

        # Session-level train/val split
        n_val = max(1, int(len(ticks_files) * val_fraction))
        val_sessions = set(ticks_files[:n_val])
        train_sessions = [f for f in ticks_files if f not in val_sessions]

        sessions = list(val_sessions) if not train else train_sessions

        # Build per-session window index: (session_path, end_idx)
        self._index = []
        self._obs_cache = {}

        for session_path in sessions:
            catch_path = session_path.parent / (session_path.stem + "_catches.json")
            if not catch_path.exists():
                continue

            # Read .ticks and extract observations (GPS ticks at 1 Hz)
            obs_list = []
            try:
                with Replayer(str(session_path)) as rep:
                    for tick in rep.iter_all():
                        if tick.gps:  # only GPS ticks produce observations (1 Hz)
                            obs_list.append({
                                "ts": tick.gps.ts,
                                "east_m": tick.gps.lat,  # placeholder
                                "north_m": tick.gps.lon,  # placeholder
                                "depth_m": tick.sonar.depth_m if tick.sonar else 0.0,
                                "speed_kts": tick.gps.speed_kts,
                                "heading_deg": tick.gps.heading_deg,
                                "confidence": 0.7,  # placeholder
                                "fwd_bytes": tick.sonar.forward_scan if tick.sonar else None,
                            })
            except Exception as e:
                # Skip sessions with read errors
                print(f"Warning: skipping {session_path}: {e}")
                continue

            self._obs_cache[str(session_path)] = obs_list
            n_obs = len(obs_list)

            # All valid windows: end_idx in [window_size, n_obs)
            for end_idx in range(self.window_size, n_obs):
                self._index.append((str(session_path), end_idx))

        # Load all catches into memory
        self._catches = {}
        for session_path in sessions:
            catch_path = session_path.parent / (session_path.stem + "_catches.json")
            if catch_path.exists():
                try:
                    with open(catch_path) as f:
                        data = json.load(f)
                        self._catches[str(session_path)] = data.get("catches", [])
                except Exception as e:
                    print(f"Warning: skipping catches for {session_path}: {e}")
                    self._catches[str(session_path)] = []

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        session_path, end_idx = self._index[idx]
        obs_window = self._obs_cache[session_path][end_idx - self.window_size:end_idx]

        # Navigation tensor: (T, 7)
        nav = np.stack([
            encode_nav(o["east_m"], o["north_m"], o["depth_m"],
                      o["speed_kts"], o["heading_deg"], o["confidence"])
            for o in obs_window
        ]).astype(np.float32)

        # Scan tensor: (T, 1, 24, 60, 128)
        # Each forward_scan is (24, 60, 128) quantized to uint8
        scans = np.zeros((self.window_size, 1, 24, 60, 128), dtype=np.float32)
        scan_valid = np.zeros(self.window_size, dtype=bool)
        for t, o in enumerate(obs_window):
            if o["fwd_bytes"] is not None:
                try:
                    arr = np.frombuffer(o["fwd_bytes"], dtype=np.uint8).reshape(24, 60, 128)
                    scans[t, 0] = arr.astype(np.float32) / 255.0
                    scan_valid[t] = True
                except (ValueError, TypeError):
                    pass

        # Label: multi-label binary target
        # label[i] = 1.0 if any catch of species i in (last_ts, last_ts + horizon_s]
        last_ts = obs_window[-1]["ts"]
        catches = self._catches.get(session_path, [])
        label = np.zeros(len(SPECIES), dtype=np.float32)
        for catch in catches:
            catch_ts = catch.get("ts", 0)
            if last_ts < catch_ts <= last_ts + self.horizon_s:
                try:
                    s_idx = SPECIES.index(catch["species"])
                    label[s_idx] = 1.0
                except (ValueError, KeyError):
                    pass

        if self.augment:
            scans, nav = self._augment(scans, nav, scan_valid)

        return {
            "scans": torch.from_numpy(scans),
            "scan_valid": torch.from_numpy(scan_valid),
            "nav": torch.from_numpy(nav),
            "label": torch.from_numpy(label),
        }

    def _augment(self, scans, nav, scan_valid):
        """
        Apply data augmentation:
        - Heading rotation: random azimuth shift (circular)
        - Speed jitter: multiply by uniform [0.85, 1.15]
        - Temporal flip: reverse time axis (50% probability)

        Args:
            scans: (T, 1, n_az, n_beam, n_range) scan array
            nav: (T, 7) navigation array
            scan_valid: (T,) boolean validity mask

        Returns:
            augmented (scans, nav)
        """
        # Heading rotation: random azimuth shift
        az_shift = np.random.randint(0, self.cfg.n_az)
        scans = np.roll(scans, az_shift, axis=2)

        # Update heading angle: az_shift of 1 corresponds to 360/n_az degrees
        heading_delta = (az_shift / self.cfg.n_az) * 360.0 - 180.0
        heading_rad = np.arctan2(nav[0, 4], nav[0, 5])
        heading_orig_deg = np.degrees(heading_rad)
        new_heading_deg = heading_orig_deg + heading_delta
        new_heading_rad = np.radians(new_heading_deg)
        nav[:, 4] = np.sin(new_heading_rad)
        nav[:, 5] = np.cos(new_heading_rad)

        # Speed jitter
        speed_factor = np.random.uniform(0.85, 1.15)
        nav[:, 3] *= speed_factor

        # Temporal flip
        if np.random.rand() < 0.5:
            scans = scans[::-1].copy()
            nav = nav[::-1].copy()

        return scans, nav
