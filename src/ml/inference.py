"""
src/ml/inference.py

Stateful ring-buffer inference engine for real-time fish catch prediction.

The InferenceEngine accumulates observations (navigation + sonar) until it has
a full window (60 ticks), then runs model inference and returns species
probabilities.
"""
from typing import Optional
import numpy as np
import torch

from ml.config import ModelConfig
from ml.model import FishCatchTransformer
from ml.dataset import encode_nav


class InferenceEngine:
    """
    Stateful ring-buffer inference engine for fish catch prediction.

    Accumulates observations until a full window (60 ticks) is filled, then
    runs model inference to predict catch probabilities for each species.
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Load a trained checkpoint and initialize ring buffers.

        Args:
            model_path: Path to .pt checkpoint (from training)
            device: 'cpu' or 'cuda'

        Checkpoint format:
            {
                "model_state_dict": {...},
                "config": {...ModelConfig dict...},
                "species": ["largemouth bass", "rainbow trout", "common carp", "bluegill bream"],
                "horizon_s": 300.0,
                ...
            }
        """
        # Load checkpoint
        ckpt = torch.load(model_path, map_location=device)

        # Reconstruct config and model
        cfg_dict = ckpt["config"]
        cfg = ModelConfig(**cfg_dict)

        self.model = FishCatchTransformer(cfg)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(device)
        self.model.eval()

        self.device = device
        self.cfg = cfg
        self.species = ckpt.get("species", cfg.species)
        self.horizon_s = ckpt.get("horizon_s", cfg.horizon_s)

        # Ring buffers (window_size = 60)
        self.window_size = cfg.window_size
        self.scans = np.zeros((cfg.window_size, 1, 24, 60, 128), dtype=np.float32)
        self.nav = np.zeros((cfg.window_size, 7), dtype=np.float32)
        self.scan_valid = np.zeros(cfg.window_size, dtype=bool)
        self.cursor = 0
        self.filled = 0  # number of ticks accumulated

    def push(self, obs, fwd_bytes: Optional[bytes]) -> Optional[dict]:
        """
        Add observation to ring buffer. Returns predictions when buffer is full.

        Args:
            obs: Object with east_m, north_m, depth_m, speed_kts, heading_deg, confidence
            fwd_bytes: Optional forward scan bytes (24*60*128 = 184320 bytes when uint8)

        Returns:
            None if buffer not yet full, or
            {
                "horizon_s": 300.0,
                "predictions": {
                    "largemouth bass": 0.73,
                    "rainbow trout": 0.45,
                    "common carp": 0.12,
                    "bluegill bream": 0.88
                }
            }
        """
        # Append to ring buffer at cursor position
        self.nav[self.cursor] = encode_nav(
            obs.east_m, obs.north_m, obs.depth_m,
            obs.speed_kts, obs.heading_deg, obs.confidence
        )

        # Decode and append scan
        if fwd_bytes is not None:
            try:
                arr = np.frombuffer(fwd_bytes, dtype=np.uint8).reshape(24, 60, 128)
                self.scans[self.cursor, 0] = arr.astype(np.float32) / 255.0
                self.scan_valid[self.cursor] = True
            except (ValueError, TypeError):
                self.scan_valid[self.cursor] = False
        else:
            self.scan_valid[self.cursor] = False

        # Advance cursor
        self.cursor = (self.cursor + 1) % self.window_size
        self.filled = min(self.filled + 1, self.window_size)

        # If not yet full, return None
        if self.filled < self.window_size:
            return None

        # Buffer is full — run inference
        # Reorder ring buffer to chronological order
        if self.cursor == 0:
            # Buffer wrapped exactly — already in order
            scans_ordered = self.scans.copy()
            nav_ordered = self.nav.copy()
            valid_ordered = self.scan_valid.copy()
        else:
            # Rotate: [cursor:] + [:cursor]
            scans_ordered = np.concatenate([
                self.scans[self.cursor:],
                self.scans[:self.cursor]
            ], axis=0)
            nav_ordered = np.concatenate([
                self.nav[self.cursor:],
                self.nav[:self.cursor]
            ], axis=0)
            valid_ordered = np.concatenate([
                self.scan_valid[self.cursor:],
                self.scan_valid[:self.cursor]
            ], axis=0)

        # Convert to tensors and run model
        with torch.no_grad():
            scans_t = torch.from_numpy(scans_ordered[None, ...]).to(self.device)  # (1, T, 1, 24, 60, 128)
            nav_t = torch.from_numpy(nav_ordered[None, ...]).to(self.device)      # (1, T, 7)
            valid_t = torch.from_numpy(valid_ordered[None, ...]).to(self.device)  # (1, T)

            logits = self.model(scans_t, valid_t, nav_t)  # (1, 4)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # (4,)

        # Return predictions
        predictions = {
            species: float(probs[i])
            for i, species in enumerate(self.species)
        }

        return {
            "horizon_s": self.horizon_s,
            "predictions": predictions,
        }

    def reset(self):
        """Clear all buffers (called on stream restart)."""
        self.scans[:] = 0
        self.nav[:] = 0
        self.scan_valid[:] = False
        self.cursor = 0
        self.filled = 0
