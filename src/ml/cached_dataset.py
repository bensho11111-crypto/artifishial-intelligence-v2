"""
Cached synthetic dataset loader for NPZ or HDF5-backed training.
"""
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class CachedSyntheticDataset(Dataset):
    """Load training samples from NPZ or HDF5 cache."""

    def __init__(self, cache_path: str, augment: bool = False):
        """
        Args:
            cache_path: Path to NPZ or HDF5 cache file
            augment: Whether to apply augmentation (heading rotation, speed jitter)
        """
        self.cache_path = Path(cache_path)
        self.augment = augment

        # Load data (NPZ or HDF5)
        if str(cache_path).endswith(".npz"):
            self.data = np.load(cache_path)
            self.n_samples = len(self.data["scans"])
            self.is_npz = True
        else:
            # HDF5 mode (lazy loading)
            with h5py.File(self.cache_path, "r") as f:
                self.n_samples = f.attrs.get("n_samples", len(f["scans"]))
            self.is_npz = False
            self.data = None

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int):
        """Load sample from NPZ or HDF5."""
        if self.is_npz:
            scans = torch.from_numpy(self.data["scans"][idx]).float()  # (60, 1, 24, 60, 128)
            valids = torch.from_numpy(self.data["valids"][idx]).bool()  # (60,)
            navs = torch.from_numpy(self.data["navs"][idx]).float()     # (60, 7)
            labels = torch.from_numpy(self.data["labels"][idx]).float() # (4,)
        else:
            with h5py.File(self.cache_path, "r") as f:
                scans = torch.from_numpy(f["scans"][idx]).float()
                valids = torch.from_numpy(f["valids"][idx]).bool()
                navs = torch.from_numpy(f["navs"][idx]).float()
                labels = torch.from_numpy(f["labels"][idx]).float()

        # Optional: augmentation (heading rotation, speed jitter, temporal flip)
        if self.augment:
            # Heading rotation: roll azimuth axis and update sin/cos
            if np.random.rand() < 0.5:
                az_roll = np.random.randint(1, 24)
                scans = torch.roll(scans, az_roll, dims=2)  # Roll azimuth axis

                # Update heading in nav: add az_roll * 360/24 degrees
                heading_shift_deg = az_roll * 360.0 / 24
                for t in range(navs.shape[0]):
                    heading_rad = np.arccos(navs[t, 5].item())  # cos(heading) stored
                    heading_deg = np.degrees(heading_rad)
                    heading_deg = (heading_deg + heading_shift_deg) % 360
                    heading_rad = np.radians(heading_deg)
                    navs[t, 4] = torch.tensor(np.sin(heading_rad), dtype=torch.float32)
                    navs[t, 5] = torch.tensor(np.cos(heading_rad), dtype=torch.float32)

            # Speed jitter
            if np.random.rand() < 0.5:
                speed_factor = np.random.uniform(0.85, 1.15)
                navs[:, 3] *= speed_factor

            # Temporal flip
            if np.random.rand() < 0.5:
                scans = torch.flip(scans, dims=[0])
                navs = torch.flip(navs, dims=[0])
                valids = torch.flip(valids, dims=[0])

        return {
            "scans": scans,
            "valids": valids,
            "navs": navs,
            "labels": labels,
        }


def collate_batch(batch):
    """Collate function for DataLoader."""
    scans = torch.stack([item["scans"] for item in batch])
    valids = torch.stack([item["valids"] for item in batch])
    navs = torch.stack([item["navs"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])

    return {
        "scans": scans,
        "valids": valids,
        "navs": navs,
        "labels": labels,
    }


def get_cached_dataloader(
    cache_path: str,
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle: bool = True,
    augment: bool = False,
) -> DataLoader:
    """
    Create a DataLoader from HDF5 cache.

    Args:
        cache_path: Path to HDF5 cache file
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        augment: Whether to apply augmentation

    Returns:
        DataLoader instance
    """
    dataset = CachedSyntheticDataset(cache_path, augment=augment)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_batch,
        pin_memory=True,
    )
