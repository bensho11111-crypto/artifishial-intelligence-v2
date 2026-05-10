#!/usr/bin/env python3
"""Sanity check sonar data - investigate why scans look empty."""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ml.config import ModelConfig
from ml.dataset import FishCatchDataset

print("Loading dataset...\n")
cfg = ModelConfig()
train_ds = FishCatchDataset("data/training_300/", cfg, augment=False, val_fraction=0.15, train=True)

all_labels = train_ds._h5_group['labels'][:]
n_species_per_window = all_labels.sum(axis=1)

print("="*70)
print("SONAR DATA SANITY CHECK")
print("="*70)

# Find samples: one with 0 catches, one with catches
idx_no_catch = np.where(n_species_per_window == 0)[0][0]
idx_with_catch = np.where(n_species_per_window > 0)[0][0]

print(f"\nSample WITHOUT catches: index {idx_no_catch}")
print(f"Sample WITH catches: index {idx_with_catch}, species count: {int(n_species_per_window[idx_with_catch])}")

# Load the actual scan data (lazy load to avoid memory issues)
print(f"\nLoading sonar scans...")

# Check the data type and range BEFORE visualization
for label, idx in [("WITHOUT catches", idx_no_catch), ("WITH catches", idx_with_catch)]:
    print(f"\n{label.upper()} (Sample {idx}):")

    for t in [0, 30, 59]:
        scan = train_ds._h5_group['scans'][idx, t, 0, :, :, :]  # (24, 60, 128)

        print(f"  Time {t}s:")
        print(f"    Shape: {scan.shape}")
        print(f"    Data type: {scan.dtype}")
        print(f"    Min: {scan.min()}, Max: {scan.max()}, Mean: {scan.mean():.2f}")
        print(f"    Non-zero values: {(scan > 0).sum()}/{scan.size} ({(scan > 0).sum()/scan.size*100:.1f}%)")

        # Find where the maximum values are
        max_idx = np.unravel_index(scan.argmax(), scan.shape)
        print(f"    Max location (az, beam, range): {max_idx}, value: {scan[max_idx]}")

# ===== VISUALIZATION: Compare with and without catches =====
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('Sonar Data Comparison: WITH vs WITHOUT Fish Catches', fontsize=14, fontweight='bold')

for idx, label in [(idx_no_catch, "NO catches"), (idx_with_catch, "WITH catches")]:
    row = 0 if idx == idx_no_catch else 1

    for t, time_step in enumerate([0, 20, 40, 59]):
        scan = train_ds._h5_group['scans'][idx, time_step, 0, :, :, :]

        # Show multiple representations
        ax = axes[row, t]

        # Combine across beams to see range pattern
        scan_2d = scan.max(axis=1)  # Max across 60 beams

        # Use linear scale (not log) to see actual values
        im = ax.imshow(scan_2d, cmap='viridis', aspect='auto', origin='lower')

        ax.set_title(f'{label}, t={time_step}s\nMin={scan.min()} Max={scan.max()}', fontsize=10)
        ax.set_xlabel('Range (0-128)')
        ax.set_ylabel('Azimuth (0-24)')

        # Add colorbar with actual values
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Amplitude')

plt.tight_layout()
plt.savefig('sonar_sanity_check.png', dpi=150, bbox_inches='tight')
print("\n\nSaved: sonar_sanity_check.png")

# ===== DETAILED HISTOGRAM =====
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Sonar Amplitude Distribution', fontsize=14, fontweight='bold')

for idx, label in [(idx_no_catch, "NO catches"), (idx_with_catch, "WITH catches")]:
    scan = train_ds._h5_group['scans'][idx, 30, 0, :, :, :]  # mid-window scan
    ax = axes[0 if idx == idx_no_catch else 1]

    # Flatten and show distribution
    amplitudes = scan.flatten()

    ax.hist(amplitudes, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Amplitude Value', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title(f'{label} - Amplitude Distribution (t=30s)')
    ax.grid(axis='y', alpha=0.3)

    # Add statistics
    stats_text = f'Mean: {amplitudes.mean():.2f}\nStd: {amplitudes.std():.2f}\nMax: {amplitudes.max()}\nZeros: {(amplitudes == 0).sum()}'
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('sonar_amplitude_distribution.png', dpi=150, bbox_inches='tight')
print("Saved: sonar_amplitude_distribution.png")

# ===== CHECK DATA LOADING =====
print("\n" + "="*70)
print("DATA LOADING CHECK")
print("="*70)

# Verify HDF5 dataset shape
print(f"\nHDF5 dataset shapes:")
print(f"  scans: {train_ds._h5_group['scans'].shape}")
print(f"  labels: {train_ds._h5_group['labels'].shape}")
print(f"  nav: {train_ds._h5_group['nav'].shape}")

# Check if there are any high-amplitude regions in the entire dataset
print(f"\nScanning entire dataset for high amplitudes...")
sample_indices = np.random.choice(len(train_ds), 100, replace=False)
max_amplitudes = []

for idx in sample_indices:
    for t in [0, 30, 59]:
        scan = train_ds._h5_group['scans'][idx, t, 0, :, :, :]
        max_amplitudes.append(scan.max())

max_amplitudes = np.array(max_amplitudes)
print(f"  Max amplitude across 300 scans: {max_amplitudes.max()}")
print(f"  Mean of max amplitudes: {max_amplitudes.mean():.2f}")
print(f"  Median of max amplitudes: {np.median(max_amplitudes):.2f}")
print(f"  Min of max amplitudes: {max_amplitudes.min()}")

# Distribution of max amplitudes
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(max_amplitudes, bins=50, color='coral', alpha=0.7, edgecolor='black')
ax.set_xlabel('Maximum Amplitude per Scan', fontweight='bold')
ax.set_ylabel('Frequency', fontweight='bold')
ax.set_title('Distribution of Maximum Amplitudes (300 random scans)')
ax.grid(axis='y', alpha=0.3)
ax.axvline(max_amplitudes.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {max_amplitudes.mean():.2f}')
ax.legend()
plt.tight_layout()
plt.savefig('max_amplitude_distribution.png', dpi=150, bbox_inches='tight')
print("Saved: max_amplitude_distribution.png")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
if max_amplitudes.mean() < 1:
    print("[WARNING] Sonar amplitudes are very small (mean < 1)")
    print("   This could indicate:")
    print("   1. Data is normalized to [0, 1] range")
    print("   2. The sonar gain is set too low")
    print("   3. The data is genuinely weak")
    print("   4. The data is corrupted or never populated")
else:
    print("[OK] Sonar amplitudes appear reasonable")
