#!/usr/bin/env python3
"""Visualize training data distribution and samples."""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ml.config import ModelConfig
from ml.dataset import FishCatchDataset

print("Loading dataset...")
cfg = ModelConfig()
train_ds = FishCatchDataset("data/training_300/", cfg, augment=False, val_fraction=0.15, train=True)

# Extract data (use subset to avoid memory issues)
all_labels = train_ds._h5_group['labels'][:]  # (N, 4) - small, load all
all_nav = train_ds._h5_group['nav'][:]  # (N, 60, 7) - manageable, load all
# Don't load all scans at once - they're huge (315 GB total!)

n_samples = all_labels.shape[0]
species = ["Largemouth Bass", "Rainbow Trout", "Common Carp", "Bluegill Bream"]

print(f"Dataset size: {n_samples:,} windows")
print(f"Each window: 60 ticks (1 minute)")
print()

# ===== FIGURE 1: Label Distribution =====
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Training Data Analysis', fontsize=16, fontweight='bold')

# 1a: Positive rate by species
pos_counts = all_labels.sum(axis=0)
pos_rates = pos_counts / n_samples * 100
ax = axes[0, 0]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
bars = ax.bar(range(4), pos_rates, color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('Positive Rate (%)', fontweight='bold')
ax.set_title('Positive Label Rate by Species')
ax.set_xticks(range(4))
ax.set_xticklabels(species, rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)
for i, (bar, rate) in enumerate(zip(bars, pos_rates)):
    ax.text(i, rate + 0.1, f'{rate:.2f}%', ha='center', fontweight='bold')

# 1b: Multi-label distribution
ax = axes[0, 1]
n_species_per_window = all_labels.sum(axis=1)
counts, bins = np.histogram(n_species_per_window, bins=np.arange(0, 6) - 0.5, density=False)
ax.bar(range(5), counts, color='skyblue', alpha=0.7, edgecolor='black')
ax.set_xlabel('Number of Species Caught', fontweight='bold')
ax.set_ylabel('Number of Windows', fontweight='bold')
ax.set_title('Multi-Label Distribution')
ax.set_xticks(range(5))
ax.grid(axis='y', alpha=0.3)
for i, count in enumerate(counts):
    ax.text(i, count + 20, f'{int(count)}\n({count/n_samples*100:.1f}%)', ha='center', fontsize=9)

# 1c: Navigation data distribution (speed)
ax = axes[1, 0]
speeds = all_nav[:, :, 3].flatten()  # speed_kts is index 3
speeds_unnormalized = speeds * 15  # denormalize
ax.hist(speeds_unnormalized, bins=50, color='green', alpha=0.7, edgecolor='black')
ax.set_xlabel('Speed (knots)', fontweight='bold')
ax.set_ylabel('Frequency', fontweight='bold')
ax.set_title('Speed Distribution')
ax.grid(axis='y', alpha=0.3)
ax.axvline(speeds_unnormalized.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {speeds_unnormalized.mean():.1f}')
ax.legend()

# 1d: Navigation data distribution (heading)
ax = axes[1, 1]
headings = np.arctan2(all_nav[:, :, 4], all_nav[:, :, 5]).flatten()  # atan2(sin, cos)
headings_deg = np.degrees(headings) % 360
ax.hist(headings_deg, bins=36, color='orange', alpha=0.7, edgecolor='black')
ax.set_xlabel('Heading (degrees)', fontweight='bold')
ax.set_ylabel('Frequency', fontweight='bold')
ax.set_title('Heading Distribution')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('training_data_overview.png', dpi=150, bbox_inches='tight')
print("Saved: training_data_overview.png")

# ===== FIGURE 2: Sample Sonar Data =====
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Sample Sonar Scans (Forward Scan Data)', fontsize=16, fontweight='bold')

# Find samples with different label counts
samples_to_show = []
for n_sp in range(3):
    mask = n_species_per_window == n_sp
    indices = np.where(mask)[0]
    if len(indices) > 0:
        samples_to_show.append(indices[0])

# Show samples at different time steps (load only what we need)
for idx, sample_idx in enumerate(samples_to_show[:2]):  # Only first 2 samples
    for t, time_step in enumerate([0, 30, 59]):  # beginning, middle, end
        ax = axes[idx, t]
        # Load only this one sample from HDF5
        scan = train_ds._h5_group['scans'][sample_idx, time_step, 0, :, :, :]  # (24, 60, 128)

        # Visualize as a 2D image: range bins vs azimuth
        scan_2d = scan.max(axis=1)  # Max across beams for visualization
        im = ax.imshow(scan_2d, cmap='viridis', aspect='auto', origin='lower')

        n_sp = int(n_species_per_window[sample_idx])
        ax.set_title(f'Sample {sample_idx}, t={time_step}s\n{n_sp} species caught', fontsize=10)
        ax.set_xlabel('Range bins (0-128)')
        ax.set_ylabel('Azimuth (0-24)')
        plt.colorbar(im, ax=ax, label='Amplitude')

plt.tight_layout()
plt.savefig('sample_sonar_scans.png', dpi=150, bbox_inches='tight')
print("Saved: sample_sonar_scans.png")

# ===== FIGURE 3: Statistics =====
fig = plt.figure(figsize=(12, 8))
fig.suptitle('Dataset Statistics', fontsize=16, fontweight='bold')

# Create a text summary
stats_text = f"""
DATASET STATISTICS (Training Split)

Total Windows: {n_samples:,}
Total Ticks: {n_samples * 60:,} (at 1 Hz)
Time Covered: {n_samples / (60 * 60 * 24):.1f} days

LABEL DISTRIBUTION:
  Total Positive Windows: {(all_labels.max(axis=1) > 0).sum():,} ({(all_labels.max(axis=1) > 0).sum()/n_samples*100:.1f}%)
  Total Negative Windows: {(all_labels.max(axis=1) == 0).sum():,} ({(all_labels.max(axis=1) == 0).sum()/n_samples*100:.1f}%)

BY SPECIES:
  {species[0]}: {pos_counts[0]:,} catches ({pos_rates[0]:.2f}%)
  {species[1]}: {pos_counts[1]:,} catches ({pos_rates[1]:.2f}%)
  {species[2]}: {pos_counts[2]:,} catches ({pos_rates[2]:.2f}%)
  {species[3]}: {pos_counts[3]:,} catches ({pos_rates[3]:.2f}%)

NAVIGATION RANGES:
  Speed: {speeds_unnormalized.min():.1f} - {speeds_unnormalized.max():.1f} knots (mean: {speeds_unnormalized.mean():.1f})
  Depth: {all_nav[:, :, 2].min()*32:.1f} - {all_nav[:, :, 2].max()*32:.1f} meters
  Position: East/North normalized to [-1, 1]

IMBALANCE RATIO:
  Negative:Positive = {(all_labels.max(axis=1) == 0).sum():,} : {(all_labels.max(axis=1) > 0).sum():,}
  (~{(all_labels.max(axis=1) == 0).sum() / (all_labels.max(axis=1) > 0).sum():.1f}:1)

MODEL CHALLENGE:
  With {pos_rates[0]:.1f}% label sparsity, predicting "no catch" for everything
  gives {100-pos_rates[0]:.1f}% accuracy without learning anything!
"""

ax = fig.add_subplot(111)
ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontfamily='monospace',
        fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round',
        facecolor='wheat', alpha=0.5))
ax.axis('off')

plt.tight_layout()
plt.savefig('dataset_statistics.png', dpi=150, bbox_inches='tight')
print("Saved: dataset_statistics.png")

# Print summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Positive rate: {(all_labels.max(axis=1) > 0).sum()/n_samples*100:.2f}%")
print(f"Imbalance ratio: {(all_labels.max(axis=1) == 0).sum() / (all_labels.max(axis=1) > 0).sum():.1f}:1")
print(f"\nKey insight: The model needs to learn from just {(all_labels.max(axis=1) > 0).sum():,}")
print(f"positive examples out of {n_samples:,} total - this is EXTREMELY sparse!")
print("="*70)

plt.show()
