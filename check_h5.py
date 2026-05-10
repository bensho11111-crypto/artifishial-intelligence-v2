#!/usr/bin/env python3
import h5py
import os

h5_path = "data/training_300_cache.h5"
if not os.path.exists(h5_path):
    print(f"File not found: {h5_path}")
    exit(1)

print(f"Opening {h5_path}...")
with h5py.File(h5_path, 'r') as f:
    print(f"Keys in file: {list(f.keys())}")
    for key in f.keys():
        print(f"\n  {key}:")
        group = f[key]
        print(f"    Type: {type(group)}")
        if isinstance(group, h5py.Group):
            print(f"    Subkeys: {list(group.keys())}")
            for subkey in group.keys():
                dataset = group[subkey]
                print(f"      {subkey}: shape={dataset.shape}, dtype={dataset.dtype}")
