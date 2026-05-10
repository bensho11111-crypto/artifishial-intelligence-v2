#!/usr/bin/env python3
import json

files = [
    "eval_larger_dataset.json",
    "eval_approach_detection.json",
    "eval_loss_tuning.json",
    "eval_class_resampling.json"
]

print("Checking evaluation results...\n")
for fname in files:
    with open(fname) as f:
        data = json.load(f)

    # Check first 3 episodes
    print(f"{fname}:")
    print(f"  Model: {data['model_path']}")
    print(f"  Captain score: {data['captain_score']}")
    print(f"  Oracle fraction: {data['oracle_fraction']:.4f}")

    # Check per-episode catches for ModelGuidedCaptain
    m_catches = [e['catches_by_species'].get('ModelGuidedCaptain', 0) for e in data['episodes'][:3]]
    print(f"  ModelGuidedCaptain catches (ep 0-2): {m_catches}")
    print()
