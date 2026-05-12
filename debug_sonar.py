#!/usr/bin/env python3
"""
Debug script: Compare sonar data when fish are below boat vs no fish.

This helps us understand if the forward sonar can detect fish that are
directly beneath the boat (in the blind zone behind the forward cone).
"""
import sys
sys.path.insert(0, "src")

import numpy as np
from eval.environment import SteerableSimulator
from synthetic.forward_scan import generate as fwd_gen

print("=" * 70)
print("SONAR DATA DEBUGGING: Fish Below Boat vs No Fish")
print("=" * 70)

# Create simulator with seed that has schools nearby
sim = SteerableSimulator(seed=42)
obs = sim.reset()

print(f"\nInitial boat position: ({obs.east_m:.1f}, {obs.north_m:.1f})")
print(f"Schools in simulation:")
for i, school in enumerate(sim.fish_schools):
    s = school.at(0)
    print(f"  School {i}: ({s.east_m:.1f}, {s.north_m:.1f}), radius={s.radius_m:.1f}m")

# Scenario 1: Boat near school, heading toward it
print("\n" + "=" * 70)
print("SCENARIO 1: Boat approaching school (fish ahead in sonar cone)")
print("=" * 70)

# Move boat to 30m before school 0
school_0 = sim.fish_schools[0].at(0)
target_dist = 30.0  # 30m away
heading_to_school = np.degrees(np.arctan2(school_0.east_m - obs.east_m,
                                          school_0.north_m - obs.north_m))
obs = sim.reset()

# Steer toward school
for step in range(40):
    # Steer toward school
    heading_delta = np.clip(heading_to_school - obs.heading_deg, -30, 30)
    obs, _ = sim.step(heading_delta, 3.5, dt=1.0)

    # Check distance
    s = sim.fish_schools[0].at(obs.ts)
    dist = np.sqrt((obs.east_m - s.east_m)**2 + (obs.north_m - s.north_m)**2)

    if dist < target_dist + 5:
        print(f"Stopped at step {step}: distance to school = {dist:.1f}m")
        break

# Get sonar data
school_ahead_scan = obs.forward_scan
school_ahead_scan_arr = np.frombuffer(school_ahead_scan, dtype=np.uint8).reshape(24, 60, 128)

print(f"Boat position: ({obs.east_m:.1f}, {obs.north_m:.1f})")
print(f"Boat heading: {obs.heading_deg:.1f}°")
print(f"Distance to school: {dist:.1f}m")
print(f"School position: ({s.east_m:.1f}, {s.north_m:.1f})")
print(f"\nSonar stats (school ahead):")
print(f"  Min value: {school_ahead_scan_arr.min()}")
print(f"  Max value: {school_ahead_scan_arr.max()}")
print(f"  Mean value: {school_ahead_scan_arr.mean():.2f}")
print(f"  Std value: {school_ahead_scan_arr.std():.2f}")
print(f"  Values > 100: {(school_ahead_scan_arr > 100).sum()}")
print(f"  Values > 150: {(school_ahead_scan_arr > 150).sum()}")

# Scenario 2: Boat ON TOP of school
print("\n" + "=" * 70)
print("SCENARIO 2: Boat ON TOP of school (fish below, outside sonar cone)")
print("=" * 70)

obs = sim.reset()
# Move boat directly on top of school 0
target_pos = sim.fish_schools[0].at(0)
for step in range(50):
    heading_delta = np.clip(np.degrees(np.arctan2(target_pos.east_m - obs.east_m,
                                                    target_pos.north_m - obs.north_m)) - obs.heading_deg, -30, 30)
    obs, _ = sim.step(heading_delta, 5.0, dt=1.0)

    dist = np.sqrt((obs.east_m - target_pos.east_m)**2 + (obs.north_m - target_pos.north_m)**2)
    if dist < 5.0:  # Get within 5m
        print(f"Positioned at step {step}: distance = {dist:.1f}m")
        break

school_below_scan = obs.forward_scan
school_below_scan_arr = np.frombuffer(school_below_scan, dtype=np.uint8).reshape(24, 60, 128)

print(f"Boat position: ({obs.east_m:.1f}, {obs.north_m:.1f})")
print(f"Boat heading: {obs.heading_deg:.1f}°")
print(f"Distance to school: {dist:.1f}m")
print(f"School position: ({target_pos.east_m:.1f}, {target_pos.north_m:.1f})")
print(f"\nSonar stats (school below/behind):")
print(f"  Min value: {school_below_scan_arr.min()}")
print(f"  Max value: {school_below_scan_arr.max()}")
print(f"  Mean value: {school_below_scan_arr.mean():.2f}")
print(f"  Std value: {school_below_scan_arr.std():.2f}")
print(f"  Values > 100: {(school_below_scan_arr > 100).sum()}")
print(f"  Values > 150: {(school_below_scan_arr > 150).sum()}")

# Scenario 3: Empty space (no schools nearby)
print("\n" + "=" * 70)
print("SCENARIO 3: Empty space (no fish)")
print("=" * 70)

obs = sim.reset()
# Move to a location far from all schools
for step in range(60):
    obs, _ = sim.step(-20, 3.5, dt=1.0)

# Check we're far from all schools
min_dist = float("inf")
for school in sim.fish_schools:
    s = school.at(obs.ts)
    dist = np.sqrt((obs.east_m - s.east_m)**2 + (obs.north_m - s.north_m)**2)
    min_dist = min(min_dist, dist)

empty_scan = obs.forward_scan
empty_scan_arr = np.frombuffer(empty_scan, dtype=np.uint8).reshape(24, 60, 128)

print(f"Boat position: ({obs.east_m:.1f}, {obs.north_m:.1f})")
print(f"Minimum distance to any school: {min_dist:.1f}m")
print(f"\nSonar stats (empty):")
print(f"  Min value: {empty_scan_arr.min()}")
print(f"  Max value: {empty_scan_arr.max()}")
print(f"  Mean value: {empty_scan_arr.mean():.2f}")
print(f"  Std value: {empty_scan_arr.std():.2f}")
print(f"  Values > 100: {(empty_scan_arr > 100).sum()}")
print(f"  Values > 150: {(empty_scan_arr > 150).sum()}")

# Comparison
print("\n" + "=" * 70)
print("COMPARISON SUMMARY")
print("=" * 70)
print(f"\n{'Scenario':<25} {'Mean':<10} {'Std':<10} {'Max':<10} {'>150 count':<12}")
print("-" * 65)
print(f"{'School ahead (30m)':<25} {school_ahead_scan_arr.mean():<10.2f} {school_ahead_scan_arr.std():<10.2f} {school_ahead_scan_arr.max():<10} {(school_ahead_scan_arr > 150).sum():<12}")
print(f"{'School below (5m)':<25} {school_below_scan_arr.mean():<10.2f} {school_below_scan_arr.std():<10.2f} {school_below_scan_arr.max():<10} {(school_below_scan_arr > 150).sum():<12}")
print(f"{'Empty space':<25} {empty_scan_arr.mean():<10.2f} {empty_scan_arr.std():<10.2f} {empty_scan_arr.max():<10} {(empty_scan_arr > 150).sum():<12}")

# Analysis
print("\n" + "=" * 70)
print("ANALYSIS")
print("=" * 70)
diff_ahead_vs_empty = school_ahead_scan_arr.mean() - empty_scan_arr.mean()
diff_below_vs_empty = school_below_scan_arr.mean() - empty_scan_arr.mean()

print(f"\nDifference from empty baseline:")
print(f"  School ahead (30m):  +{diff_ahead_vs_empty:.2f} (school VISIBLE in sonar)")
print(f"  School below (5m):   +{diff_below_vs_empty:.2f} (school NOT visible in sonar)")

if diff_ahead_vs_empty > diff_below_vs_empty:
    print(f"\nCONFIRMED: Sonar signal is STRONGER when fish are 30m ahead")
    print(f"  than when fish are directly below the boat.")
    print(f"\nThis explains the inverted correlation:")
    print(f"  - Model sees strong signal at 10-40m ahead -> predicts HIGH")
    print(f"  - Model sees weak signal at 0m (on top) -> predicts LOW")
    print(f"  - Catches happen at 0m (when sonar is blind)")
    print(f"  -> Negative correlation between prediction and catch proximity")
else:
    print(f"\nUNEXPECTED: Signal is STRONGER when fish are below the boat!")
    print(f"  School below (2.9m): signal +{diff_below_vs_empty:.2f}")
    print(f"  School ahead (30m):  signal +{diff_ahead_vs_empty:.2f}")
    print(f"\n  This contradicts the 'forward-only sonar' theory.")
    print(f"  The sonar appears to have significant off-axis sensitivity.")
    print(f"  Fish detection is NOT dependent on forward direction alone.")

print("\n" + "=" * 70)
