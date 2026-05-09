#!/usr/bin/env python3
"""Simple profiling of data generation bottlenecks."""
import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from synthetic.generator import generate
import math
import random
import json

print("=" * 60)
print("DATA GENERATION PROFILING (180s sessions)")
print("=" * 60)

# Profile 180s session (what we'll actually generate)
print("\n1. Synthetic world generation")
start = time.time()
session = generate(duration_s=180.0, seed=42)
gen_time = time.time() - start
print(f"   Time: {gen_time:.2f}s")
print(f"   Ticks: {len(session.ticks)}")

# Profile catch generation
print("\n2. Catch event processing")
from tools.generate_dataset import generate_catches, _enu_from_gps

world_state_rows = []
start = time.time()
for tick in session.ticks:
    if tick.gps:
        east_m, north_m = _enu_from_gps(tick.gps.lat, tick.gps.lon)
        world_state_rows.append({
            "ts": tick.gps.ts,
            "east_m": east_m,
            "north_m": north_m,
            "depth_m": tick.sonar.depth_m if tick.sonar else 0.0,
            "speed_kts": tick.gps.speed_kts,
            "heading_deg": tick.gps.heading_deg,
            "confidence": 0.9,
        })

enu_time = time.time() - start
print(f"   ENU conversion: {enu_time:.3f}s ({len(world_state_rows)} obs)")

start = time.time()
catches = generate_catches(world_state_rows, session, session.duration_s)
catch_time = time.time() - start
print(f"   Catch generation: {catch_time:.3f}s ({len(catches)} catches)")

# Profile disk I/O (Recorder not tested due to temp file issue)
# But we know from earlier: ~3-5 seconds for recording
recorder_est = 5.0
print(f"   Recorder (estimated): {recorder_est:.1f}s")

# Summary
print("\n" + "=" * 60)
print("ANALYSIS FOR 100 SESSIONS × 180s")
print("=" * 60)

time_per_session = gen_time + enu_time + catch_time + recorder_est
print(f"\nTime per session: {time_per_session:.1f}s")
print(f"  - World generation: {gen_time:.1f}s")
print(f"  - ENU conversion: {enu_time:.2f}s")
print(f"  - Catch generation: {catch_time:.2f}s")
print(f"  - Recorder (est): {recorder_est:.1f}s")

total_time = time_per_session * 100
print(f"\nTotal time for 100 sessions: {total_time:.0f}s = {total_time/3600:.1f}h")
print(f"Optimized (with parallelization): {total_time/4:.0f}s = {total_time/3600/4:.2f}h")

print("\n" + "=" * 60)
print("BOTTLENECK ANALYSIS")
print("=" * 60)
print("\nIdentified bottlenecks:")
print(f"  1. Recorder I/O (est): {recorder_est/time_per_session*100:.0f}%")
print(f"  2. World generation: {gen_time/time_per_session*100:.0f}%")
print(f"  3. Catch generation: {catch_time/time_per_session*100:.0f}%")
print(f"  4. ENU conversion: {enu_time/time_per_session*100:.0f}%")

print("\nOptimization opportunities:")
print("  ✓ Recorder: Batch writes (reduce syscalls)")
print("  ✓ Catch generation: Vectorize with NumPy")
print("  ✓ Parallelization: Generate multiple sessions in parallel")
print("  ✓ World generation: Pre-compute floor model (reuse across sessions)")
