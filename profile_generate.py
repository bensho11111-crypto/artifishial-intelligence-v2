#!/usr/bin/env python3
"""Profile data generation to identify bottlenecks."""
import sys
import os
import time
import tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from synthetic.generator import generate
from synthetic.forward_scan import generate as fwd_gen
from ticks.recorder import Recorder
import math
import random

# Test 1: Time the generate() function
print("=" * 60)
print("PROFILING DATA GENERATION")
print("=" * 60)

print("\n1. Synthetic world generation (generate())")
start = time.time()
session = generate(duration_s=90.0, seed=42)
gen_time = time.time() - start
print(f"   Time: {gen_time:.2f}s for 90s session")
print(f"   Ticks: {len(session.ticks)} ({len(session.ticks)/gen_time:.0f} ticks/sec)")

# Test 2: Time Recorder I/O
print("\n2. Tick recording (Recorder)")
with tempfile.NamedTemporaryFile(suffix=".ticks", delete=True) as f:
    start = time.time()
    with Recorder(f.name) as rec:
        rec.set_metadata("test", "1")
        for tick in session.ticks:
            rec.record(tick)
    io_time = time.time() - start
print(f"   Time: {io_time:.2f}s for {len(session.ticks)} ticks")
print(f"   Rate: {len(session.ticks)/io_time:.0f} ticks/sec")

# Test 3: Time sonar generation
print("\n3. Forward scan generation")
start = time.time()
scan_count = 0
for tick in session.ticks[:50]:  # Sample first 50
    if tick.gps:
        e, n = 0, 0
        fwd_bytes = fwd_gen(e, n, 0, session.floor, [], random.Random(0))
        scan_count += 1
sonar_time = time.time() - start
print(f"   Time: {sonar_time:.3f}s for {scan_count} scans")
print(f"   Rate: {scan_count/sonar_time:.1f} scans/sec")

# Test 4: Catch generation
print("\n4. Catch event generation (Poisson)")
from tools.generate_dataset import generate_catches

world_state_rows = []
for i, tick in enumerate(session.ticks):
    if tick.gps and i % 5 == 0:  # GPS at 1 Hz, sampled here
        from tools.generate_dataset import _enu_from_gps
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

start = time.time()
catches = generate_catches(world_state_rows, session, session.duration_s)
catch_time = time.time() - start
print(f"   Time: {catch_time:.3f}s for {len(world_state_rows)} observations")
print(f"   Rate: {len(world_state_rows)/catch_time:.0f} obs/sec")
print(f"   Catches generated: {len(catches)}")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
total_time = gen_time + io_time + sonar_time + catch_time
print(f"Total per 90s session: {total_time:.2f}s")
print(f"  generate():      {gen_time:.2f}s ({gen_time/total_time*100:.0f}%)")
print(f"  Recorder I/O:    {io_time:.2f}s ({io_time/total_time*100:.0f}%)")
print(f"  Sonar (sample):  {sonar_time:.2f}s ({sonar_time/total_time*100:.0f}%)")
print(f"  Catch events:    {catch_time:.2f}s ({catch_time/total_time*100:.0f}%)")
print(f"\nEstimated time for 100 sessions @ 180s each:")
time_per_session = (gen_time / 90) * 180 + io_time * 2 + catch_time * 2
total_est = time_per_session * 100
print(f"  Per session: {time_per_session:.1f}s")
print(f"  Total: {total_est:.0f}s ({total_est/3600:.1f}h)")
