#!/usr/bin/env python3
"""Debug script to inspect synthetic generator output."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from synthetic.generator import generate

session = generate(duration_s=90.0, seed=42)

print("=== Session Info ===")
print(f"Duration: {session.duration_s}s")
print(f"Fish schools: {len(session.fish_schools)}")
for i, school in enumerate(session.fish_schools):
    s = school.at(45.0)  # midpoint
    print(f"  School {i}: east={s.east_m:.1f}, north={s.north_m:.1f}, radius={s.radius_m:.1f}, density={s.density}")

print(f"\nFloor model: {session.floor}")

print(f"\nTicks: {len(session.ticks)}")
found_gps = 0
for i, tick in enumerate(session.ticks):
    if tick.gps is not None:
        found_gps += 1
        if i < 5 or i == 45 or i == 90:
            print(f"  Tick {i}: ts={tick.gps.ts:.1f}, lat={tick.gps.lat:.3f}, lon={tick.gps.lon:.3f}")
            print(f"    heading={tick.gps.heading_deg:.1f}, speed={tick.gps.speed_kts:.2f}")

print(f"\nTotal GPS ticks: {found_gps}")

# Check what attributes the session has
print(f"\nSession attributes:")
for attr in dir(session):
    if not attr.startswith('_'):
        val = getattr(session, attr)
        if not callable(val):
            print(f"  {attr}: {type(val).__name__}")
