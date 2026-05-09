#!/usr/bin/env python3
"""Quick test of captain agents and simulator."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from eval.environment import SteerableSimulator
from eval.captain import RandomCaptain, StraightCaptain, OracleCaptain, ModelGuidedCaptain

# Test SteerableSimulator
print("Testing SteerableSimulator...")
sim = SteerableSimulator(seed=0)
obs = sim.reset()
print(f"  Initial obs: east={obs.east_m:.1f}, north={obs.north_m:.1f}, depth={obs.depth_m:.1f}")
print(f"  Forward scan size: {len(obs.forward_scan) if obs.forward_scan else 0} bytes")

# Test one step
obs2, catches = sim.step(heading_delta_deg=5.0, speed_kts=3.5, dt=1.0)
print(f"  After 1 step: east={obs2.east_m:.1f}, north={obs2.north_m:.1f}")
print(f"  Catches: {len(catches)}")

# Test captains
print("\nTesting captains...")
sim = SteerableSimulator(seed=0)
obs = sim.reset()
state = {
    "east_m": obs.east_m,
    "north_m": obs.north_m,
    "heading_deg": obs.heading_deg,
    "speed_kts": obs.speed_kts,
    "depth_m": obs.depth_m,
    "t": obs.ts,
}

random_cap = RandomCaptain(seed=0)
delta, speed = random_cap.decide(state)
print(f"  RandomCaptain: delta={delta:.1f}, speed={speed:.1f}")

straight_cap = StraightCaptain()
delta, speed = straight_cap.decide(state)
print(f"  StraightCaptain: delta={delta:.1f}, speed={speed:.1f}")

model_cap = ModelGuidedCaptain(seed=0)
delta, speed = model_cap.decide(state)
print(f"  ModelGuidedCaptain (pre-60): delta={delta:.1f}, speed={speed:.1f}")

oracle_cap = OracleCaptain(schools=sim.fish_schools, seed=0)
delta, speed = oracle_cap.decide(state)
print(f"  OracleCaptain: delta={delta:.1f}, speed={speed:.1f}")

print("\nAll tests passed!")
