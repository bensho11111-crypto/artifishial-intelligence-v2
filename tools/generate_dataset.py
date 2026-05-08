"""
tools/generate_dataset.py

Generate synthetic .ticks + _catches.json training datasets.

Creates pairs of recordings and ground-truth catch labels for training the
fish detection model. Uses the synthetic session generator to create
deterministic, reproducible fishing scenarios.
"""
import argparse
import sys
import os
import json
import math
import random
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from synthetic.generator import generate, SPECIES_NAMES, START_LAT, START_LON
from ticks.recorder import Recorder


def _enu_from_gps(lat, lon):
    """Convert GPS lat/lon to metric ENU coordinates."""
    R = 6_371_000.0
    north = R * math.radians(lat - START_LAT)
    east = R * math.radians(lon - START_LON) * math.cos(math.radians(START_LAT))
    return east, north


def generate_catches(world_state_rows, session, duration_s):
    """
    Generate Poisson-based catch events from fish school proximity.

    For each second of the session, checks if the boat is within any fish school.
    If so, generates catch events according to a Poisson process with rate
    proportional to overlap and school density.

    Args:
        world_state_rows: list of dicts with keys ts, east_m, north_m, depth_m, speed_kts, heading_deg, confidence
        session: GeneratedSession with fish_schools attribute
        duration_s: session duration in seconds

    Returns:
        list of {"ts": float, "species": str} catch records
    """
    catches = []

    for t in range(int(duration_s)):
        # Find observation nearest to this second
        obs_t = None
        for row in world_state_rows:
            if row["ts"] <= t < row["ts"] + 1:
                obs_t = row
                break
        if obs_t is None:
            continue

        boat_e = obs_t["east_m"]
        boat_n = obs_t["north_m"]

        for school in session.fish_schools:
            # Get school position at time t
            s = school.at(float(t))
            dist = math.sqrt((boat_e - s.east_m)**2 + (boat_n - s.north_m)**2)

            if dist < s.radius_m:
                # Overlap factor: 1.0 at center, 0.0 at edge
                overlap = 1.0 - dist / s.radius_m
                # Poisson rate: density * overlap * 2.0 for better training signal
                lam = s.density * overlap * 2.0

                # Poisson process: P(catch) = 1 - exp(-lam)
                if random.random() < 1.0 - math.exp(-lam):
                    catches.append({
                        "ts": float(t),
                        "species": SPECIES_NAMES.get(s.species, s.species),
                    })

    return catches


def main():
    p = argparse.ArgumentParser(
        description="Generate synthetic .ticks + _catches.json training dataset pairs"
    )
    p.add_argument("out_dir", help="Output directory for .ticks and _catches.json files")
    p.add_argument("--n-sessions", type=int, default=20, help="Number of sessions to generate")
    p.add_argument("--duration", type=float, default=120.0, help="Session duration in seconds")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = p.parse_args()

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for i in range(args.n_sessions):
        print(f"[{i+1}/{args.n_sessions}] Generating session {i}...")
        random.seed(args.seed + i)
        np.random.seed(args.seed + i)

        # Generate synthetic session
        session = generate(duration_s=args.duration, seed=args.seed + i)

        session_name = f"synthetic_{i:04d}"
        ticks_path = out_path / f"{session_name}.ticks"
        catches_path = out_path / f"{session_name}_catches.json"

        # Write .ticks file with all ticks from the session
        with Recorder(str(ticks_path)) as rec:
            rec.set_metadata("source", "synthetic")
            rec.set_metadata("seed", str(args.seed + i))
            for tick in session.ticks:
                rec.record(tick)

        # Collect world state observations for catch generation
        # Convert GPS lat/lon to metric ENU coordinates
        world_state_rows = []
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
                    "confidence": 0.9,  # synthetic → high confidence
                })

        # Generate and write catches.json
        catches = generate_catches(world_state_rows, session, session.duration_s)

        with open(catches_path, "w") as f:
            json.dump({
                "session_id": session_name,
                "horizon_s": 300,
                "catches": catches,
            }, f, indent=2)

        print(f"  -> {ticks_path} ({len(world_state_rows)} observations)")
        print(f"  -> {catches_path} ({len(catches)} catches)")


if __name__ == "__main__":
    main()
