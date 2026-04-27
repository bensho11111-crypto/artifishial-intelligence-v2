"""
tools/debug_fish_overlap.py

Diagnose why blue points appear "on" fish in the AR view at a given
replay timestamp. The forward-scan generates many low-confidence floor
returns; when those returns share XY with a fish school they appear
visually overlapping with the orange fish points.

Usage:
    PYTHONPATH=src python tools/debug_fish_overlap.py --ts 35
"""
from __future__ import annotations
import argparse
import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _silenced():
    return contextlib.redirect_stdout(io.StringIO())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts",        type=float, default=35.0)
    ap.add_argument("--duration",  type=float, default=120.0)
    ap.add_argument("--seed",      type=int,   default=42)
    ap.add_argument("--radius_m",  type=float, default=1.0,
                    help="horizontal radius for 'overlap' check")
    args = ap.parse_args()

    import numpy as np
    from synthetic.generator import generate
    from processing.fusion import Fusion
    from processing.world_state import WorldState

    print(f"Pre-baking {args.duration:.0f}s synthetic session (seed={args.seed})...")
    with _silenced():
        s = generate(duration_s=args.duration, seed=args.seed)
    ws = WorldState(); f = Fusion()
    with _silenced():
        for t in s.ticks:
            for o in f.process(t): ws.add(o)

    ts = args.ts
    view = ws.state_at(ts)
    d = view._data
    is_fish_mask = d[:, 7] < 0.5
    fish  = d[is_fish_mask]
    floor = d[~is_fish_mask]

    print(f"\n=== ts = {ts:.1f}s ===")
    print(f"observations: total={len(d)}  floor={len(floor)}  fish={len(fish)}")
    if len(fish) == 0 or len(floor) == 0:
        print("(no fish or no floor — nothing to check)")
        return

    # Confidence breakdown of floor returns. <= 0.45 → frontend applies
    # the muted-cyan "forward-scan" tint, which is the blue the user sees.
    fwd_floor_n = int((floor[:, 4] <= 0.45).sum())
    reg_floor_n = len(floor) - fwd_floor_n
    print(f"  regular-ping floor (conf > 0.45):  {reg_floor_n:6d}")
    print(f"  forward-scan floor (conf <= 0.45): {fwd_floor_n:6d}  "
          f"<-- these render as muted cyan/blue")

    # Depth check — confirm floors are NOT at fish depth (i.e. no
    # misclassification of fish as floor).
    print(f"\nDepth ranges:")
    print(f"  fish:  {fish[:,3].min():5.1f} .. {fish[:,3].max():5.1f} m  (mean {fish[:,3].mean():.1f})")
    print(f"  floor: {floor[:,3].min():5.1f} .. {floor[:,3].max():5.1f} m  (mean {floor[:,3].mean():.1f})")
    overlap = floor[(floor[:,3] >= fish[:,3].min()) & (floor[:,3] <= fish[:,3].max())]
    print(f"  floor obs at any fish depth: {len(overlap)} "
          f"({len(overlap)/len(floor)*100:.1f}% of floor)")

    # Spatial overlap — for each fish, distance to nearest floor obs
    # horizontally and the vertical separation of that nearest floor.
    from scipy.spatial import cKDTree
    tree = cKDTree(floor[:, 1:3])
    dists, nn = tree.query(fish[:, 1:3], k=1)
    vdists = np.abs(fish[:, 3] - floor[nn, 3])
    print(f"\nFish vs nearest floor:")
    print(f"  horizontal: median {np.median(dists):5.2f} m   p90 {np.percentile(dists,90):5.2f} m")
    print(f"  vertical:   median {np.median(vdists):5.2f} m   p10 {np.percentile(vdists,10):5.2f} m   "
          f"min {vdists.min():.2f} m")
    close_h = (dists < args.radius_m).sum()
    print(f"  {close_h}/{len(fish)} fish ({close_h/len(fish)*100:.0f}%) have a floor obs "
          f"within {args.radius_m:.1f} m horizontally")

    # Vertical-overlap test — are floors at the SAME 3D location as fish,
    # or are they cleanly separated below?
    truly_overlapping = ((dists < args.radius_m) & (vdists < 1.0)).sum()
    print(f"  {truly_overlapping}/{len(fish)} fish have a floor obs within 1m "
          f"in BOTH XY and Z (would be a misclassification bug)")

    # Verdict
    print(f"\nVerdict:")
    if truly_overlapping > len(fish) * 0.05:
        print("  ⚠ Floor obs are co-located with fish in 3D — likely a misclassification")
        print("    bug in fusion.py (fish echoes being marked is_floor=True).")
    else:
        print("  ✓ Floor obs sit several metres BELOW the fish (forward-scan looking past")
        print("    each fish to the seafloor). Fish are correctly tagged. The visual")
        print(f"    'blue on fish' is camera-projection overlap: {fwd_floor_n} forward-scan")
        print("    floor returns cluster around the boat's forward cone every frame.")


if __name__ == "__main__":
    main()
