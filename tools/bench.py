"""
tools/bench.py

Profile the synthetic → fusion → world_state pipeline and emit a
deterministic hash of the resulting WorldState so we can verify that
optimisations don't alter behaviour.

Usage:
    PYTHONPATH=src python tools/bench.py            # default 30s session
    PYTHONPATH=src python tools/bench.py --duration 120
"""
from __future__ import annotations
import argparse
import hashlib
import io
import os
import sys
import time
import contextlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _hash_worldstate(ws) -> str:
    """
    Deterministic content hash of every Observation in the WorldState.
    Rounded to avoid float-noise jitter from minor numeric reordering.
    """
    h = hashlib.sha256()
    if ws._data is None:
        return "empty"
    arr = ws._data
    # Sort by ts, east, north, depth to remove ordering dependence
    import numpy as np
    order = np.lexsort((arr[:, 3], arr[:, 2], arr[:, 1], arr[:, 0]))
    sorted_arr = arr[order]
    for row in sorted_arr:
        h.update(b"|".join(f"{v:.4f}".encode() for v in row))
    return h.hexdigest()[:16]


def _silenced_stdout():
    """Suppress fusion's per-tick prints during the benchmark itself."""
    return contextlib.redirect_stdout(io.StringIO())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--seed",     type=int,   default=42)
    ap.add_argument("--repeat",   type=int,   default=3)
    args = ap.parse_args()

    from synthetic.generator import generate
    from processing.fusion import Fusion
    from processing.world_state import WorldState

    timings = {"generate": [], "fusion+add": [], "to_pointcloud": [], "total": []}
    final_hash = None
    final_n = 0

    for i in range(args.repeat):
        t0 = time.perf_counter()

        with _silenced_stdout():
            session = generate(duration_s=args.duration, seed=args.seed)

        t1 = time.perf_counter()

        ws = WorldState()
        f  = Fusion()
        with _silenced_stdout():
            for tick in session.ticks:
                for obs in f.process(tick):
                    ws.add(obs)

        t2 = time.perf_counter()

        pc = ws.to_pointcloud(current_ts=args.duration)
        t3 = time.perf_counter()

        h = _hash_worldstate(ws)
        n = len(ws)

        timings["generate"].append(t1 - t0)
        timings["fusion+add"].append(t2 - t1)
        timings["to_pointcloud"].append(t3 - t2)
        timings["total"].append(t3 - t0)

        if final_hash is None:
            final_hash = h
            final_n = n
        elif final_hash != h:
            print(f"!! BEHAVIOUR DRIFT  run {i}: hash {h} != {final_hash}")

    def _stats(name):
        vals = timings[name]
        return f"{name:14s}  best {min(vals)*1000:7.1f} ms   "\
               f"median {sorted(vals)[len(vals)//2]*1000:7.1f} ms   "\
               f"mean {sum(vals)/len(vals)*1000:7.1f} ms"

    print(f"duration={args.duration:.0f}s  seed={args.seed}  repeats={args.repeat}")
    print(f"observations: {final_n}")
    print(f"hash:         {final_hash}")
    print()
    for name in ("generate", "fusion+add", "to_pointcloud", "total"):
        print(_stats(name))


if __name__ == "__main__":
    main()
