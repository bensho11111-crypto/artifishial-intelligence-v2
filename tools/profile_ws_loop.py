"""
tools/profile_ws_loop.py

Profile the per-iteration work the WebSocket /ws/state loop performs,
sweeping replay position from 0 -> duration to measure how each stage
scales with elapsed time.

Stages timed (mirrors api.py:_build_update + dumps + advance):
  state_at, to_pointcloud, extract_boat, echo_at, fwd_scan_at,
  build_mesh, json.dumps

Usage:
    PYTHONPATH=src python tools/profile_ws_loop.py --duration 120 --steps 12
"""
from __future__ import annotations
import argparse
import contextlib
import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _silenced():
    return contextlib.redirect_stdout(io.StringIO())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--steps",    type=int,   default=12)
    ap.add_argument("--seed",     type=int,   default=42)
    args = ap.parse_args()

    from synthetic.generator import generate
    from processing.fusion import Fusion
    from processing.world_state import WorldState
    from rendering.pointcloud import build_pointcloud_payload, extract_boat
    from rendering.mesh import build_mesh
    import base64
    try:
        import orjson
        def _dumps(o):
            return orjson.dumps(o, option=orjson.OPT_SERIALIZE_NUMPY)
        json_label = "orjson"
    except ImportError:
        def _dumps(o):
            return json.dumps(o).encode("utf-8")
        json_label = "stdlib"

    print(f"Pre-baking {args.duration:.0f}s synthetic session...")
    with _silenced():
        session = generate(duration_s=args.duration, seed=args.seed)
    ws = WorldState()
    f  = Fusion()
    with _silenced():
        for tick in session.ticks:
            for obs in f.process(tick):
                ws.add(obs)
    print(f"  {len(ws)} observations\n")

    stages = ["state_at", "to_pointcloud", "extract_boat", "echo_at",
              "fwd_scan_at", "build_mesh", json_label, "TOTAL"]
    header = f"{'ts':>6}  {'npts':>5}  " + "  ".join(f"{s:>13}" for s in stages)
    print(header)
    print("-" * len(header))

    last_mesh_payload = None
    for i in range(args.steps + 1):
        ts = (i / args.steps) * args.duration
        timings = {}

        t0 = time.perf_counter()
        view = ws.state_at(ts)
        t1 = time.perf_counter()
        timings["state_at"] = t1 - t0

        pc = build_pointcloud_payload(ws, ts)
        t2 = time.perf_counter()
        timings["to_pointcloud"] = t2 - t1

        boat = extract_boat(ws.state_at(ts))
        t3 = time.perf_counter()
        timings["extract_boat"] = t3 - t2

        echo = ws.echo_at(ts)
        t4 = time.perf_counter()
        timings["echo_at"] = t4 - t3

        fwd = ws.forward_scan_at(ts)
        t5 = time.perf_counter()
        timings["fwd_scan_at"] = t5 - t4

        mesh = build_mesh(ws, ts)
        t6 = time.perf_counter()
        timings["build_mesh"] = t6 - t5

        payload = {
            "type": "map_update",
            "ts": ts,
            "duration_s": args.duration,
            "pointcloud": pc,
            "boat": boat,
            "replay": {"position_s": ts, "duration_s": args.duration,
                       "paused": False, "fraction": ts / args.duration},
        }
        if echo:
            payload["echo"] = base64.b64encode(echo).decode("ascii")
        if fwd:
            payload["forward_scan"] = base64.b64encode(fwd).decode("ascii")
        if mesh is not None:
            payload["mesh"] = mesh
        t7 = time.perf_counter()
        s = _dumps(payload)
        t8 = time.perf_counter()
        timings[json_label] = t8 - t7
        timings["TOTAL"] = t8 - t0

        n = len(pc.get("x", []))
        row = f"{ts:6.1f}  {n:5d}  " + "  ".join(
            f"{timings[s]*1000:11.2f} ms" for s in stages)
        print(row)

    # JSON byte size at end (raw, every-frame, all-fields)
    print(f"\nRaw final payload size: {len(s)/1024:.1f} KiB "
          f"(mesh present: {mesh is not None})")

    # ── Second pass: simulate the optimized loop ─────────────────────────────
    # - single state_at(ts) per frame, view reused
    # - slim pointcloud (drop ts/heading/speed_kts)
    # - mesh built only when floor-count grew, throttled to every 2s
    # - extract_boat from view directly
    print("\n--- Optimized loop simulation ---")
    print(header)
    print("-" * len(header))

    last_floor_n = -1
    last_mesh_wall = -1e9
    cached_mesh = None
    sim_wall = 0.0
    for i in range(args.steps + 1):
        ts = (i / args.steps) * args.duration
        timings = {s: 0.0 for s in stages}

        t0 = time.perf_counter()
        view = ws.state_at(ts)
        t1 = time.perf_counter()
        timings["state_at"] = t1 - t0

        pc = view.to_pointcloud(floor_only=False, current_ts=ts)
        for k in ("ts", "heading", "speed_kts"):
            pc.pop(k, None)
        t2 = time.perf_counter()
        timings["to_pointcloud"] = t2 - t1

        boat = extract_boat(view)
        t3 = time.perf_counter()
        timings["extract_boat"] = t3 - t2

        echo = ws.echo_at(ts)
        t4 = time.perf_counter()
        timings["echo_at"] = t4 - t3

        fwd = ws.forward_scan_at(ts)
        t5 = time.perf_counter()
        timings["fwd_scan_at"] = t5 - t4

        # Mesh: throttle to 2s + only rebuild if floor_n changed
        floor_n = int((view._data[:, 7] > 0.5).sum()) if view._data is not None else 0
        send_mesh = None
        if sim_wall - last_mesh_wall >= 2.0 and floor_n != last_floor_n:
            cached_mesh = view.to_mesh(min_points=4)
            last_floor_n = floor_n
            last_mesh_wall = sim_wall
            send_mesh = cached_mesh
        t6 = time.perf_counter()
        timings["build_mesh"] = t6 - t5

        payload = {
            "type": "map_update", "ts": ts,
            "duration_s": args.duration,
            "pointcloud": pc, "boat": boat,
            "replay": {"position_s": ts, "duration_s": args.duration,
                       "paused": False, "fraction": ts / args.duration},
        }
        if echo:
            payload["echo"] = base64.b64encode(echo).decode("ascii")
        if fwd:
            payload["forward_scan"] = base64.b64encode(fwd).decode("ascii")
        if send_mesh is not None:
            payload["mesh"] = send_mesh
        t7 = time.perf_counter()
        s = _dumps(payload)
        t8 = time.perf_counter()
        timings[json_label] = t8 - t7
        timings["TOTAL"] = t8 - t0

        sim_wall += timings["TOTAL"] + 0.1   # +0.1 for the asyncio.sleep

        n = len(pc.get("x", []))
        row = f"{ts:6.1f}  {n:5d}  " + "  ".join(
            f"{timings[s]*1000:11.2f} ms" for s in stages)
        print(row)

    print(f"\nOptimized final payload size: {len(s)/1024:.1f} KiB "
          f"(mesh sent this frame: {'yes' if send_mesh is not None else 'no'})")


if __name__ == "__main__":
    main()
