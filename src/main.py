"""
src/main.py

Entry point.  Three modes:

  python src/main.py                          # synthetic (generates data at startup)
  python src/main.py --replay session.ticks   # replay a recorded .ticks file
  python src/main.py --live /dev/ttyUSB0      # stream from serial port

In synthetic and replay modes all ticks are pre-processed at startup so
seeking is instant — the WorldState is complete before the server starts.
"""
import argparse
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


def parse_args():
    p = argparse.ArgumentParser(description="Artifishial Intelligence v3")
    p.add_argument("--replay",    default=None, metavar="FILE",
                   help=".ticks SQLite recording to replay")
    p.add_argument("--live",      default=None, metavar="PORT",
                   help="Serial port for live NMEA (e.g. /dev/ttyUSB0 or COM3)")
    p.add_argument("--gt",        default=None, metavar="FILE",
                   help="ground_truth.json (optional, enables fish overlay)")
    p.add_argument("--host",      default="0.0.0.0")
    p.add_argument("--port",      default=8000, type=int)
    p.add_argument("--duration",  default=120.0, type=float,
                   help="Duration for synthetic session (seconds)")
    return p.parse_args()


def _prebake(tick_iter, fusion) -> "WorldState":
    """
    Fast-path: process all ticks through fusion without any sleep.
    Returns a fully-populated WorldState.
    For a 2-minute session (~600 ticks) this runs in <10ms.
    """
    from processing.world_state import WorldState
    ws = WorldState()
    for tick in tick_iter:
        for obs in fusion.process(tick):   # list: [bottom] + [fish...]
            ws.add(obs)
    return ws


async def _live_loop(serial_port: str, fusion, world_state):
    """Background task: stream live NMEA ticks and add observations."""
    from ticks.live import LiveNMEAReader
    from ticks.models import Tick
    reader = LiveNMEAReader(serial_port)
    async for gps_tick in reader.stream():
        tick = Tick(ts=gps_tick.ts, gps=gps_tick)
        for obs in fusion.process(tick):
            world_state.add(obs)


async def main():
    args = parse_args()

    from processing.fusion import Fusion
    from processing.world_state import WorldState
    from server.api import app, set_session
    from server.replay_controller import ReplayController
    import uvicorn

    fusion      = Fusion()
    world_state = WorldState()
    replay_ctrl = None
    ground_truth = None

    # ── Load ground truth if supplied ────────────────────────────────────────
    if args.gt:
        from ground_truth.manifest import GroundTruth
        ground_truth = GroundTruth.from_file(args.gt)

    # ── Mode: replay .ticks file ──────────────────────────────────────────────
    if args.replay:
        from ticks.replayer import Replayer
        with Replayer(args.replay) as rep:
            print(f"[main] Pre-baking replay: {args.replay} "
                  f"({rep.duration_s:.1f}s)")
            world_state = _prebake(rep.iter_all(), fusion)
            duration    = rep.duration_s
        replay_ctrl = ReplayController(duration_s=duration)
        print(f"[main] {len(world_state)} observations loaded")

    # ── Mode: synthetic session ───────────────────────────────────────────────
    elif not args.live:
        from synthetic.generator import generate
        from ground_truth.manifest import GroundTruth
        print(f"[main] Generating {args.duration:.0f}s synthetic session…")
        session     = generate(duration_s=args.duration)
        world_state = _prebake(session.ticks, fusion)
        replay_ctrl = ReplayController(duration_s=session.duration_s)
        if ground_truth is None:
            ground_truth = GroundTruth.from_dict(session.to_ground_truth())
        print(f"[main] {len(world_state)} observations, "
              f"{len(session.fish_schools)} fish schools")

    # ── Mode: live serial ─────────────────────────────────────────────────────
    else:
        print(f"[main] Live mode: {args.live}")
        # replay_ctrl stays None — no seeking in live mode

    set_session(world_state, replay_ctrl, ground_truth,
                duration_s=replay_ctrl.duration_s if replay_ctrl else None)

    print(f"\n{'='*52}")
    print(f"  Artifishial Intelligence v3")
    print(f"  AR viewer : http://localhost:{args.port}")
    print(f"  WebSocket : ws://localhost:{args.port}/ws/state")
    print(f"  Mode      : "
          f"{'LIVE' if args.live else 'REPLAY' if args.replay else 'SYNTHETIC'}")
    print(f"{'='*52}\n")

    config = uvicorn.Config(app, host=args.host, port=args.port,
                            log_level="warning", loop="asyncio")
    server = uvicorn.Server(config)

    if args.live:
        await asyncio.gather(
            _live_loop(args.live, fusion, world_state),
            server.serve(),
        )
    else:
        await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
