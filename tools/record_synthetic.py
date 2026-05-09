"""
tools/record_synthetic.py

Generate a synthetic fishing session and record every Tick — including
the raw echo + forward-scan bytes — into a .ticks SQLite file.

Usage:
    python tools/record_synthetic.py out.ticks --duration 120
"""
import argparse
import os
import sys

# Make src/ importable when invoked from the repo root.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))


def main() -> None:
    p = argparse.ArgumentParser(description="Record a synthetic .ticks session")
    p.add_argument("out", help="Output .ticks path")
    p.add_argument("--duration", type=float, default=120.0,
                   help="Session duration in seconds (default: 120)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    from synthetic.generator import generate
    from ticks.recorder import Recorder

    print(f"[record] Generating {args.duration:.0f}s synthetic session…")
    session = generate(duration_s=args.duration, seed=args.seed)
    print(f"[record] {len(session.ticks)} ticks, "
          f"{len(session.fish_schools)} fish schools")

    if os.path.exists(args.out):
        os.remove(args.out)

    with Recorder(args.out) as rec:
        rec.set_metadata("duration_s", str(session.duration_s))
        rec.set_metadata("source", "synthetic")
        rec.set_metadata("seed", str(args.seed))
        for tick in session.ticks:
            rec.record(tick)

    size_mb = os.path.getsize(args.out) / 1024 / 1024
    print(f"[record] Wrote {args.out} ({size_mb:.1f} MiB)")


if __name__ == "__main__":
    main()
