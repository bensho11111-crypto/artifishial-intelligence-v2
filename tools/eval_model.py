"""
tools/eval_model.py

CLI for evaluating a trained catch-prediction model via captain agents.

Usage:
    python tools/eval_model.py --model checkpoints/best.pt [--episodes 10] [--duration 300] [--seeds 0,1,2] [--out report.json]
"""
import argparse
import json
import sys
import os
from pathlib import Path
from dataclasses import asdict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eval.metrics import evaluate_model


def main():
    p = argparse.ArgumentParser(
        description="Evaluate catch-prediction model via captain agents"
    )
    p.add_argument(
        "--model",
        default=None,
        metavar="FILE",
        help="Path to trained .pt checkpoint (optional; if not provided, baseline captains only)",
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes per captain (default: 10)",
    )
    p.add_argument(
        "--duration",
        type=float,
        default=300.0,
        help="Duration of each episode in seconds (default: 300)",
    )
    p.add_argument(
        "--seeds",
        metavar="SEEDS",
        default=None,
        help="Comma-separated seeds (e.g. '0,1,2'); if not provided, use 0..episodes-1",
    )
    p.add_argument(
        "--out",
        metavar="FILE",
        default=None,
        help="Output JSON report (if not provided, print to stdout)",
    )
    args = p.parse_args()

    # Parse seeds
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
    else:
        seeds = None

    # Check model exists (or use None for baseline testing)
    if args.model and not os.path.exists(args.model):
        print(f"Warning: model file not found: {args.model}", file=sys.stderr)
        print(f"Proceeding with None model (baseline captains only)", file=sys.stderr)

    print(f"Evaluating model: {args.model}")
    print(f"Episodes: {args.episodes}, Duration: {args.duration}s")
    if seeds:
        print(f"Seeds: {seeds}")
    print()

    # Run evaluation
    print("Running evaluation...")
    report = evaluate_model(
        model_path=args.model,
        n_episodes=args.episodes,
        duration_s=args.duration,
        seeds=seeds,
    )
    print()

    # Format output
    output = _format_report(report)

    if args.out:
        # Write JSON
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize with custom handling for special floats
        def serialize(obj):
            if isinstance(obj, float):
                if obj == float("inf"):
                    return "inf"
                elif obj == float("-inf"):
                    return "-inf"
                return obj
            raise TypeError(f"cannot serialize {type(obj)}")

        with open(out_path, "w") as f:
            json.dump(asdict(report), f, indent=2, default=serialize)
        print(f"Report written to: {out_path}")
    else:
        print(output)


def _format_report(report) -> str:
    """Format report for console output."""
    lines = []

    # Per-episode table
    lines.append("=" * 100)
    lines.append("PER-EPISODE RESULTS")
    lines.append("=" * 100)
    lines.append(
        f"{'Seed':<5} {'Captain':<20} {'Catches':<10} {'Distance (m)':<15} {'Species':<40}"
    )
    lines.append("-" * 100)

    for ep in report.episodes:
        species_str = ", ".join(
            f"{s}:{c}"
            for s, c in sorted(ep.catches_by_species.items())
        )
        lines.append(
            f"{ep.seed:<5} {ep.captain_name:<20} {ep.total_catches:<10} {ep.distance_traveled:<15.1f} {species_str:<40}"
        )

    # Summary
    lines.append("")
    lines.append("=" * 100)
    lines.append("SUMMARY METRICS")
    lines.append("=" * 100)
    lines.append(f"Model: {report.model_path}")
    lines.append(f"Episodes: {report.n_episodes}, Duration: {report.duration_s}s")
    lines.append("")

    # Per-captain stats
    lines.append("Mean catches by captain:")
    for captain, mean_catches in sorted(report.mean_catches.items()):
        lines.append(f"  {captain:<25} {mean_catches:>8.2f}")
    lines.append("")

    # Primary metric
    lines.append(f"Captain Score (ModelGuided / Random):")
    if report.captain_score == float("inf"):
        lines.append(f"  {report.captain_score} [CI: inf]")
    else:
        ci_low, ci_high = report.captain_score_ci
        if ci_low == float("inf"):
            lines.append(f"  {report.captain_score:.3f} [CI: inf]")
        else:
            lines.append(f"  {report.captain_score:.3f} [CI: {ci_low:.3f}, {ci_high:.3f}]")

    # Secondary metric
    lines.append(f"Oracle Fraction (ModelGuided / Oracle):")
    if report.oracle_fraction == float("inf"):
        lines.append(f"  {report.oracle_fraction} [CI: inf]")
    else:
        ci_low, ci_high = report.oracle_fraction_ci
        if ci_low == float("inf"):
            lines.append(f"  {report.oracle_fraction:.3f} [CI: inf]")
        else:
            lines.append(
                f"  {report.oracle_fraction:.3f} [CI: {ci_low:.3f}, {ci_high:.3f}]"
            )

    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
