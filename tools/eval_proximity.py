#!/usr/bin/env python3
"""
Proximity Correlation Metric: Measure how well model predictions correlate with actual fish proximity.

Usage:
    python tools/eval_proximity.py --model checkpoints_simple_cpu/best.pt [--duration 300] [--seeds 0,1,2]
"""
import argparse
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eval.metrics import compute_proximity_correlation
from eval.environment import SteerableSimulator
from ml.inference import InferenceEngine


def main():
    p = argparse.ArgumentParser(
        description="Measure correlation between model predictions and proximity to fish schools"
    )
    p.add_argument(
        "--model",
        required=True,
        metavar="FILE",
        help="Path to trained .pt checkpoint",
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
        default="0,1,2",
        help="Comma-separated seeds (default: 0,1,2)",
    )
    p.add_argument(
        "--out",
        metavar="FILE",
        default=None,
        help="Output JSON file (if not provided, print to stdout)",
    )
    args = p.parse_args()

    # Parse seeds
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    print(f"Model: {args.model}")
    print(f"Duration: {args.duration}s per episode")
    print(f"Seeds: {seeds}")
    print()

    # Load model
    try:
        engine = InferenceEngine(args.model, device="cpu")
    except Exception as e:
        print(f"ERROR loading model: {e}", file=sys.stderr)
        sys.exit(1)

    # Run evaluation for each seed
    results_by_seed = {}
    all_correlations = []

    print("Computing proximity correlations...")
    for seed in seeds:
        print(f"  Seed {seed}...", end=" ", flush=True)
        sim = SteerableSimulator(seed=seed)
        engine_fresh = InferenceEngine(args.model, device="cpu")

        result = compute_proximity_correlation(sim, engine_fresh, duration_s=args.duration)
        results_by_seed[seed] = result
        all_correlations.append(result["correlation"])
        print(f"r={result['correlation']:.3f}, p={result['p_value']:.4f}")

    # Aggregate statistics
    correlations_arr = np.array(all_correlations)
    mean_corr = float(np.mean(correlations_arr))
    std_corr = float(np.std(correlations_arr))

    summary = {
        "model": args.model,
        "duration_s": args.duration,
        "seeds": seeds,
        "results_by_seed": results_by_seed,
        "mean_correlation": mean_corr,
        "std_correlation": std_corr,
        "interpretation": _interpret_correlation(mean_corr),
    }

    # Print summary
    print("\n" + "=" * 80)
    print("PROXIMITY CORRELATION SUMMARY")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Mean correlation (across {len(seeds)} seeds): {mean_corr:.4f} ± {std_corr:.4f}")
    print(f"Interpretation: {summary['interpretation']}")
    print()

    # Print per-seed details
    print("PER-SEED BREAKDOWN:")
    print("-" * 80)
    for seed in seeds:
        r = results_by_seed[seed]["correlation"]
        p = results_by_seed[seed]["p_value"]
        n = results_by_seed[seed]["n_observations"]
        mean_pred = results_by_seed[seed]["mean_prediction"]
        mean_dist = results_by_seed[seed]["mean_distance_to_nearest_school"]

        print(f"Seed {seed}:")
        print(f"  Correlation: {r:.4f} (p={p:.4f}, n={n})")
        print(f"  Mean prediction: {mean_pred:.4f}")
        print(f"  Mean distance to nearest school: {mean_dist:.1f}m")
        print()

    # Print distance bins
    if seeds:
        seed_0_result = results_by_seed[seeds[0]]
        print("MEAN PREDICTION BY DISTANCE (Seed 0):")
        print("-" * 80)
        for bin_label, mean_pred in seed_0_result["mean_pred_by_distance"].items():
            bar = "=" * int(mean_pred * 50)
            try:
                print(f"  {bin_label:15s}: {mean_pred:.4f}  {bar}")
            except UnicodeEncodeError:
                # Fallback if terminal doesn't support encoding
                clean_label = bin_label.encode('ascii', errors='replace').decode('ascii')
                print(f"  {clean_label:15s}: {mean_pred:.4f}  {bar}")
        print()

    # Save output
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Report saved to: {out_path}")


def _interpret_correlation(r: float) -> str:
    """Interpret correlation value."""
    if r < 0:
        return f"INVERTED: Model predicts HIGH when fish are FAR (r={r:.3f})"
    elif r < 0.2:
        return f"WEAK/NONE: Model predictions don't correlate with proximity (r={r:.3f})"
    elif r < 0.4:
        return f"WEAK: Model shows slight correlation with proximity (r={r:.3f})"
    elif r < 0.6:
        return f"MODERATE: Model predictions somewhat track proximity (r={r:.3f})"
    elif r < 0.8:
        return f"STRONG: Model predictions well-aligned with proximity (r={r:.3f})"
    else:
        return f"EXCELLENT: Model predictions highly correlated with proximity (r={r:.3f})"


if __name__ == "__main__":
    import numpy as np
    main()
