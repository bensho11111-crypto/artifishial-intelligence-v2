"""
Compare evaluation results from all 4 experiments.

Usage:
    python compare_experiments.py
"""
import json
from pathlib import Path
import sys

def load_eval_results(file_path):
    """Load evaluation results from JSON file."""
    try:
        with open(file_path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {file_path} not found")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing {file_path}: {e}")
        return None


def format_metric(value):
    """Format metric with proper handling of inf/-inf/nan."""
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        if value == float('inf'):
            return "∞"
        elif value == float('-inf'):
            return "-∞"
        elif value != value:  # NaN check
            return "NaN"
        elif isinstance(value, float) and value < 1:
            return f"{value:.4f}"
        else:
            return f"{value:.2f}"
    return str(value)


def extract_metrics(results):
    """Extract key metrics from evaluation results."""
    if not results:
        return None

    return {
        'captain_score': results.get('captain_score'),
        'oracle_fraction': results.get('oracle_fraction'),
        'captain_score_ci': results.get('captain_score_ci'),
        'oracle_fraction_ci': results.get('oracle_fraction_ci'),
        'episodes': len(results.get('episodes', [])),
        'mean_catches': {
            'RandomCaptain': results.get('mean_catches', {}).get('RandomCaptain'),
            'ModelGuidedCaptain': results.get('mean_catches', {}).get('ModelGuidedCaptain'),
            'OracleCaptain': results.get('mean_catches', {}).get('OracleCaptain'),
        }
    }


def main():
    base_path = Path(".")

    experiments = [
        ("larger-dataset (baseline)", "eval_larger_dataset.json"),
        ("approach-detection (proximity)", "eval_approach_detection.json"),
        ("loss-tuning (aggressive)", "eval_loss_tuning.json"),
        ("class-resampling (weighted)", "eval_class_resampling.json"),
    ]

    print("\n" + "="*80)
    print("EXPERIMENT COMPARISON: Label Sparsity Solutions")
    print("="*80 + "\n")

    results = {}
    for name, file_path in experiments:
        print(f"Loading {name}...", end=" ")
        data = load_eval_results(file_path)
        if data:
            print("[OK]")
            results[name] = extract_metrics(data)
        else:
            print("[PENDING]")
            results[name] = None

    print("\n" + "-"*80)
    print("RESULTS SUMMARY")
    print("-"*80 + "\n")

    # Table header
    print(f"{'Experiment':<35} {'Oracle Frac':<15} {'Captain Score':<15} {'Episodes':<10}")
    print("-"*80)

    # Table rows
    for name, metrics in results.items():
        if metrics:
            oracle = format_metric(metrics['oracle_fraction'])
            captain = format_metric(metrics['captain_score'])
            episodes = metrics['episodes']
            print(f"{name:<35} {oracle:<15} {captain:<15} {episodes:<10}")
        else:
            print(f"{name:<35} {'(not ready)':<15} {'':<15} {'':<10}")

    print("\n" + "-"*80)
    print("DETAILED METRICS")
    print("-"*80 + "\n")

    for name, metrics in results.items():
        if metrics:
            print(f"\n{name}")
            print("  Oracle Fraction (% of Oracle):")
            print(f"    Value: {format_metric(metrics['oracle_fraction'])}")
            if metrics['oracle_fraction_ci']:
                ci = metrics['oracle_fraction_ci']
                print(f"    95% CI: [{format_metric(ci[0])}, {format_metric(ci[1])}]")

            print("  Captain Score (Model / Random):")
            print(f"    Value: {format_metric(metrics['captain_score'])}")
            if metrics['captain_score_ci']:
                ci = metrics['captain_score_ci']
                print(f"    95% CI: [{format_metric(ci[0])}, {format_metric(ci[1])}]")

            print("  Mean Catches:")
            for captain_type in ['RandomCaptain', 'ModelGuidedCaptain', 'OracleCaptain']:
                catches = metrics['mean_catches'].get(captain_type)
                print(f"    {captain_type:<20}: {format_metric(catches)}")

    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80 + "\n")
    print("oracle_fraction: What % of the oracle's catch rate does our model achieve?")
    print("                 Higher is better (max 100%). Target: >5%")
    print("")
    print("captain_score:   How many times better is ModelGuided vs Random?")
    print("                 Higher is better. Baseline ~1.6x, target >5x")
    print("")
    print("The best approach is the one with highest oracle_fraction and captain_score.")


if __name__ == "__main__":
    main()
