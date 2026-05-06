"""
Evaluation metrics and loop for captain agents.

Primary metric: captain_score = ModelGuidedCaptain_mean_catches / RandomCaptain_mean_catches
Secondary metric: oracle_fraction = ModelGuidedCaptain_mean_catches / OracleCaptain_mean_catches

Both include bootstrapped 95% confidence intervals.
"""
import math
from dataclasses import dataclass, asdict
import numpy as np

from eval.environment import SteerableSimulator
from eval.captain import (
    RandomCaptain,
    StraightCaptain,
    ModelGuidedCaptain,
    OracleCaptain,
)
from ml.inference import InferenceEngine


@dataclass
class EpisodeResult:
    """Result of a single episode (captain + seed)."""

    episode_id: str
    captain_name: str
    seed: int
    total_catches: int
    catches_by_species: dict
    distance_traveled: float
    trajectory: list  # list of (east_m, north_m) tuples


@dataclass
class EvaluationReport:
    """Summary of evaluation over many episodes."""

    model_path: str
    n_episodes: int
    duration_s: float
    episodes: list

    # Primary metrics
    captain_score: float
    captain_score_ci: tuple  # (low, high)
    oracle_fraction: float
    oracle_fraction_ci: tuple  # (low, high)

    # Per-captain mean catches
    mean_catches: dict  # captain_name -> float
    mean_catches_by_species: dict  # captain_name -> {species -> float}


def run_episode(
    captain,
    inference_engine,
    sim: SteerableSimulator,
    duration_s: float,
) -> EpisodeResult:
    """
    Run a single episode: captain guides boat through simulation.

    Args:
        captain: CaptainAgent instance
        inference_engine: InferenceEngine or None
        sim: SteerableSimulator instance
        duration_s: simulation duration in seconds

    Returns:
        EpisodeResult with catches and trajectory
    """
    obs = sim.reset()
    trajectory = [(obs.east_m, obs.north_m)]
    all_catches = []
    catches_by_species = {}

    for step in range(int(duration_s)):
        # Get model predictions (if engine available)
        if inference_engine is not None:
            predictions = inference_engine.push(obs, obs.forward_scan)
        else:
            predictions = None

        # Captain decides
        state = {
            "east_m": obs.east_m,
            "north_m": obs.north_m,
            "heading_deg": obs.heading_deg,
            "speed_kts": obs.speed_kts,
            "depth_m": obs.depth_m,
            "t": obs.ts,
        }
        heading_delta, speed_kts = captain.decide(state, predictions)

        # Simulate step
        obs, step_catches = sim.step(heading_delta, speed_kts, dt=1.0)
        trajectory.append((obs.east_m, obs.north_m))

        # Accumulate catches
        for catch in step_catches:
            all_catches.append(catch)
            species = catch["species"]
            catches_by_species[species] = catches_by_species.get(species, 0) + 1

    # Compute distance traveled
    distance = 0.0
    for i in range(len(trajectory) - 1):
        e1, n1 = trajectory[i]
        e2, n2 = trajectory[i + 1]
        distance += math.sqrt((e2 - e1) ** 2 + (n2 - n1) ** 2)

    return EpisodeResult(
        episode_id=f"{sim._seed}_{captain.__class__.__name__}",
        captain_name=captain.__class__.__name__,
        seed=sim._seed,
        total_catches=len(all_catches),
        catches_by_species=catches_by_species,
        distance_traveled=distance,
        trajectory=trajectory,
    )


def evaluate_model(
    model_path: str,
    n_episodes: int = 10,
    duration_s: float = 300.0,
    seeds: list = None,
) -> EvaluationReport:
    """
    Evaluate model quality via captain agents.

    Runs 4 captains × n_episodes episodes (4 random seeds per captain if not provided).
    Computes captain_score and oracle_fraction with bootstrapped 95% CIs.

    Args:
        model_path: path to trained .pt checkpoint
        n_episodes: number of episodes per captain
        duration_s: duration of each episode in seconds
        seeds: list of seeds to use; if None, use [0..n_episodes-1]

    Returns:
        EvaluationReport with metrics and CIs
    """
    if seeds is None:
        seeds = list(range(n_episodes))
    else:
        seeds = seeds[: n_episodes]  # trim or use provided
        while len(seeds) < n_episodes:
            seeds.append(len(seeds))

    # Load model (inference engine may fail gracefully if model doesn't exist)
    try:
        model_engine = InferenceEngine(model_path, device="cpu")
    except Exception as e:
        print(f"Warning: could not load model: {e}")
        model_engine = None

    all_episodes = []
    captain_catches = {
        "RandomCaptain": [],
        "StraightCaptain": [],
        "ModelGuidedCaptain": [],
        "OracleCaptain": [],
    }

    for seed in seeds:
        print(f"  Seed {seed}...", end=" ", flush=True)

        # RandomCaptain
        sim = SteerableSimulator(seed=seed)
        captain = RandomCaptain(seed=seed)
        engine = None  # RandomCaptain doesn't need model
        ep = run_episode(captain, engine, sim, duration_s)
        all_episodes.append(ep)
        captain_catches["RandomCaptain"].append(ep.total_catches)
        print(f"R:{ep.total_catches}", end=" ", flush=True)

        # StraightCaptain
        sim = SteerableSimulator(seed=seed)
        captain = StraightCaptain()
        engine = None  # StraightCaptain doesn't need model
        ep = run_episode(captain, engine, sim, duration_s)
        all_episodes.append(ep)
        captain_catches["StraightCaptain"].append(ep.total_catches)
        print(f"S:{ep.total_catches}", end=" ", flush=True)

        # ModelGuidedCaptain
        sim = SteerableSimulator(seed=seed)
        captain = ModelGuidedCaptain(seed=seed)
        engine = InferenceEngine(model_path, device="cpu") if model_engine else None
        ep = run_episode(captain, engine, sim, duration_s)
        all_episodes.append(ep)
        captain_catches["ModelGuidedCaptain"].append(ep.total_catches)
        print(f"M:{ep.total_catches}", end=" ", flush=True)

        # OracleCaptain
        sim = SteerableSimulator(seed=seed)
        captain = OracleCaptain(schools=sim.fish_schools, seed=seed)
        engine = None  # OracleCaptain doesn't use model
        ep = run_episode(captain, engine, sim, duration_s)
        all_episodes.append(ep)
        captain_catches["OracleCaptain"].append(ep.total_catches)
        print(f"O:{ep.total_catches}")

    # Compute metrics and bootstrap CIs
    random_catches = np.array(captain_catches["RandomCaptain"])
    model_catches = np.array(captain_catches["ModelGuidedCaptain"])
    oracle_catches = np.array(captain_catches["OracleCaptain"])

    captain_score, captain_score_ci = _bootstrap_ratio(
        model_catches, random_catches
    )
    oracle_fraction, oracle_fraction_ci = _bootstrap_ratio(
        model_catches, oracle_catches
    )

    # Per-captain mean catches
    mean_catches = {
        name: float(np.mean(counts))
        for name, counts in captain_catches.items()
    }

    # Per-captain, per-species mean catches
    mean_catches_by_species = {}
    for captain_name in captain_catches.keys():
        species_counts = {}
        eps = [e for e in all_episodes if e.captain_name == captain_name]
        if eps:
            all_species = set()
            for e in eps:
                all_species.update(e.catches_by_species.keys())
            for species in all_species:
                counts = [e.catches_by_species.get(species, 0) for e in eps]
                species_counts[species] = float(np.mean(counts))
        mean_catches_by_species[captain_name] = species_counts

    return EvaluationReport(
        model_path=model_path,
        n_episodes=n_episodes,
        duration_s=duration_s,
        episodes=all_episodes,
        captain_score=captain_score,
        captain_score_ci=captain_score_ci,
        oracle_fraction=oracle_fraction,
        oracle_fraction_ci=oracle_fraction_ci,
        mean_catches=mean_catches,
        mean_catches_by_species=mean_catches_by_species,
    )


def _bootstrap_ratio(numerator: np.ndarray, denominator: np.ndarray, n_boot: int = 1000) -> tuple:
    """
    Bootstrap 95% CI for ratio of means: numerator / denominator.

    Handles edge case: if denominator_mean < 1e-6, return (inf, inf).

    Args:
        numerator: array of catch counts (one per episode)
        denominator: array of catch counts (one per episode)
        n_boot: number of bootstrap resamples

    Returns:
        (ratio_mean, (ci_low, ci_high))
    """
    denom_mean = float(np.mean(denominator))
    if denom_mean < 1e-6:
        return float("inf"), (float("inf"), float("inf"))

    # Resample with replacement and compute ratio each time
    ratios = []
    rng = np.random.RandomState(42)
    for _ in range(n_boot):
        num_sample = rng.choice(numerator, size=len(numerator), replace=True)
        denom_sample = rng.choice(denominator, size=len(denominator), replace=True)
        ratio = float(np.mean(num_sample)) / float(np.mean(denom_sample))
        ratios.append(ratio)

    ratios = np.array(ratios)
    ci_low = float(np.percentile(ratios, 2.5))
    ci_high = float(np.percentile(ratios, 97.5))
    ratio_mean = np.mean(numerator) / np.mean(denominator)

    return float(ratio_mean), (ci_low, ci_high)
