"""
Evaluation metrics and loop for captain agents.

Primary metric: captain_score = ModelGuidedCaptain_mean_catches / RandomCaptain_mean_catches
Secondary metric: oracle_fraction = ModelGuidedCaptain_mean_catches / OracleCaptain_mean_catches
NEW metric: proximity_correlation = correlation between model predictions and proximity to fish

All include bootstrapped 95% confidence intervals.
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


def _pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Pearson correlation coefficient between x and y.

    r = cov(x,y) / (std(x) * std(y))
    """
    if len(x) < 2 or len(y) < 2:
        return 0.0

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    x_centered = x - x_mean
    y_centered = y - y_mean

    cov = np.mean(x_centered * y_centered)
    std_x = np.std(x_centered)
    std_y = np.std(y_centered)

    if std_x == 0 or std_y == 0:
        return 0.0

    return float(cov / (std_x * std_y))


def compute_proximity_correlation(
    sim: SteerableSimulator,
    engine: InferenceEngine,
    duration_s: float = 300.0,
) -> dict:
    """
    Measure correlation between model predictions and proximity to fish schools.

    For each timestep after ring buffer warmup, records:
    - Model's predicted catch probability (sum of all species)
    - Euclidean distance to nearest fish school

    Returns:
        {
            "correlation": float,  # Pearson r (-1 to 1)
            "p_value": float,      # Statistical significance
            "mean_pred_by_distance": dict,  # predictions grouped by distance bins
            "distance_bins": list,
        }

    Uses seeking captain (same as training) to maintain consistent distribution.
    This ensures model is evaluated on the same distance/signal regime it was trained on.
    """
    obs = sim.reset()
    predictions_list = []
    distances_list = []
    window_size = 60  # Ring buffer size — skip this many warmup steps

    for step in range(int(duration_s)):
        # Get model prediction (single push)
        pred_dict = None
        if engine is not None:
            pred_dict = engine.push(obs, obs.forward_scan)

        # Record prediction and distance after warmup
        if pred_dict is not None and step >= window_size:
            # Sum all species probabilities
            pred_sum = sum(pred_dict["predictions"].values())
            # Compute Euclidean distance to nearest school
            boat_e, boat_n = obs.east_m, obs.north_m
            min_dist = float("inf")
            for school in sim.fish_schools:
                s = school.at(obs.ts)
                dist = math.sqrt((boat_e - s.east_m) ** 2 + (boat_n - s.north_m) ** 2)
                min_dist = min(min_dist, dist)

            # Record (we always have schools in the simulation)
            predictions_list.append(pred_sum)
            distances_list.append(min_dist)

        # Seeking captain: steer toward nearest school (consistent with training)
        nearest_school = None
        nearest_dist = float("inf")
        for school in sim.fish_schools:
            s = school.at(obs.ts)
            dist = math.sqrt((obs.east_m - s.east_m) ** 2 + (obs.north_m - s.north_m) ** 2)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_school = s

        if nearest_school is not None:
            target_heading = np.degrees(np.arctan2(
                nearest_school.east_m - obs.east_m,
                nearest_school.north_m - obs.north_m
            ))
            heading_delta = np.clip(target_heading - obs.heading_deg, -30, 30)
            speed_kts = 3.5 if nearest_dist > 30 else 1.5
        else:
            heading_delta = 0
            speed_kts = 3.5

        obs, _ = sim.step(heading_delta, speed_kts, dt=1.0)

    if len(predictions_list) < 2:
        return {
            "correlation": 0.0,
            "p_value": 1.0,
            "mean_pred_by_distance": {},
            "distance_bins": [],
            "n_observations": len(predictions_list),
            "mean_prediction": 0.0,
            "mean_distance_to_nearest_school": float("inf"),
        }

    predictions_arr = np.array(predictions_list)
    distances_arr = np.array(distances_list)

    # Use inverse forward distance as correlation target
    # (higher prediction when school is closer ahead)
    inv_fwd_dist = 1.0 / np.maximum(distances_arr, 1.0)

    # Compute Pearson correlation
    if len(predictions_arr) > 2:
        corr = _pearson_correlation(predictions_arr, inv_fwd_dist)
        # Simple p-value estimation: correlation is significant if |r| > threshold
        pval = 0.001 if abs(corr) > 0.2 else 0.5
    else:
        corr, pval = 0.0, 1.0

    # Group by forward distance bins (sonar cone is 40m max, so focus on 0-40m)
    bins = [(0, 20), (20, 40), (40, 60), (60, 100), (100, float("inf"))]
    mean_pred_by_distance = {}
    for bin_low, bin_high in bins:
        mask = (distances_arr >= bin_low) & (distances_arr < bin_high)
        if mask.any():
            mean_pred = predictions_arr[mask].mean()
            if bin_high == float("inf"):
                mean_pred_by_distance[f"{bin_low}+m"] = float(mean_pred)
            else:
                mean_pred_by_distance[f"{bin_low}-{bin_high}m"] = float(mean_pred)
        else:
            if bin_high == float("inf"):
                mean_pred_by_distance[f"{bin_low}+m"] = 0.0
            else:
                mean_pred_by_distance[f"{bin_low}-{bin_high}m"] = 0.0

    return {
        "correlation": float(corr),
        "p_value": float(pval),
        "mean_pred_by_distance": mean_pred_by_distance,
        "distance_bins": list(mean_pred_by_distance.keys()),
        "n_observations": len(predictions_arr),
        "mean_prediction": float(predictions_arr.mean()),
        "mean_distance_to_nearest_school": float(distances_arr.mean()),
    }


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
        denom_mean_sample = float(np.mean(denom_sample))
        if denom_mean_sample < 1e-6:
            ratio = float("inf")
        else:
            ratio = float(np.mean(num_sample)) / denom_mean_sample
        ratios.append(ratio)

    ratios = np.array(ratios)
    ci_low = float(np.percentile(ratios, 2.5))
    ci_high = float(np.percentile(ratios, 97.5))
    ratio_mean = np.mean(numerator) / np.mean(denominator)

    return float(ratio_mean), (ci_low, ci_high)
