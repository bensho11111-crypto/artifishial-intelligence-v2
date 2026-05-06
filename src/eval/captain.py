"""
Captain agents that operate in SteerableSimulator.

Each captain decides (heading_delta_deg, speed_kts) based on state and optional model predictions.
"""
import math
from abc import ABC, abstractmethod
from collections import deque
import random


class CaptainAgent(ABC):
    """Base class for boat captain agents."""

    @abstractmethod
    def decide(self, state: dict, predictions: dict = None) -> tuple[float, float]:
        """
        Decide heading change and speed.

        Args:
            state: dict with keys east_m, north_m, heading_deg, speed_kts, depth_m, t
            predictions: None or dict with "horizon_s" and "predictions" (species -> prob)

        Returns:
            (heading_delta_deg, speed_kts)
        """
        pass


class RandomCaptain(CaptainAgent):
    """Random walk baseline: uniform ±30° heading, uniform 2-5 kts speed."""

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def decide(self, state: dict, predictions: dict = None) -> tuple[float, float]:
        heading_delta = self._rng.uniform(-30.0, 30.0)
        speed_kts = self._rng.uniform(2.0, 5.0)
        return heading_delta, speed_kts


class StraightCaptain(CaptainAgent):
    """Sinusoidal drift baseline: gentle oscillation at constant ~3.5 kts."""

    def decide(self, state: dict, predictions: dict = None) -> tuple[float, float]:
        t = state["t"]
        heading_delta = 5.0 * math.sin(0.02 * t)
        speed_kts = 3.5
        return heading_delta, speed_kts


class ModelGuidedCaptain(CaptainAgent):
    """Gradient-following agent: smooths predictions and explores when rising."""

    HISTORY_LEN = 5
    EXPLORE_TICKS = 10
    DEAD_BAND = 0.02
    EXPLORE_DELTAS = [-30.0, 30.0]

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)
        self._tick_count = 0
        self._pred_history = deque(maxlen=self.HISTORY_LEN)
        self._mode = "normal"  # "normal" or "explore"
        self._explore_dir = 0  # 0 for -30, 1 for +30
        self._explore_ticks = 0

    def decide(self, state: dict, predictions: dict = None) -> tuple[float, float]:
        """
        Gradient-following logic:
        - First 60 ticks: gentle spiral to explore
        - After: smooth predictions, exploit when stable/rising, explore when falling
        """
        t_index = int(state["t"])  # tick count

        # Phase 1: Pre-model (no predictions yet)
        if t_index < 60:
            return 5.0, 3.5

        # Phase 2: Model-guided (predictions available)
        # Compute smoothed prediction sum
        if predictions is not None and "predictions" in predictions:
            pred_sum = sum(predictions["predictions"].values())
        else:
            pred_sum = 0.0

        self._pred_history.append(pred_sum)
        smoothed_sum = sum(self._pred_history) / len(self._pred_history)

        # Compute decision
        last_pred_sum = (
            self._pred_history[-2]
            if len(self._pred_history) >= 2
            else smoothed_sum
        )

        # Normal mode
        if self._mode == "normal":
            if smoothed_sum >= last_pred_sum - self.DEAD_BAND:
                # Rising or flat within dead-band: hold current heading
                return 0.0, 3.5
            else:
                # Falling: enter explore mode
                self._mode = "explore"
                self._explore_dir = self._rng.choice([0, 1])
                self._explore_ticks = 0

        # Explore mode
        if self._mode == "explore":
            heading_delta = self.EXPLORE_DELTAS[self._explore_dir]
            self._explore_ticks += 1

            # Exit explore if prediction rose
            if smoothed_sum > last_pred_sum:
                self._mode = "normal"
                return heading_delta, 3.5

            # Exit and flip direction if timer expired
            if self._explore_ticks >= self.EXPLORE_TICKS:
                self._explore_dir = 1 - self._explore_dir
                self._explore_ticks = 0

            return heading_delta, 3.5

        return 0.0, 3.5


class OracleCaptain(CaptainAgent):
    """Oracle: steers toward nearest fish school (cheats; upper bound)."""

    def __init__(self, schools: list, seed: int = 42):
        """
        Args:
            schools: list of FishSchool objects from session.fish_schools
            seed: unused (deterministic)
        """
        self._schools = schools

    def decide(self, state: dict, predictions: dict = None) -> tuple[float, float]:
        """Steer toward nearest school at current time t."""
        t = state["t"]
        boat_e = state["east_m"]
        boat_n = state["north_m"]
        boat_heading = state["heading_deg"]
        boat_speed = state["speed_kts"]

        # Find nearest school
        nearest_school = None
        min_dist = float("inf")
        for school in self._schools:
            s = school.at(t)
            dist = math.sqrt((boat_e - s.east_m) ** 2 + (boat_n - s.north_m) ** 2)
            if dist < min_dist:
                min_dist = dist
                nearest_school = s

        if nearest_school is None:
            return 0.0, 3.5

        # Steer toward school
        angle_to_school = math.degrees(
            math.atan2(
                nearest_school.east_m - boat_e,
                nearest_school.north_m - boat_n,
            )
        ) % 360.0

        # Shortest arc difference
        delta = (angle_to_school - boat_heading + 180) % 360 - 180
        heading_delta = max(-30.0, min(30.0, delta))

        # Slow down inside school radius
        if min_dist < nearest_school.radius_m:
            speed_kts = 1.5
        else:
            speed_kts = 3.5

        return heading_delta, speed_kts
