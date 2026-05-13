"""
Steerable fishing simulator for captain agent evaluation.

Decouples boat control from the synthetic generator's hardcoded trajectory.
Reuses fish schools and floor model from a short throwaway generate() call.
"""
import math
import random

from synthetic.generator import generate, SPECIES_NAMES
from synthetic.forward_scan import generate as fwd_gen
from ticks.models import Observation


class SteerableSimulator:
    """
    Step-by-step fishing simulation with injectable boat control.

    Boat state: (east_m, north_m, heading_deg, speed_kts, t)
    Catches: Poisson λ = density × overlap × 0.35 per school per step
    """

    SOFT_LIMIT = 300.0
    BOUNDARY_NUDGE_CAP = 15.0

    def __init__(self, seed: int = 42):
        """
        Initialize simulator from synthetic world.

        Calls generate(duration_s=90.0, seed=seed) to extract fish schools
        and floor model (90s route gives valid indices 5, 30, 60 for schools).
        Discards the hardcoded trajectory.
        """
        session = generate(duration_s=90.0, seed=seed)
        self._schools = session.fish_schools  # List[FishSchool], 4 schools
        self._floor = session.floor           # FloorModel

        self._rng = random.Random(seed)
        self._seed = seed

        # Boat state (initialized in reset())
        self._east_m = 0.0
        self._north_m = 0.0
        self._heading = 0.0      # compass degrees from north
        self._speed_kts = 3.5
        self._t = 0.0

    @property
    def fish_schools(self):
        return self._schools

    @property
    def current_time(self):
        return self._t

    def reset(self) -> Observation:
        """Reset boat to origin facing north, return initial observation."""
        self._east_m = 0.0
        self._north_m = 0.0
        self._heading = 0.0
        self._speed_kts = 3.5
        self._t = 0.0
        depth = self._floor.depth_at(0.0, 0.0)
        return self._make_obs(depth)

    def step(
        self,
        heading_delta_deg: float,
        speed_kts: float,
        dt: float = 1.0,
    ) -> tuple[Observation, list[dict]]:
        """
        Advance simulation by dt seconds with captain command.

        Args:
            heading_delta_deg: change in heading (degrees)
            speed_kts: requested speed in knots
            dt: timestep in seconds

        Returns:
            (Observation, list of catch events)
        """
        # 1. Boundary nudge (before applying captain's delta)
        nudge = self._boundary_nudge()
        self._heading = (self._heading + heading_delta_deg + nudge) % 360.0

        # 2. Clamp speed and update physics
        self._speed_kts = max(0.5, min(speed_kts, 15.0))
        speed_ms = self._speed_kts * 0.5144  # knots to m/s
        h_rad = math.radians(self._heading)
        self._east_m += speed_ms * dt * math.sin(h_rad)
        self._north_m += speed_ms * dt * math.cos(h_rad)
        self._t += dt

        # 3. Generate depth + sonar
        depth = max(0.6, self._floor.depth_at(self._east_m, self._north_m))
        fwd_rng = random.Random(int(self._t * 997) ^ int(self._east_m * 31))
        schools_now = [s.at(self._t) for s in self._schools]
        fwd_bytes = fwd_gen(
            self._east_m, self._north_m, self._heading,
            self._floor, schools_now, fwd_rng
        )

        # 4. Catch events: Poisson λ = density × overlap × 0.35
        # Increased from 0.08 to generate ~5-10% positive labels in training data
        catches = []
        for s in schools_now:
            dist = math.sqrt(
                (self._east_m - s.east_m)**2 + (self._north_m - s.north_m)**2
            )
            if dist < s.radius_m:
                overlap = 1.0 - dist / s.radius_m
                lam = s.density * overlap * 0.35
                if self._rng.random() < 1.0 - math.exp(-lam):
                    catches.append({
                        "ts": self._t,
                        "species": SPECIES_NAMES.get(s.species, s.species),
                    })

        return self._make_obs(depth, fwd_bytes), catches

    def _make_obs(self, depth: float, fwd_bytes=None) -> Observation:
        """Construct an Observation from current state."""
        return Observation(
            ts=self._t,
            east_m=self._east_m,
            north_m=self._north_m,
            depth_m=depth,
            confidence=0.85,  # synthetic, constant
            heading_deg=self._heading,
            speed_kts=self._speed_kts,
            is_floor=True,
            forward_scan=fwd_bytes,
        )

    def _boundary_nudge(self) -> float:
        """
        Soft nudge back toward center if boat drifts outside limit.

        Returns: heading delta in degrees (capped ±15°)
        """
        if abs(self._east_m) > self.SOFT_LIMIT or abs(self._north_m) > self.SOFT_LIMIT:
            # Heading toward center (0, 0)
            angle_to_center = math.degrees(
                math.atan2(-self._east_m, -self._north_m)
            ) % 360.0
            # Shortest arc difference
            delta = (angle_to_center - self._heading + 180) % 360 - 180
            # Proportional nudge, capped
            return max(-self.BOUNDARY_NUDGE_CAP, min(self.BOUNDARY_NUDGE_CAP, delta * 0.5))
        return 0.0
