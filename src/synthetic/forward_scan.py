"""
src/synthetic/forward_scan.py

Simulates one frame of a forward-facing sonar (e.g. Garmin LiveScope Forward).

Geometry: a vertical fan of beams pointing ahead of and below the boat.
  - Beam 0   = BEAM_MIN_DEG depression (nearly horizontal, long forward range)
  - Beam N-1 = BEAM_MAX_DEG depression (steep, shorter range)
  - Each beam steps along its ray and records the first strong return (floor)
    plus any mid-water returns (fish schools).

Output: N_BEAMS * N_RANGE bytes, row-major uint8 (beam index varies slowest).
"""
from __future__ import annotations
import math
import random as _random
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from synthetic.floor import FloorModel
    from synthetic.generator import FishSchool

# ── Public constants (must be mirrored in the frontend JS) ────────────────────
MAX_RANGE_M   = 40.0
N_BEAMS       = 60
N_RANGE       = 128
BEAM_MIN_DEG  = 5.0
BEAM_MAX_DEG  = 64.0


def generate(east_m: float, north_m: float, heading_deg: float,
             floor: "FloorModel", schools: "List[FishSchool]",
             rng: Optional[_random.Random] = None) -> bytes:
    """
    Return N_BEAMS * N_RANGE bytes representing one forward-scan frame.
    """
    rng = rng or _random.Random()

    scan = bytearray(N_BEAMS * N_RANGE)

    # Background noise matching down-scan noise floor
    for i in range(len(scan)):
        scan[i] = rng.randint(2, 12)

    hdg_rad = math.radians(heading_deg)
    fwd_e   = math.sin(hdg_rad)   # east component of boat's forward direction
    fwd_n   = math.cos(hdg_rad)   # north component

    step_m = MAX_RANGE_M / N_RANGE

    for b in range(N_BEAMS):
        theta = math.radians(
            BEAM_MIN_DEG + b * (BEAM_MAX_DEG - BEAM_MIN_DEG) / max(N_BEAMS - 1, 1)
        )
        sin_t = math.sin(theta)
        cos_t = math.cos(theta)

        for ri in range(N_RANGE):
            r     = (ri + 0.5) * step_m
            ray_e = east_m  + r * fwd_e * cos_t
            ray_n = north_m + r * fwd_n * cos_t
            ray_d = r * sin_t               # depth along this ray

            floor_d = floor.depth_at(ray_e, ray_n)

            if ray_d >= floor_d:
                # Floor return — Gaussian blob, then stop tracing this beam
                amp = min(255, 185 + rng.randint(-20, 20))
                for dr in range(-3, 4):
                    ri2 = ri + dr
                    if 0 <= ri2 < N_RANGE:
                        v = int(amp * math.exp(-0.5 * (dr / 1.5) ** 2))
                        idx = b * N_RANGE + ri2
                        scan[idx] = min(255, scan[idx] + v)
                break

            # Fish school returns
            for s in schools:
                dx   = ray_e - s.east_m
                dy   = ray_n - s.north_m
                dz   = ray_d - s.depth_m
                if dx*dx + dy*dy + dz*dz < s.radius_m ** 2:
                    amp = int(s.density * 160) + rng.randint(-10, 10)
                    idx = b * N_RANGE + ri
                    scan[idx] = min(255, scan[idx] + max(0, amp))

    return bytes(scan)
