"""
src/synthetic/forward_scan.py

Simulates one frame of a forward-facing sonar (e.g. Garmin LiveScope Forward).

Geometry: a 3D cone of beams covering a vertical fan (5°–64° depression) AND a
horizontal azimuthal spread (±67°, 134° total), matching the approximate FOV of
real forward-facing units.  Each (vertical, horizontal) beam pair is a ray that
is marched in 3D; returns are projected onto a 2D Cartesian image:

    x-axis: forward distance along the boat's heading (0 → MAX_FORWARD_M)
    y-axis: depth below surface                       (0 → MAX_DEPTH_M)

A horizontal beam-taper (Gaussian on azimuth angle) attenuates returns off the
main axis, mimicking the phased-array beam pattern.  A depth-dependent volume
reverberation term raises the noise floor near the surface.

Output: N_DEPTH × N_FORWARD bytes (row-major, depth index outer).

Differences from a real unit:
  - No multi-ping coherent processing (each frame is independent)
  - Uniform noise rather than correlated reverberation
  - Simplified swimbladder/aspect models for fish amplitude
"""
from __future__ import annotations
import math
import random as _random
from typing import TYPE_CHECKING, List, Optional

import numpy as np

if TYPE_CHECKING:
    from synthetic.floor import FloorModel
    from synthetic.generator import FishSchool

# ── Output image dimensions (must be mirrored in frontend JS) ─────────────────
N_FORWARD     = 120    # pixels along heading axis
N_DEPTH       = 80     # pixels along depth axis
MAX_FORWARD_M = 40.0   # metres
MAX_DEPTH_M   = 25.0   # metres

# ── Internal beam grid (not exported — only affects quality/speed) ─────────────
_N_VERT       = 30     # vertical beams  (depression 5° → 64°)
_N_HORIZ      = 60     # horizontal beams (azimuth −67° → +67°)
_N_RANGE      = 80     # range steps per beam
_BEAM_MIN_DEG = 5.0
_BEAM_MAX_DEG = 64.0
_HORIZ_FOV    = 134.0  # total horizontal FOV (degrees)
_SIGMA_H_DEG  = 22.0   # Gaussian taper sigma — controls brightness rolloff toward edges

# ── Backward-compat constants still used by fusion.py ─────────────────────────
# (kept so existing import sites don't break; values now describe the image)
N_BEAMS  = N_DEPTH    # aliased for legacy imports
N_RANGE  = N_FORWARD  # aliased for legacy imports


def generate(east_m: float, north_m: float, heading_deg: float,
             floor: "FloorModel", schools: "List[FishSchool]",
             rng: Optional[_random.Random] = None) -> bytes:
    """
    Simulate one forward-facing sonar frame.

    Returns N_DEPTH * N_FORWARD bytes (uint8, row-major: depth outer).
    """
    rng = rng or _random.Random()

    # ── Pre-compute beam geometry ──────────────────────────────────────────────
    hdg_rad  = math.radians(heading_deg)
    fwd_e    = math.sin(hdg_rad)    # heading unit vector — east component
    fwd_n    = math.cos(hdg_rad)    # heading unit vector — north component
    right_e  =  math.cos(hdg_rad)  # rightward perpendicular — east
    right_n  = -math.sin(hdg_rad)  # rightward perpendicular — north

    theta_v   = np.linspace(_BEAM_MIN_DEG, _BEAM_MAX_DEG, _N_VERT)   # (N_VERT,)
    theta_h   = np.linspace(-_HORIZ_FOV/2, _HORIZ_FOV/2, _N_HORIZ)  # (N_HORIZ,)
    step_m    = MAX_FORWARD_M / _N_RANGE

    tv, th   = np.meshgrid(np.radians(theta_v), np.radians(theta_h), indexing='ij')
    cos_tv   = np.cos(tv)   # (N_VERT, N_HORIZ)
    sin_tv   = np.sin(tv)
    cos_th   = np.cos(th)
    sin_th   = np.sin(th)

    # Forward and lateral components of the horizontal direction per beam
    fwd_comp   = cos_tv * cos_th   # how much of the ray goes forward
    right_comp = cos_tv * sin_th   # how much goes sideways

    # Horizontal beam taper (Gaussian on azimuth angle) — shape (N_HORIZ,)
    taper = np.exp(-0.5 * (theta_h / _SIGMA_H_DEG) ** 2)

    # ── Ray positions for all (v, h, r) ───────────────────────────────────────
    r_vals = (np.arange(_N_RANGE) + 0.5) * step_m   # (N_RANGE,)

    # Broadcast to (N_VERT, N_HORIZ, N_RANGE)
    r3 = r_vals[np.newaxis, np.newaxis, :]
    fc = fwd_comp[:, :, np.newaxis]
    rc = right_comp[:, :, np.newaxis]

    ray_e = east_m  + r3 * (fwd_e * fc + right_e * rc)
    ray_n = north_m + r3 * (fwd_n * fc + right_n * rc)
    ray_d = r3 * sin_tv[:, :, np.newaxis]   # depth (positive down)

    # ── Floor depth at every ray position (vectorised grid lookup) ────────────
    ix = np.clip(((ray_e + floor._half) / floor.CELL_SIZE_M).astype(np.int32),
                 0, floor._n - 1)
    iy = np.clip(((ray_n + floor._half) / floor.CELL_SIZE_M).astype(np.int32),
                 0, floor._n - 1)
    floor_d3 = floor._grid[ix, iy]   # (N_VERT, N_HORIZ, N_RANGE)

    # ── Build output image ────────────────────────────────────────────────────
    # Volume reverberation: noise decays with depth (realistic water column scatter)
    image = np.zeros((N_DEPTH, N_FORWARD), dtype=np.float32)
    for dz in range(N_DEPTH):
        d_m = (dz + 0.5) * MAX_DEPTH_M / N_DEPTH
        reverb = 12.0 * math.exp(-0.08 * d_m)
        for fx in range(N_FORWARD):
            image[dz, fx] = rng.uniform(2, max(2.5, reverb))

    # ── Floor returns ─────────────────────────────────────────────────────────
    below = ray_d >= floor_d3                            # (N_VERT, N_HORIZ, N_RANGE)
    has_floor = below.any(axis=2)                        # (N_VERT, N_HORIZ)
    floor_ri  = np.where(has_floor, below.argmax(axis=2), -1)

    floor_atten_2way = 0.006   # average two-way absorption Np/m

    for v in range(_N_VERT):
        for h in range(_N_HORIZ):
            ri = int(floor_ri[v, h])
            if ri < 0:
                continue
            r = (ri + 0.5) * step_m
            proj_fwd = r * float(fwd_comp[v, h])
            proj_dep = r * float(sin_tv[v, h])
            fx = int(proj_fwd / MAX_FORWARD_M * N_FORWARD)
            dz = int(proj_dep / MAX_DEPTH_M   * N_DEPTH)
            if not (0 <= fx < N_FORWARD and 0 <= dz < N_DEPTH):
                continue
            t = float(taper[h])
            atten = math.exp(-2.0 * floor_atten_2way * proj_dep)
            amp = 185.0 * t * atten * rng.uniform(0.88, 1.12)
            image[dz, fx] = max(image[dz, fx], amp)

    # ── Fish school returns ───────────────────────────────────────────────────
    for s in schools:
        r2 = s.radius_m ** 2
        # Check each (v, h, r) against the school sphere
        dx3 = ray_e - s.east_m
        dy3 = ray_n - s.north_m
        dz3 = ray_d - s.depth_m
        in_school = (dx3*dx3 + dy3*dy3 + dz3*dz3) < r2   # (N_VERT, N_HORIZ, N_RANGE)

        # Mask out anything at or beyond the floor in this beam
        # so fish behind the floor don't contribute
        beyond_floor = np.cumsum(below, axis=2) > 0
        in_school &= ~beyond_floor

        has_fish = in_school.any(axis=2)   # (N_VERT, N_HORIZ)
        fish_ri  = np.where(has_fish, in_school.argmax(axis=2), -1)

        ts_avg  = (getattr(s, 'ts_lf', 0.7) + getattr(s, 'ts_hf', 0.7)) / 2.0
        for v in range(_N_VERT):
            for h in range(_N_HORIZ):
                ri = int(fish_ri[v, h])
                if ri < 0:
                    continue
                r = (ri + 0.5) * step_m
                proj_fwd = r * float(fwd_comp[v, h])
                proj_dep = r * float(sin_tv[v, h])
                fx = int(proj_fwd / MAX_FORWARD_M * N_FORWARD)
                dz = int(proj_dep / MAX_DEPTH_M   * N_DEPTH)
                if not (0 <= fx < N_FORWARD and 0 <= dz < N_DEPTH):
                    continue
                t = float(taper[h])
                atten = math.exp(-0.006 * proj_dep)
                amp = s.density * ts_avg * t * atten * 180.0
                amp += rng.uniform(-10, 10)
                image[dz, fx] = max(image[dz, fx], amp)

    out = np.clip(image, 0, 255).astype(np.uint8)
    return out.tobytes()
