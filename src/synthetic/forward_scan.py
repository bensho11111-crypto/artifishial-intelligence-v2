"""
src/synthetic/forward_scan.py

Simulates one frame of a forward-facing sonar (LiveScope-style "Forward" mode).

Geometry: a 3D wedge ahead of the boat.
  - Vertical:   N_BEAMS depression angles, BEAM_MIN_DEG..BEAM_MAX_DEG
  - Horizontal: N_AZIMUTH azimuth angles, swept across ±AZIMUTH_HALF_DEG
                of the boat's heading
  - Range:      N_RANGE bins out to MAX_RANGE_M

Output: N_AZIMUTH * N_BEAMS * N_RANGE bytes, row-major uint8.
        Index = ((az * N_BEAMS) + beam) * N_RANGE + range
"""
from __future__ import annotations
import math
import random as _random
import numpy as np
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from synthetic.floor import FloorModel
    from synthetic.generator import FishSchool

# ── Public constants (must be mirrored in the frontend JS) ────────────────────
MAX_RANGE_M       = 40.0
N_BEAMS           = 60     # vertical (depression) beams
N_RANGE           = 128    # range bins
BEAM_MIN_DEG      = 5.0
BEAM_MAX_DEG      = 64.0
N_AZIMUTH         = 24     # horizontal beams (swept across the cone)
AZIMUTH_HALF_DEG  = 30.0   # ±30° → 60° total horizontal FOV (LiveScope-like)


def generate(east_m: float, north_m: float, heading_deg: float,
             floor: "FloorModel", schools: "List[FishSchool]",
             rng: Optional[_random.Random] = None) -> bytes:
    """
    Return N_AZIMUTH * N_BEAMS * N_RANGE bytes representing one forward-scan
    wedge. Vectorised with numpy.
    """
    rng = rng or _random.Random()
    np_rng = np.random.default_rng(rng.randint(0, 2**32 - 1))

    # ── Build ray geometry (azimuth, beam, range) ────────────────────────────
    az_offsets = np.linspace(-AZIMUTH_HALF_DEG, AZIMUTH_HALF_DEG, N_AZIMUTH,
                             dtype=np.float32)
    beam_degs  = np.linspace(BEAM_MIN_DEG, BEAM_MAX_DEG, N_BEAMS,
                             dtype=np.float32)
    step_m     = MAX_RANGE_M / N_RANGE
    r_arr      = (np.arange(N_RANGE, dtype=np.float32) + 0.5) * step_m

    # World-frame heading direction for each azimuth offset
    az_world_rad = np.radians(heading_deg + az_offsets)        # (N_AZIMUTH,)
    fwd_e = np.sin(az_world_rad)                                # (N_AZIMUTH,)
    fwd_n = np.cos(az_world_rad)

    theta = np.radians(beam_degs)                               # (N_BEAMS,)
    sin_t = np.sin(theta)                                       # (N_BEAMS,)
    cos_t = np.cos(theta)

    # Broadcast to (N_AZIMUTH, N_BEAMS, N_RANGE)
    horiz = cos_t[None, :, None] * r_arr[None, None, :]         # horizontal slant
    ray_e = east_m  + fwd_e[:, None, None] * horiz
    ray_n = north_m + fwd_n[:, None, None] * horiz
    ray_d = sin_t[None, :, None] * r_arr[None, None, :]         # depth along ray

    # ── Floor depth lookup (vectorised against floor._grid) ──────────────────
    n_cells = floor._n
    half    = floor._half
    cell    = floor.CELL_SIZE_M
    ix = np.clip(((ray_e + half) / cell).astype(np.int32), 0, n_cells - 1)
    iy = np.clip(((ray_n + half) / cell).astype(np.int32), 0, n_cells - 1)
    floor_d = floor._grid[ix, iy]                               # (AZ, BEAM, RANGE)

    below_floor = ray_d >= floor_d                              # bool mask

    # First range bin per (az, beam) where ray descends below floor
    # argmax on a bool array returns the first True index, or 0 if all False.
    first_hit = np.argmax(below_floor, axis=2)                  # (AZ, BEAM)
    has_hit   = below_floor.any(axis=2)                         # (AZ, BEAM)

    # ── Background noise ──────────────────────────────────────────────────────
    scan = np_rng.integers(2, 13, size=(N_AZIMUTH, N_BEAMS, N_RANGE),
                           dtype=np.int16)

    # ── Floor returns: Gaussian blob centred on first_hit ────────────────────
    # Add along the range axis with a ±3 bin spread (sigma ≈ 1.5).
    blob_offsets = np.arange(-3, 4, dtype=np.int32)             # (7,)
    blob_weights = np.exp(-0.5 * (blob_offsets / 1.5) ** 2)     # (7,)
    floor_amp = 185 + np_rng.integers(-20, 21, size=first_hit.shape,
                                       dtype=np.int16)          # (AZ, BEAM)

    az_idx, beam_idx = np.meshgrid(np.arange(N_AZIMUTH), np.arange(N_BEAMS),
                                    indexing='ij')              # (AZ, BEAM)
    for k, (off, w) in enumerate(zip(blob_offsets, blob_weights)):
        ri2 = first_hit + off
        valid = has_hit & (ri2 >= 0) & (ri2 < N_RANGE)
        if not valid.any():
            continue
        a = az_idx[valid]
        b = beam_idx[valid]
        r = ri2[valid]
        v = (floor_amp[valid] * w).astype(np.int16)
        scan[a, b, r] = np.minimum(255, scan[a, b, r] + v)

    # ── Floor occlusion: zero out everything past the floor hit ──────────────
    # For beams that hit, returns beyond first_hit + 3 must remain background.
    # We achieve this by clearing (az, beam, ri > first_hit+3) before fish add.
    ri_grid = np.arange(N_RANGE, dtype=np.int32)[None, None, :]
    cutoff  = (first_hit + 4)[..., None]                        # (AZ, BEAM, 1)
    past_floor = has_hit[..., None] & (ri_grid >= cutoff)
    # Replace with fresh background noise (already in scan; just suppress fish).
    # We'll apply this mask AFTER fish are added.

    # ── Fish school returns ──────────────────────────────────────────────────
    if schools:
        for s in schools:
            dx = ray_e - s.east_m
            dy = ray_n - s.north_m
            dz = ray_d - s.depth_m
            inside = (dx*dx + dy*dy + dz*dz) < (s.radius_m ** 2)
            if not inside.any():
                continue
            base_amp = int(s.density * 160)
            jitter = np_rng.integers(-10, 11, size=inside.shape, dtype=np.int16)
            add = np.where(inside, np.maximum(0, base_amp + jitter), 0)
            scan = np.minimum(255, scan + add)

    # Apply floor occlusion (suppress anything past the floor hit, then refill
    # with low-level noise so the blocked region looks like attenuated water).
    if past_floor.any():
        suppressed = np_rng.integers(2, 8, size=scan.shape, dtype=np.int16)
        scan = np.where(past_floor, suppressed, scan)

    return scan.astype(np.uint8).tobytes()
