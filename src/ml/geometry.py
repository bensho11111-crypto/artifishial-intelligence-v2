import numpy as np
from ml.config import ModelConfig


def build_geometry_tensor(cfg: ModelConfig) -> np.ndarray:
    """
    Returns shape (N_AZ, N_BEAM, N_RANGE, 3), float32.
    Axis 3: [delta_east_m, delta_north_m, depth_m] in boat frame
    (boat-forward = north, boat-right = east, heading=0 convention).

    Source: forward_scan.py lines 46–66, extracted as pure numpy.
    """
    az_offsets  = np.linspace(-cfg.az_half_deg, cfg.az_half_deg, cfg.n_az)
    beam_degs   = np.linspace(cfg.beam_min_deg, cfg.beam_max_deg, cfg.n_beam)
    step_m      = cfg.scan_max_range_m / cfg.n_range
    r_arr       = (np.arange(cfg.n_range) + 0.5) * step_m

    az_rad  = np.radians(az_offsets)    # (AZ,)
    fwd_e   = np.sin(az_rad)            # east component of forward direction
    fwd_n   = np.cos(az_rad)            # north component
    sin_b   = np.sin(np.radians(beam_degs))  # (BEAM,)
    cos_b   = np.cos(np.radians(beam_degs))

    # Broadcast: (AZ, BEAM, RANGE)
    horiz  = cos_b[None, :, None] * r_arr[None, None, :]
    de     = fwd_e[:, None, None] * horiz   # Δeast
    dn     = fwd_n[:, None, None] * horiz   # Δnorth
    depth  = sin_b[None, :, None] * r_arr[None, None, :]  # (1, BEAM, RANGE)
    depth  = np.repeat(depth, cfg.n_az, axis=0)            # (AZ, BEAM, RANGE)

    return np.stack([de, dn, depth], axis=-1).astype(np.float32)
