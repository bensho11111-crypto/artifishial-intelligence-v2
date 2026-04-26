"""
src/processing/fusion.py

Stateful Kalman filter that fuses GpsTick + SonarTick → Observation(s).

Call process(tick) for each tick in timestamp order.
Returns a list of Observations on each valid GPS tick:
  [0]  bottom return (depth = floor depth)
  [1+] fish / secondary returns found above the bottom in the echo array

Deterministic: all timestamps come from ticks, not from time.time().

Set FUSION_DEBUG=1 in the environment for per-tick observation logging.
"""
from __future__ import annotations
import math
import os
from typing import List, Optional
import numpy as np
from ticks.models import Tick, SonarTick, GpsTick, Observation

_DEBUG = os.environ.get("FUSION_DEBUG", "") not in ("", "0", "false", "False")

# Must match the constants used in the sonar echo generator
_ECHO_MAX_RANGE_M = 60.0
_FISH_AMPLITUDE_THRESHOLD = 40   # echo counts — noise floor is 2-15
_FISH_MIN_SEPARATION_M    = 1.0  # ignore peaks closer than this


def _latlon_to_enu(lat: float, lon: float,
                   origin_lat: float, origin_lon: float):
    R = 6_371_000.0
    north = R * math.radians(lat - origin_lat)
    east  = R * math.radians(lon - origin_lon) * math.cos(math.radians(origin_lat))
    return east, north


def _parse_echo_returns(echo: bytes, floor_depth_m: float,
                        east_m: float, north_m: float,
                        ts: float, heading_deg: float,
                        speed_kts: float) -> List[Observation]:
    """
    Scan the echo array for secondary amplitude peaks above the bottom.
    Each peak becomes a fish/mid-water Observation.

    Peaks are found by scanning for regions above _FISH_AMPLITUDE_THRESHOLD,
    then taking the maximum within each region.  Only returns above the floor
    (by at least 1m) are reported.
    """
    if not echo or len(echo) < 10:
        return []

    n = len(echo)
    floor_idx = int(floor_depth_m / _ECHO_MAX_RANGE_M * n)
    min_sep_idx = max(1, int(_FISH_MIN_SEPARATION_M / _ECHO_MAX_RANGE_M * n))
    # The floor return is a Gaussian with this sigma (must match generator.py).
    # Its tail at min_sep_idx is still ~88 counts — above threshold — so we cut
    # off at 3×sigma where the tail drops to ~2 counts (noise floor).
    floor_sigma = max(3, int(n * 0.012))
    near_field = int(0.5 / _ECHO_MAX_RANGE_M * n)
    search_end = max(near_field + 1, floor_idx - max(min_sep_idx, 3 * floor_sigma))

    observations: List[Observation] = []
    i = near_field
    while i < search_end:
        if echo[i] >= _FISH_AMPLITUDE_THRESHOLD:
            # Walk forward to find the peak of this return
            peak_val = echo[i]
            peak_idx = i
            j = i + 1
            while j < search_end and echo[j] >= _FISH_AMPLITUDE_THRESHOLD // 2:
                if echo[j] > peak_val:
                    peak_val = echo[j]
                    peak_idx = j
                j += 1
            fish_depth = peak_idx * _ECHO_MAX_RANGE_M / n
            conf = round(min(0.65, peak_val / 200.0), 3)
            observations.append(Observation(
                ts          = ts,
                east_m      = east_m,
                north_m     = north_m,
                depth_m     = fish_depth,
                confidence  = conf,
                heading_deg = heading_deg,
                speed_kts   = speed_kts,
                is_floor    = False,
            ))
            i = max(j, i + min_sep_idx)
        else:
            i += 1

    return observations


# Constants must match synthetic/forward_scan.py
_FWD_N_BEAMS    = 60
_FWD_N_RANGE    = 128
_FWD_MAX_RANGE_M = 40.0
_FWD_BEAM_MIN_DEG = 5.0
_FWD_BEAM_MAX_DEG = 64.0
_FWD_MIN_DEP_DEG  = 15.0   # ignore beams shallower than this — poor depth accuracy
_FWD_N_AZIMUTH    = 24
_FWD_AZIMUTH_HALF_DEG = 30.0
# Floor returns: 185 ± 20 → min 165.  Fish returns: density*160 + 10 → max 154.
# Threshold of 155 cleanly separates the two without overlapping either range.
_FWD_FLOOR_THRESH = 155
_FWD_FISH_THRESH  = 50     # amplitude threshold for mid-water (fish) returns
# Floor Gaussian sigma in forward_scan.py is 1.5 bins; tail at sigma*3≈5 bins
# before the floor peak still reads ~22 counts (below noise floor), so cut off there.
_FWD_FLOOR_SIGMA  = 5      # bins to exclude before floor hit when searching for fish


# Precomputed beam geometry (constant per process)
_FWD_BEAM_DEG = np.linspace(_FWD_BEAM_MIN_DEG, _FWD_BEAM_MAX_DEG, _FWD_N_BEAMS,
                            dtype=np.float64)
_FWD_BEAM_VALID    = _FWD_BEAM_DEG >= _FWD_MIN_DEP_DEG               # (BEAM,) bool
_FWD_BEAM_SIN      = np.sin(np.radians(_FWD_BEAM_DEG))
_FWD_BEAM_COS      = np.cos(np.radians(_FWD_BEAM_DEG))
_FWD_BEAM_ANG_FAC  = ((_FWD_BEAM_DEG - _FWD_MIN_DEP_DEG)
                     / (_FWD_BEAM_MAX_DEG - _FWD_MIN_DEP_DEG))
_FWD_AZ_OFFSET     = np.linspace(-_FWD_AZIMUTH_HALF_DEG, _FWD_AZIMUTH_HALF_DEG,
                                 _FWD_N_AZIMUTH, dtype=np.float64)
_FWD_STEP_M        = _FWD_MAX_RANGE_M / _FWD_N_RANGE


def _parse_forward_returns(scan: bytes, east_m: float, north_m: float,
                           heading_deg: float, ts: float, speed_kts: float,
                           floor_depth_m: float = 20.0) -> List[Observation]:
    """
    Extract floor and fish Observations from a forward-scan frame.

    Vectorised per-(az, beam) floor extraction; per-ray Python loop is only
    entered for fish detection on rays that have any above-threshold bins.
    """
    expected_size = _FWD_N_AZIMUTH * _FWD_N_BEAMS * _FWD_N_RANGE
    if not scan or len(scan) < expected_size:
        return []

    arr = np.frombuffer(scan, dtype=np.uint8, count=expected_size).reshape(
        _FWD_N_AZIMUTH, _FWD_N_BEAMS, _FWD_N_RANGE)

    # World-frame ray direction per azimuth (broadcast over beams later)
    az_world_rad = np.radians(heading_deg + _FWD_AZ_OFFSET)              # (AZ,)
    fwd_e_az     = np.sin(az_world_rad)                                  # (AZ,)
    fwd_n_az     = np.cos(az_world_rad)

    # ── Floor detection: largest ri where amp >= FLOOR_THRESH ────────────────
    mask = arr >= _FWD_FLOOR_THRESH                                      # (AZ, BEAM, RANGE)
    has_floor = mask.any(axis=2)                                         # (AZ, BEAM)
    # argmax on reversed range → distance from end of last True; subtract for index
    last_true = (_FWD_N_RANGE - 1) - mask[:, :, ::-1].argmax(axis=2)     # (AZ, BEAM)
    floor_ri  = np.where(has_floor, last_true, -1)                        # (AZ, BEAM)

    # Tolerance gate (must match scalar version exactly)
    sin_t_b   = _FWD_BEAM_SIN[None, :]                                   # (1, BEAM)
    safe_sin  = np.where(sin_t_b > 0, sin_t_b, 1.0)
    expected  = (floor_depth_m / safe_sin) / _FWD_STEP_M                 # (1, BEAM)
    expected_ri = expected.astype(np.int64)
    expected_ri = np.broadcast_to(expected_ri, floor_ri.shape)
    tolerance = np.maximum(12, (expected_ri * 0.35).astype(np.int64))
    in_tol    = np.abs(floor_ri - expected_ri) <= tolerance
    keep_floor = (floor_ri >= 0) & in_tol & _FWD_BEAM_VALID[None, :]
    # When candidate exists but fails tolerance, scalar version sets floor_ri=-1
    floor_ri = np.where(keep_floor, floor_ri, -1)

    floor_amp = np.zeros(floor_ri.shape, dtype=np.uint8)
    az_idx_g, beam_idx_g = np.where(keep_floor)
    if az_idx_g.size:
        floor_amp[az_idx_g, beam_idx_g] = arr[az_idx_g, beam_idx_g,
                                              floor_ri[az_idx_g, beam_idx_g]]

    observations: List[Observation] = []

    # Emit floor observations
    if az_idx_g.size:
        ri_f       = floor_ri[az_idx_g, beam_idx_g]
        r_f        = (ri_f + 0.5) * _FWD_STEP_M
        cos_t_f    = _FWD_BEAM_COS[beam_idx_g]
        sin_t_f    = _FWD_BEAM_SIN[beam_idx_g]
        ang_fac_f  = _FWD_BEAM_ANG_FAC[beam_idx_g]
        east_f     = east_m  + r_f * fwd_e_az[az_idx_g] * cos_t_f
        north_f    = north_m + r_f * fwd_n_az[az_idx_g] * cos_t_f
        depth_f    = r_f * sin_t_f
        amp_f      = floor_amp[az_idx_g, beam_idx_g]
        conf_f     = np.minimum(0.45, (amp_f / 255.0) * 0.45 * ang_fac_f)
        for k in range(az_idx_g.size):
            observations.append(Observation(
                ts=ts, east_m=float(east_f[k]),
                north_m=float(north_f[k]),
                depth_m=float(depth_f[k]),
                confidence=round(float(conf_f[k]), 3),
                heading_deg=heading_deg, speed_kts=speed_kts,
                is_floor=True,
            ))

    # ── Fish detection: per-ray Python loop, restricted to rays with hits ────
    # Build per-ray search_end (excludes bins behind floor + sigma).
    search_end = np.where(floor_ri >= 0, floor_ri - _FWD_FLOOR_SIGMA, _FWD_N_RANGE)
    search_end = np.maximum(0, search_end)                               # (AZ, BEAM)

    # Quickly rule out rays whose entire searchable region is below threshold.
    # We zero out bins beyond search_end then check if any ≥ FISH_THRESH remain.
    ri_grid = np.arange(_FWD_N_RANGE, dtype=np.int64)[None, None, :]
    in_search = ri_grid < search_end[..., None]
    fish_candidate = (arr >= _FWD_FISH_THRESH) & in_search & _FWD_BEAM_VALID[None, :, None]
    has_fish_ray   = fish_candidate.any(axis=2)                          # (AZ, BEAM)
    fish_az, fish_beam = np.where(has_fish_ray)

    half_thresh = _FWD_FISH_THRESH // 2
    for k in range(fish_az.size):
        a = int(fish_az[k]); b = int(fish_beam[k])
        end = int(search_end[a, b])
        ray = arr[a, b]
        cos_t = _FWD_BEAM_COS[b]
        sin_t = _FWD_BEAM_SIN[b]
        fe    = float(fwd_e_az[a]); fn = float(fwd_n_az[a])
        ri = 0
        while ri < end:
            amp = int(ray[ri])
            if amp >= _FWD_FISH_THRESH:
                peak_amp, peak_ri = amp, ri
                j = ri + 1
                while j < end and int(ray[j]) >= half_thresh:
                    v = int(ray[j])
                    if v > peak_amp:
                        peak_amp = v
                        peak_ri  = j
                    j += 1
                r    = (peak_ri + 0.5) * _FWD_STEP_M
                conf = round(min(0.45, (peak_amp / 255.0) * 0.45), 3)
                observations.append(Observation(
                    ts=ts, east_m=east_m + r*fe*cos_t,
                    north_m=north_m + r*fn*cos_t,
                    depth_m=r*sin_t, confidence=conf,
                    heading_deg=heading_deg, speed_kts=speed_kts,
                    is_floor=False,
                ))
                ri = max(j, ri + 1)
            else:
                ri += 1

    return observations


class Fusion:
    """
    4-state Kalman filter: [east, north, ve, vn] in local ENU.

    process() returns a list:
      - index 0: bottom Observation (always present when GPS+sonar available)
      - index 1+: secondary echo returns (fish arches) above the bottom
    """

    TRANSD_DRAFT_M = 0.30
    MAX_SPEED_KTS  = 15.0
    MAX_HDOP       = 5.0

    def __init__(self):
        self._origin: Optional[tuple] = None
        self._x = np.zeros(4)
        self._P = np.eye(4) * 100.0
        self._Q = np.diag([0.1, 0.1, 0.5, 0.5])
        self._R = np.diag([4.0, 4.0])
        self._last_ts: float = 0.0
        self._pending_sonar: Optional[SonarTick] = None

    def process(self, tick: Tick) -> List[Observation]:
        """
        Feed one Tick. Returns a list of Observations (may be empty).
        Index 0 is the bottom return; subsequent entries are fish echoes.
        """
        if tick.sonar:
            self._pending_sonar = tick.sonar
        if tick.gps:
            return self._update_gps(tick.gps)
        return []

    def reset(self, preserve_position: bool = True) -> None:
        if preserve_position:
            self._x[2] = 0.0
            self._x[3] = 0.0
        else:
            self._x = np.zeros(4)
        self._P = np.eye(4) * 100.0
        self._pending_sonar = None

    @property
    def origin(self) -> Optional[tuple]:
        return self._origin

    def _update_gps(self, gps: GpsTick) -> List[Observation]:
        if not gps.lat or not gps.lon:
            return []
        if gps.speed_kts > self.MAX_SPEED_KTS or gps.hdop > self.MAX_HDOP:
            return []

        dt = max(gps.ts - self._last_ts, 0.001)
        self._last_ts = gps.ts

        if self._origin is None:
            self._origin = (gps.lat, gps.lon)

        east, north = _latlon_to_enu(gps.lat, gps.lon, *self._origin)

        F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
        self._x = F @ self._x
        self._P = F @ self._P @ F.T + self._Q

        H = np.array([[1,0,0,0],[0,1,0,0]])
        y = np.array([east, north]) - H @ self._x
        S = H @ self._P @ H.T + self._R
        K = self._P @ H.T @ np.linalg.inv(S)
        self._x = self._x + K @ y
        self._P = (np.eye(4) - K @ H) @ self._P

        if self._pending_sonar is None:
            return []

        sonar = self._pending_sonar
        e     = float(self._x[0])
        n_pos = float(self._x[1])
        depth = max(0.01, sonar.depth_m - self.TRANSD_DRAFT_M)
        conf  = (sonar.signal_db / 100.0) * (
            1 - 0.3 * min(1.0, gps.speed_kts / self.MAX_SPEED_KTS)
        ) * (1 - 0.2 * min(1.0, (gps.hdop - 1.0) / (self.MAX_HDOP - 1.0)))

        bottom = Observation(
            ts=gps.ts, east_m=e, north_m=n_pos,
            depth_m=depth,
            confidence=round(max(0.0, min(1.0, conf)), 3),
            heading_deg=gps.heading_deg,
            speed_kts=gps.speed_kts,
            echo=sonar.echo or None,
            forward_scan=sonar.forward_scan or None,
        )

        # Parse echo array for secondary returns (fish arches)
        fish = _parse_echo_returns(
            sonar.echo, sonar.depth_m, e, n_pos,
            gps.ts, gps.heading_deg, gps.speed_kts,
        )

        # Extract floor observations from forward-facing scan
        fwd_obs = _parse_forward_returns(
            sonar.forward_scan, e, n_pos,
            gps.heading_deg, gps.ts, gps.speed_kts,
            floor_depth_m=depth,
        ) if sonar.forward_scan else []

        if not _DEBUG:
            return [bottom] + fish + fwd_obs

        print(f"[tick ts={gps.ts:7.2f}]  floor {depth:.2f} m  "
              f"E={e:7.1f} N={n_pos:7.1f}  hdg={gps.heading_deg:5.1f} deg  "
              f"[BLUE]  fwd={len(fwd_obs)} pts  fish={len(fish)}")
        for f in fish:
            print(f"               fish echo  depth={f.depth_m:.2f} m  "
                  f"conf={f.confidence:.2f}  [ORANGE]")

        # ── Diagnostic: actual vs expected forward-obs footprint ────────────
        if fwd_obs:
            actual_e  = [o.east_m  - e     for o in fwd_obs]
            actual_n  = [o.north_m - n_pos for o in fwd_obs]
            # Convert each obs into local boat frame (forward, lateral) so we
            # can read off azimuth offset and forward distance directly.
            hdg_rad = math.radians(gps.heading_deg)
            cos_h, sin_h = math.cos(hdg_rad), math.sin(hdg_rad)
            local_fwd, local_lat, az_offsets = [], [], []
            for de, dn in zip(actual_e, actual_n):
                # boat-forward = (sin h, cos h);  boat-right = (cos h, -sin h)
                fwd = de * sin_h + dn * cos_h
                lat = de * cos_h - dn * sin_h
                local_fwd.append(fwd)
                local_lat.append(lat)
                if fwd > 0.01:
                    az_offsets.append(math.degrees(math.atan2(lat, fwd)))

            # Expected cone footprint in WORLD frame
            R    = _FWD_MAX_RANGE_M
            half = math.radians(_FWD_AZIMUTH_HALF_DEG)
            # cone covers az ∈ [-half, +half] from heading; max horiz extent = R*cos(BEAM_MIN)
            R_horiz = R * math.cos(math.radians(_FWD_MIN_DEP_DEG))
            corners = []
            for az in (-half, 0.0, half):
                world_az = hdg_rad + az
                corners.append((R_horiz * math.sin(world_az),
                                R_horiz * math.cos(world_az)))
            exp_e = [c[0] for c in corners]
            exp_n = [c[1] for c in corners]

            print(f"   POINTS  d_east [{min(actual_e):+6.1f},{max(actual_e):+6.1f}]  "
                  f"d_north [{min(actual_n):+6.1f},{max(actual_n):+6.1f}]")
            print(f"   EXPECT  d_east [{min(exp_e):+6.1f},{max(exp_e):+6.1f}]  "
                  f"d_north [{min(exp_n):+6.1f},{max(exp_n):+6.1f}]  "
                  f"(cone reaches {R_horiz:.1f} m at hdg {gps.heading_deg:.0f} deg)")
            if az_offsets:
                lo, hi = min(az_offsets), max(az_offsets)
                centre_count = sum(1 for a in az_offsets if abs(a) < 2.5)
                print(f"   AZIMUTH  observed [{lo:+6.1f}, {hi:+6.1f}] deg  "
                      f"expected +/-{_FWD_AZIMUTH_HALF_DEG:.0f} deg  "
                      f"(centre+/-2.5deg: {centre_count}/{len(az_offsets)} pts)")

        return [bottom] + fish + fwd_obs
