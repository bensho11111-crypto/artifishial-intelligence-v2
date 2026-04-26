"""
src/processing/fusion.py

Stateful Kalman filter that fuses GpsTick + SonarTick → Observation(s).

Call process(tick) for each tick in timestamp order.
Returns a list of Observations on each valid GPS tick:
  [0]  bottom return (depth = floor depth)
  [1+] fish / secondary returns found above the bottom in the echo array

Deterministic: all timestamps come from ticks, not from time.time().
"""
from __future__ import annotations
import math
from typing import List, Optional
import numpy as np
from ticks.models import Tick, SonarTick, GpsTick, Observation

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
_FWD_N_FORWARD    = 120
_FWD_N_DEPTH      = 80
_FWD_MAX_FORWARD_M = 40.0
_FWD_MAX_DEPTH_M   = 25.0
_FWD_FLOOR_THRESH  = 130   # amplitude threshold for floor in cartesian image
_FWD_FISH_THRESH   = 55    # amplitude threshold for fish returns
_FWD_FLOOR_MARGIN  = 4     # depth pixels to exclude above floor when searching fish
_FWD_COL_STRIDE    = 4     # sample every Nth forward column for observations


def _parse_forward_returns(scan: bytes, east_m: float, north_m: float,
                           heading_deg: float, ts: float, speed_kts: float,
                           floor_depth_m: float = 20.0) -> List[Observation]:
    """
    Extract floor and fish Observations from a forward-scan cartesian image.

    The image is N_DEPTH × N_FORWARD bytes (depth outer, forward inner).
    For each sampled forward column:
      - Scan depth from bottom up for first return ≥ floor threshold → floor Obs
      - Scan depth above floor for peaks ≥ fish threshold → fish Obs

    The expected floor depth (from downward sonar) gates plausible floor returns
    to prevent lateral echoes at unexpected depths being misclassified.
    """
    expected_sz = _FWD_N_DEPTH * _FWD_N_FORWARD
    if not scan or len(scan) < expected_sz:
        return []

    hdg_rad = math.radians(heading_deg)
    fwd_e   = math.sin(hdg_rad)
    fwd_n   = math.cos(hdg_rad)

    step_fwd = _FWD_MAX_FORWARD_M / _FWD_N_FORWARD
    step_dep = _FWD_MAX_DEPTH_M   / _FWD_N_DEPTH

    # Plausible floor depth range (±40% of downward reading to allow slope)
    floor_lo = floor_depth_m * 0.60
    floor_hi = floor_depth_m * 1.40

    observations: List[Observation] = []

    for fx in range(0, _FWD_N_FORWARD, _FWD_COL_STRIDE):
        fwd_m = (fx + 0.5) * step_fwd
        obs_e = east_m  + fwd_m * fwd_e
        obs_n = north_m + fwd_m * fwd_n

        # ── Floor: deepest strong return within expected depth range ──────────
        floor_dz = -1
        for dz in range(_FWD_N_DEPTH - 1, -1, -1):
            amp = scan[dz * _FWD_N_FORWARD + fx]
            if amp >= _FWD_FLOOR_THRESH:
                depth_m = (dz + 0.5) * step_dep
                if floor_lo <= depth_m <= floor_hi:
                    floor_dz = dz
                break

        if floor_dz >= 0:
            depth_m  = (floor_dz + 0.5) * step_dep
            # Confidence: higher for returns close to the expected floor depth
            depth_err = abs(depth_m - floor_depth_m) / max(floor_depth_m, 1.0)
            conf = round(min(0.45, 0.45 * (1.0 - depth_err)), 3)
            observations.append(Observation(
                ts=ts, east_m=obs_e, north_m=obs_n, depth_m=depth_m,
                confidence=max(0.0, conf),
                heading_deg=heading_deg, speed_kts=speed_kts,
                is_floor=True,
            ))

        # ── Fish: bright pixels well above the floor ──────────────────────────
        fish_limit = (floor_dz - _FWD_FLOOR_MARGIN) if floor_dz >= 0 else _FWD_N_DEPTH
        dz = 0
        while dz < fish_limit:
            amp = scan[dz * _FWD_N_FORWARD + fx]
            if amp >= _FWD_FISH_THRESH:
                peak_amp, peak_dz = amp, dz
                j = dz + 1
                while j < fish_limit and scan[j * _FWD_N_FORWARD + fx] >= _FWD_FISH_THRESH // 2:
                    if scan[j * _FWD_N_FORWARD + fx] > peak_amp:
                        peak_amp = scan[j * _FWD_N_FORWARD + fx]
                        peak_dz  = j
                    j += 1
                conf = round(min(0.45, peak_amp / 255.0 * 0.45), 3)
                observations.append(Observation(
                    ts=ts, east_m=obs_e, north_m=obs_n,
                    depth_m=(peak_dz + 0.5) * step_dep,
                    confidence=conf,
                    heading_deg=heading_deg, speed_kts=speed_kts,
                    is_floor=False,
                ))
                dz = max(j, dz + 1)
            else:
                dz += 1

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
            echo_lf=sonar.echo_lf or None,
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

        print(f"[tick ts={gps.ts:7.2f}]  floor {depth:.2f} m  "
              f"E={e:7.1f} N={n_pos:7.1f}  "
              f"[BLUE]  fwd={len(fwd_obs)} pts  fish={len(fish)}")
        for f in fish:
            print(f"               fish echo  depth={f.depth_m:.2f} m  "
                  f"conf={f.confidence:.2f}  [ORANGE]")

        return [bottom] + fish + fwd_obs
