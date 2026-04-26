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


def _parse_forward_returns(scan: bytes, east_m: float, north_m: float,
                           heading_deg: float, ts: float, speed_kts: float,
                           floor_depth_m: float = 20.0) -> List[Observation]:
    """
    Extract floor and fish Observations from a forward-scan frame.

    Walks every (azimuth, beam) ray. For each:
      - Scan for the first return >= _FWD_FLOOR_THRESH  → floor Observation
      - Scan bins before the floor hit for peaks >= _FWD_FISH_THRESH → fish Observations
    """
    expected_size = _FWD_N_AZIMUTH * _FWD_N_BEAMS * _FWD_N_RANGE
    if not scan or len(scan) < expected_size:
        return []

    step_m   = _FWD_MAX_RANGE_M / _FWD_N_RANGE
    deg_span = _FWD_BEAM_MAX_DEG - _FWD_BEAM_MIN_DEG
    az_span  = 2 * _FWD_AZIMUTH_HALF_DEG

    observations: List[Observation] = []

    for a in range(_FWD_N_AZIMUTH):
        az_offset = -_FWD_AZIMUTH_HALF_DEG + a * az_span / max(_FWD_N_AZIMUTH - 1, 1)
        az_world  = math.radians(heading_deg + az_offset)
        fwd_e     = math.sin(az_world)
        fwd_n     = math.cos(az_world)
        az_base   = a * _FWD_N_BEAMS * _FWD_N_RANGE

        for b in range(_FWD_N_BEAMS):
            theta_deg = _FWD_BEAM_MIN_DEG + b * deg_span / max(_FWD_N_BEAMS - 1, 1)
            if theta_deg < _FWD_MIN_DEP_DEG:
                continue

            theta = math.radians(theta_deg)
            sin_t = math.sin(theta)
            cos_t = math.cos(theta)
            base  = az_base + b * _FWD_N_RANGE

            angle_factor = (theta_deg - _FWD_MIN_DEP_DEG) / (_FWD_BEAM_MAX_DEG - _FWD_MIN_DEP_DEG)

            # ── Find floor return ─────────────────────────────────────────────
            expected_floor_ri = int((floor_depth_m / sin_t) / step_m)
            floor_ri, floor_amp = -1, 0
            for ri in range(_FWD_N_RANGE - 1, -1, -1):
                amp = scan[base + ri]
                if amp >= _FWD_FLOOR_THRESH:
                    tolerance = max(12, int(expected_floor_ri * 0.35))
                    if abs(ri - expected_floor_ri) <= tolerance:
                        floor_amp = amp
                        floor_ri  = ri
                    break

            if floor_ri >= 0:
                r    = (floor_ri + 0.5) * step_m
                conf = round(min(0.45, (floor_amp / 255.0) * 0.45 * angle_factor), 3)
                observations.append(Observation(
                    ts=ts, east_m=east_m + r*fwd_e*cos_t,
                    north_m=north_m + r*fwd_n*cos_t,
                    depth_m=r*sin_t, confidence=conf,
                    heading_deg=heading_deg, speed_kts=speed_kts,
                    is_floor=True,
                ))

            # ── Find fish returns above the floor ─────────────────────────────
            search_end = max(0, floor_ri - _FWD_FLOOR_SIGMA) if floor_ri >= 0 else _FWD_N_RANGE
            ri = 0
            while ri < search_end:
                amp = scan[base + ri]
                if amp >= _FWD_FISH_THRESH:
                    peak_amp, peak_ri = amp, ri
                    j = ri + 1
                    while j < search_end and scan[base + j] >= _FWD_FISH_THRESH // 2:
                        if scan[base + j] > peak_amp:
                            peak_amp = scan[base + j]
                            peak_ri  = j
                        j += 1
                    r    = (peak_ri + 0.5) * step_m
                    conf = round(min(0.45, (peak_amp / 255.0) * 0.45), 3)
                    observations.append(Observation(
                        ts=ts, east_m=east_m + r*fwd_e*cos_t,
                        north_m=north_m + r*fwd_n*cos_t,
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
