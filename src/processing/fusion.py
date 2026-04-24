"""
src/processing/fusion.py

Stateful Kalman filter that fuses GpsTick + SonarTick → Observation.

Call process(tick) for each tick in timestamp order.
The filter emits an Observation whenever it has both a GPS fix and
a pending sonar reading.

Deterministic: all timestamps come from ticks, not from time.time().
"""
from __future__ import annotations
import math
from typing import Optional
import numpy as np
from ticks.models import Tick, SonarTick, GpsTick, Observation


def _latlon_to_enu(lat: float, lon: float,
                   origin_lat: float, origin_lon: float):
    R = 6_371_000.0
    north = R * math.radians(lat - origin_lat)
    east  = R * math.radians(lon - origin_lon) * math.cos(math.radians(origin_lat))
    return east, north


class Fusion:
    """
    4-state Kalman filter: [east, north, ve, vn] in local ENU.

    Process noise: Q = diag([0.1, 0.1, 0.5, 0.5])
    GPS noise:     R = diag([4.0, 4.0])  (2m sigma)
    """

    TRANSD_DRAFT_M = 0.30     # transducer depth below waterline
    MAX_SPEED_KTS  = 15.0
    MAX_HDOP       = 5.0
    SOUND_VEL_MS   = 1500.0

    def __init__(self):
        self._origin: Optional[tuple] = None   # (lat, lon) set on first fix
        self._x = np.zeros(4)
        self._P = np.eye(4) * 100.0
        self._Q = np.diag([0.1, 0.1, 0.5, 0.5])
        self._R = np.diag([4.0, 4.0])
        self._last_ts: float = 0.0
        self._pending_sonar: Optional[SonarTick] = None
        self._pending_gps:   Optional[GpsTick]   = None

    # ── public API ────────────────────────────────────────────────────────────

    def process(self, tick: Tick) -> Optional[Observation]:
        """Feed one Tick. Returns an Observation or None."""
        if tick.sonar:
            self._pending_sonar = tick.sonar
        if tick.gps:
            return self._update_gps(tick.gps)
        return None

    def reset(self, preserve_position: bool = True) -> None:
        """
        Reset for seek / loop restart.
        preserve_position=True: keep last known ENU position so the boat
        marker does not snap to (0,0).  Only velocity is zeroed.
        """
        if preserve_position:
            self._x[2] = 0.0   # ve
            self._x[3] = 0.0   # vn
        else:
            self._x = np.zeros(4)
        self._P = np.eye(4) * 100.0
        self._pending_sonar = None
        # _origin and _x[0:2] preserved if preserve_position=True

    @property
    def origin(self) -> Optional[tuple]:
        return self._origin

    # ── internal ──────────────────────────────────────────────────────────────

    def _update_gps(self, gps: GpsTick) -> Optional[Observation]:
        if not gps.lat or not gps.lon:
            return None
        if gps.speed_kts > self.MAX_SPEED_KTS or gps.hdop > self.MAX_HDOP:
            return None

        dt = max(gps.ts - self._last_ts, 0.001)
        self._last_ts = gps.ts

        if self._origin is None:
            self._origin = (gps.lat, gps.lon)

        east, north = _latlon_to_enu(gps.lat, gps.lon, *self._origin)

        # Kalman predict
        F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
        self._x = F @ self._x
        self._P = F @ self._P @ F.T + self._Q

        # Kalman update
        H = np.array([[1,0,0,0],[0,1,0,0]])
        y = np.array([east, north]) - H @ self._x
        S = H @ self._P @ H.T + self._R
        K = self._P @ H.T @ np.linalg.inv(S)
        self._x = self._x + K @ y
        self._P = (np.eye(4) - K @ H) @ self._P

        if self._pending_sonar is None:
            return None

        sonar = self._pending_sonar
        depth = max(0.01, sonar.depth_m - self.TRANSD_DRAFT_M)
        conf  = (sonar.signal_db / 100.0) * (
            1 - 0.3 * min(1.0, gps.speed_kts / self.MAX_SPEED_KTS)
        ) * (1 - 0.2 * min(1.0, (gps.hdop - 1.0) / (self.MAX_HDOP - 1.0)))

        return Observation(
            ts          = gps.ts,
            east_m      = float(self._x[0]),
            north_m     = float(self._x[1]),
            depth_m     = depth,
            confidence  = round(max(0.0, min(1.0, conf)), 3),
            heading_deg = gps.heading_deg,
            speed_kts   = gps.speed_kts,
        )
