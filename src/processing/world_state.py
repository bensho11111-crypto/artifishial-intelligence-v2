"""
src/processing/world_state.py

WorldState — the core data structure of the system.

Stores all Observations as a sorted numpy array.
Supports O(log N) time queries: state_at(ts) returns only the
observations whose timestamp <= ts.

This replaces the 480 MB voxel grid from v1.  A full 2-minute fishing
session produces ~120 observations, which fit in ~6 KB.
"""
from __future__ import annotations
import numpy as np
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ticks.models import Observation

# Column indices
_TS      = 0
_EAST    = 1
_NORTH   = 2
_DEPTH   = 3
_CONF    = 4
_HDG     = 5
_SPD     = 6
_IS_FLOOR = 7   # 1.0 = bottom return, 0.0 = fish/mid-water echo
_NCOLS   = 8


class WorldState:
    """
    Append-only sorted array of Observations.

    All public methods are O(log N) for time queries and O(1) for slices.
    Safe to call from any async context — no internal locks needed
    because asyncio is single-threaded.
    """

    def __init__(self):
        self._data: Optional[np.ndarray] = None            # shape (N, 8), float32
        self._echoes: list[tuple[float, bytes]] = []        # (ts, echo) floor obs
        self._fwd_scans: list[tuple[float, bytes]] = []     # (ts, scan) floor obs

    # ── mutation ─────────────────────────────────────────────────────────────

    def add(self, obs: "Observation") -> None:
        row = np.array([[obs.ts, obs.east_m, obs.north_m, obs.depth_m,
                         obs.confidence, obs.heading_deg, obs.speed_kts,
                         1.0 if obs.is_floor else 0.0]],
                       dtype=np.float32)
        self._data = row if self._data is None else np.vstack((self._data, row))
        if obs.is_floor and obs.echo:
            self._echoes.append((obs.ts, obs.echo))
        if obs.is_floor and obs.forward_scan:
            self._fwd_scans.append((obs.ts, obs.forward_scan))

    def reset(self) -> None:
        self._data = None
        self._echoes = []
        self._fwd_scans = []

    # ── queries ───────────────────────────────────────────────────────────────

    def state_at(self, ts: float) -> "WorldState":
        """Return a new WorldState containing only observations up to ts."""
        ws = WorldState()
        if self._data is not None and len(self._data):
            idx = int(np.searchsorted(self._data[:, _TS], ts, side="right"))
            if idx > 0:
                ws._data = self._data[:idx]
        return ws

    def echo_at(self, ts: float) -> Optional[bytes]:
        """Return the most recent floor echo at or before ts."""
        if not self._echoes:
            return None
        import bisect
        idx = bisect.bisect_right(self._echoes, (ts, b'\xff' * 512)) - 1
        return self._echoes[idx][1] if idx >= 0 else None

    def forward_scan_at(self, ts: float) -> Optional[bytes]:
        """Return the most recent forward scan frame at or before ts."""
        if not self._fwd_scans:
            return None
        import bisect
        idx = bisect.bisect_right(self._fwd_scans, (ts, b'\xff' * 512)) - 1
        return self._fwd_scans[idx][1] if idx >= 0 else None

    def latest_ts(self) -> Optional[float]:
        if self._data is None or len(self._data) == 0:
            return None
        return float(self._data[-1, _TS])

    def __len__(self) -> int:
        return 0 if self._data is None else len(self._data)

    # ── export ────────────────────────────────────────────────────────────────

    # Fish observations decay exponentially — half-life of 12 seconds so that
    # a school that has moved away fades within ~40s.
    _FISH_DECAY_RATE = 0.0578   # ln(2) / 12
    _FISH_MIN_CONF   = 0.02     # cull below this after decay

    def to_pointcloud(self, floor_only: bool = False,
                      current_ts: Optional[float] = None) -> dict:
        """
        Export as a dict of lists suitable for JSON serialisation.
        floor_only=True: exclude mid-water fish echo returns (is_floor==0).
        current_ts: when given, apply exponential confidence decay to fish
                    observations and cull those below _FISH_MIN_CONF.
        """
        empty = {"x": [], "y": [], "depth": [], "confidence": [],
                 "ts": [], "heading": [], "speed_kts": [], "is_floor": []}
        if self._data is None or len(self._data) == 0:
            return empty
        d = self._data
        if floor_only:
            d = d[d[:, _IS_FLOOR] > 0.5]
        if len(d) == 0:
            return empty

        conf = d[:, _CONF].copy()
        if current_ts is not None:
            is_fish = d[:, _IS_FLOOR] < 0.5
            if is_fish.any():
                ages  = np.maximum(0.0, current_ts - d[:, _TS])
                decay = np.where(is_fish,
                                 np.exp(-self._FISH_DECAY_RATE * ages), 1.0)
                conf *= decay
                keep = (~is_fish) | (conf >= self._FISH_MIN_CONF)
                d    = d[keep]
                conf = conf[keep]

        return {
            "x":          d[:, _EAST].tolist(),
            "y":          d[:, _NORTH].tolist(),
            "depth":      d[:, _DEPTH].tolist(),
            "confidence": conf.tolist(),
            "ts":         d[:, _TS].tolist(),
            "heading":    d[:, _HDG].tolist(),
            "speed_kts":  d[:, _SPD].tolist(),
            "is_floor":   (d[:, _IS_FLOOR] > 0.5).tolist(),
        }

    def to_mesh(self, min_points: int = 4) -> Optional[dict]:
        """
        Build a Delaunay triangulation mesh from the observation positions.
        Returns None if there are too few points.

        Much cheaper than marching cubes on a voxel grid — for typical
        sessions (<10 000 points) this runs in <50 ms.
        """
        if self._data is None or len(self._data) < min_points:
            return None
        try:
            from scipy.spatial import Delaunay  # type: ignore
        except ImportError:
            return None

        # Only triangulate bottom returns — fish echoes create duplicate 2D
        # inputs that corrupt Delaunay and pull the mesh to shallow depths.
        floor_mask = self._data[:, _IS_FLOOR] > 0.5
        data = self._data[floor_mask]
        if len(data) < min_points:
            return None

        pts    = data[:, [_EAST, _NORTH]]
        depths = data[:, _DEPTH]

        tri   = Delaunay(pts)
        verts = np.column_stack([pts, -depths])   # Z = -depth (Z-up scene)

        return {
            "vertices": verts.tolist(),
            "faces":    tri.simplices.tolist(),
            "depth":    depths.tolist(),
        }

    def to_contour_grid(self, cell_m: float = 2.0) -> Optional[dict]:
        """
        Rasterise observations onto a regular grid and return depth values
        suitable for contour rendering.  Uses nearest-neighbour interpolation.
        """
        if self._data is None or len(self._data) < 4:
            return None

        x = self._data[:, _EAST]
        y = self._data[:, _NORTH]
        d = self._data[:, _DEPTH]

        x0, x1 = float(x.min()), float(x.max())
        y0, y1 = float(y.min()), float(y.max())

        if (x1 - x0) < cell_m or (y1 - y0) < cell_m:
            return None

        cols = max(2, int((x1 - x0) / cell_m) + 1)
        rows = max(2, int((y1 - y0) / cell_m) + 1)
        grid = np.full((rows, cols), np.nan, dtype=np.float32)

        xi = ((x - x0) / cell_m).astype(int).clip(0, cols - 1)
        yi = ((y - y0) / cell_m).astype(int).clip(0, rows - 1)
        grid[yi, xi] = d

        return {
            "origin_east_m":  x0,
            "origin_north_m": y0,
            "cell_m":         cell_m,
            "rows":           rows,
            "cols":           cols,
            "depth":          np.where(np.isnan(grid), None, grid).tolist(),
        }
