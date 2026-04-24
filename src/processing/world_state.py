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
_TS    = 0
_EAST  = 1
_NORTH = 2
_DEPTH = 3
_CONF  = 4
_HDG   = 5
_SPD   = 6
_NCOLS = 7


class WorldState:
    """
    Append-only sorted array of Observations.

    All public methods are O(log N) for time queries and O(1) for slices.
    Safe to call from any async context — no internal locks needed
    because asyncio is single-threaded.
    """

    def __init__(self):
        self._data: Optional[np.ndarray] = None   # shape (N, 7), float32

    # ── mutation ─────────────────────────────────────────────────────────────

    def add(self, obs: "Observation") -> None:
        row = np.array([[obs.ts, obs.east_m, obs.north_m, obs.depth_m,
                         obs.confidence, obs.heading_deg, obs.speed_kts]],
                       dtype=np.float32)
        self._data = row if self._data is None else np.vstack((self._data, row))

    def reset(self) -> None:
        self._data = None

    # ── queries ───────────────────────────────────────────────────────────────

    def state_at(self, ts: float) -> "WorldState":
        """Return a new WorldState containing only observations up to ts."""
        ws = WorldState()
        if self._data is not None and len(self._data):
            idx = int(np.searchsorted(self._data[:, _TS], ts, side="right"))
            if idx > 0:
                ws._data = self._data[:idx]
        return ws

    def latest_ts(self) -> Optional[float]:
        if self._data is None or len(self._data) == 0:
            return None
        return float(self._data[-1, _TS])

    def __len__(self) -> int:
        return 0 if self._data is None else len(self._data)

    # ── export ────────────────────────────────────────────────────────────────

    def to_pointcloud(self) -> dict:
        """
        Export as a dict of lists suitable for JSON serialisation.
        All arrays are parallel (same length).
        """
        empty = {"x": [], "y": [], "depth": [], "confidence": [],
                 "ts": [], "heading": [], "speed_kts": []}
        if self._data is None or len(self._data) == 0:
            return empty
        d = self._data
        return {
            "x":          d[:, _EAST].tolist(),
            "y":          d[:, _NORTH].tolist(),
            "depth":      d[:, _DEPTH].tolist(),
            "confidence": d[:, _CONF].tolist(),
            "ts":         d[:, _TS].tolist(),
            "heading":    d[:, _HDG].tolist(),
            "speed_kts":  d[:, _SPD].tolist(),
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

        pts = self._data[:, [_EAST, _NORTH]]
        depths = self._data[:, _DEPTH]

        tri = Delaunay(pts)
        verts = np.column_stack([pts, -depths])   # Z = -depth (down)

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
