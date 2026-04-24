"""
src/rendering/mesh.py

On-demand mesh and contour generation from WorldState.
Uses Delaunay triangulation — much lighter than marching cubes on a
voxel grid.  For a typical 2-minute session (<200 points) this runs in <5ms.
"""
from __future__ import annotations
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from processing.world_state import WorldState


def build_mesh(state: "WorldState",
               current_ts: Optional[float] = None,
               min_points: int = 4) -> Optional[dict]:
    """Build a Delaunay mesh. Returns None if too few observations."""
    view = state.state_at(current_ts) if current_ts is not None else state
    return view.to_mesh(min_points=min_points)


def build_contour(state: "WorldState",
                  current_ts: Optional[float] = None,
                  cell_m: float = 2.0) -> Optional[dict]:
    """Build a rasterised contour grid."""
    view = state.state_at(current_ts) if current_ts is not None else state
    return view.to_contour_grid(cell_m=cell_m)
