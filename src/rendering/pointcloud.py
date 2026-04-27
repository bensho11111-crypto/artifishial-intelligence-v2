"""
src/rendering/pointcloud.py

Converts WorldState → point cloud JSON for the AR frontend.
"""
from __future__ import annotations
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from processing.world_state import WorldState

# Mirrors WorldState column indices.
_TS, _EAST, _NORTH, _DEPTH, _CONF, _HDG, _SPD, _IS_FLOOR = range(8)


def build_pointcloud_payload(state: "WorldState",
                              current_ts: Optional[float] = None) -> dict:
    """
    Build the pointcloud dict for a map_update WebSocket message.
    If current_ts is given, only include observations up to that time.
    """
    if current_ts is not None:
        view = state.state_at(current_ts)
    else:
        view = state
    return view.to_pointcloud(floor_only=False, current_ts=current_ts)


def extract_boat(state: "WorldState") -> dict:
    """Return the most recent boat position from the state.

    Fast path: read the last floor row directly from the numpy buffer
    rather than materialising a full floor-only pointcloud.
    """
    data = state._data
    if data is None or len(data) == 0:
        return {}

    floor_mask = data[:, _IS_FLOOR] > 0.5
    if not floor_mask.any():
        return {}

    # nonzero()[0] gives indices of True; take the last one.
    last_idx = int(floor_mask.nonzero()[0][-1])
    row = data[last_idx]
    return {
        "east":      round(float(row[_EAST]),  2),
        "north":     round(float(row[_NORTH]), 2),
        "heading":   round(float(row[_HDG]),   1),
        "speed_kts": round(float(row[_SPD]),   1),
    }
