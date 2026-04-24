"""
src/rendering/pointcloud.py

Converts WorldState → point cloud JSON for the AR frontend.
"""
from __future__ import annotations
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from processing.world_state import WorldState


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
    return view.to_pointcloud()


def extract_boat(state: "WorldState") -> dict:
    """Return the most recent boat position from the state."""
    pc = state.to_pointcloud()
    if not pc["x"]:
        return {}
    return {
        "east":      round(pc["x"][-1], 2),
        "north":     round(pc["y"][-1], 2),
        "heading":   round(pc["heading"][-1], 1),
        "speed_kts": round(pc["speed_kts"][-1], 1),
    }
