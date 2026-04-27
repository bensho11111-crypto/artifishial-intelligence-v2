"""
src/rendering/pointcloud.py

Converts WorldState → point cloud JSON for the AR frontend.
"""
from __future__ import annotations
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from processing.world_state import WorldState

# Mirrors WorldState column indices.
_TS, _EAST, _NORTH, _DEPTH, _CONF, _HDG, _SPD, _IS_FLOOR = range(8)

# Fish observation decay (must mirror WorldState constants).
_FISH_DECAY_RATE = 0.0578     # ln(2) / 12s half-life
_FISH_MIN_CONF   = 0.02


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


def build_pointcloud_arrays(view: "WorldState",
                             current_ts: Optional[float] = None) -> dict:
    """
    Fast path for the WebSocket loop: return contiguous numpy arrays in
    their native dtype (float32) so orjson can serialise them directly.

    This skips tolist() entirely — saving the float32→float64 promotion
    that ballooned JSON byte size (long-form double formatting), and
    saves the per-element Python-object hop. Roughly halves payload bytes
    AND encode time vs the list-based path.

    Layout matches build_pointcloud_payload but only the four numeric
    arrays the frontend actually consumes (plus bool is_floor).
    """
    empty = {"x": np.empty(0, dtype=np.float32),
             "y": np.empty(0, dtype=np.float32),
             "depth": np.empty(0, dtype=np.float32),
             "confidence": np.empty(0, dtype=np.float32),
             "is_floor": []}

    d = view._data
    if d is None or len(d) == 0:
        return empty

    # Apply fish-confidence decay, mirroring WorldState.to_pointcloud.
    # Fast path: when no fish observations exist (or current_ts not given),
    # skip the entire decay/cull pipeline — it allocates four N-sized
    # arrays whose result for floor-only data is just conf == d[:, _CONF].
    if current_ts is None:
        conf = d[:, _CONF]
    else:
        is_fish = d[:, _IS_FLOOR] < 0.5
        if not is_fish.any():
            conf = d[:, _CONF]
        else:
            ages  = np.maximum(0.0, current_ts - d[:, _TS])
            decay = np.where(is_fish,
                             np.exp(-_FISH_DECAY_RATE * ages), 1.0)
            conf  = d[:, _CONF] * decay
            keep  = (~is_fish) | (conf >= _FISH_MIN_CONF)
            d     = d[keep]
            conf  = conf[keep]

    if len(d) == 0:
        return empty

    # Contiguous copies — required by orjson's numpy fast path AND
    # cheaper to serialise than non-contiguous strided views.
    # is_floor as int8 (0/1) instead of bool list: ~600 KiB → ~240 KiB
    # over the wire and a faster encode. Frontend uses truthy semantics
    # so 0/1 is interchangeable with false/true.
    return {
        "x":          np.ascontiguousarray(d[:, _EAST]),
        "y":          np.ascontiguousarray(d[:, _NORTH]),
        "depth":      np.ascontiguousarray(d[:, _DEPTH]),
        "confidence": np.ascontiguousarray(conf),
        "is_floor":   np.ascontiguousarray(d[:, _IS_FLOOR] > 0.5).astype(np.int8),
    }


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
