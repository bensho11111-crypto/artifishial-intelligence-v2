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


def build_pointcloud_delta(view: "WorldState",
                            current_ts: Optional[float],
                            last_floor_n: int) -> dict:
    """
    Delta pointcloud for the WebSocket loop.

    Floor observations are append-only (sorted by ts); we send only the
    rows whose floor-rank index >= last_floor_n. The client maintains a
    cumulative floor buffer and appends to it.

    Fish observations are decayed/culled and rebuilt each frame, so we
    send the full current fish set.

    Returns:
      {
        "reset":        bool,    # client should clear its floor buffer first
        "floor_offset": int,     # buffer index where these new points go
        "floor_total":  int,     # client buffer size after appending
        "floor": {x, y, depth, confidence},   # contiguous float32 ndarrays
        "fish":  {x, y, depth, confidence},   # contiguous float32 ndarrays
      }

    All floor arrays carry confidence so the frontend can preserve the
    forward-scan cyan tint (conf <= 0.45 branch).
    """
    empty_f32 = np.empty(0, dtype=np.float32)
    empty_floor = {"x": empty_f32, "y": empty_f32,
                   "depth": empty_f32, "confidence": empty_f32}
    empty_fish  = {"x": empty_f32, "y": empty_f32,
                   "depth": empty_f32, "confidence": empty_f32}

    d = view._data
    if d is None or len(d) == 0:
        return {"reset": last_floor_n != 0, "floor_offset": 0,
                "floor_total": 0, "floor": empty_floor, "fish": empty_fish}

    is_floor_mask = d[:, _IS_FLOOR] > 0.5
    floor_rows = d[is_floor_mask]
    fish_rows  = d[~is_floor_mask]
    cur_floor_n = len(floor_rows)

    # Backward seek (or first frame after one) — client has more floor
    # rows than the current view; reset and resend everything.
    reset = cur_floor_n < last_floor_n
    floor_start = 0 if reset else last_floor_n

    floor_new = floor_rows[floor_start:]
    floor_payload = {
        "x":          np.ascontiguousarray(floor_new[:, _EAST]),
        "y":          np.ascontiguousarray(floor_new[:, _NORTH]),
        "depth":      np.ascontiguousarray(floor_new[:, _DEPTH]),
        "confidence": np.ascontiguousarray(floor_new[:, _CONF]),
    }

    # Fish: apply exponential decay + cull, mirroring WorldState.to_pointcloud.
    if len(fish_rows) and current_ts is not None:
        ages  = np.maximum(0.0, current_ts - fish_rows[:, _TS])
        conf  = fish_rows[:, _CONF] * np.exp(-_FISH_DECAY_RATE * ages)
        keep  = conf >= _FISH_MIN_CONF
        fish_rows = fish_rows[keep]
        fish_conf = conf[keep]
    elif len(fish_rows):
        fish_conf = fish_rows[:, _CONF]
    else:
        fish_conf = empty_f32

    # Fish decimation: forward-scan pings produce many returns from the
    # same school, peaking at ~27 K observations mid-replay. Quantise to
    # a 0.5 m 3D grid and keep the most recent per cell — visually the
    # school is unchanged, but per-frame JSON drops from ~1 MiB to ~650 KiB.
    if len(fish_rows) > 0:
        cell = 0.5
        inv  = 1.0 / cell
        ei = np.floor(fish_rows[:, _EAST]  * inv).astype(np.int64) & 0xFFFFFF
        ni = np.floor(fish_rows[:, _NORTH] * inv).astype(np.int64) & 0xFFFFFF
        di = np.floor(fish_rows[:, _DEPTH] * inv).astype(np.int64) & 0xFFFF
        key = ei | (ni << 24) | (di << 48)
        _, last_in_reversed = np.unique(key[::-1], return_index=True)
        keep_idx = (len(key) - 1) - last_in_reversed
        keep_idx.sort()
        fish_rows = fish_rows[keep_idx]
        fish_conf = fish_conf[keep_idx]

    if len(fish_rows):
        fish_payload = {
            "x":          np.ascontiguousarray(fish_rows[:, _EAST]),
            "y":          np.ascontiguousarray(fish_rows[:, _NORTH]),
            "depth":      np.ascontiguousarray(fish_rows[:, _DEPTH]),
            "confidence": np.ascontiguousarray(fish_conf),
        }
    else:
        fish_payload = empty_fish

    return {
        "reset":        reset,
        "floor_offset": floor_start,
        "floor_total":  cur_floor_n,
        "floor":        floor_payload,
        "fish":         fish_payload,
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
