"""
src/server/api.py

FastAPI WebSocket server.

WebSocket /ws/state sends map_update messages every 100ms:
  {
    "type": "map_update",
    "ts": <current replay position>,
    "duration_s": <total duration or null>,
    "paused": bool,
    "pointcloud": { "x": [...], "y": [...], "depth": [...], ... },
    "boat": { "east": ..., "north": ..., "heading": ..., "speed_kts": ... },
    "mesh": { "vertices": [...], "faces": [...] } | null  (every 2s, when changed)
  }

Inbound WS commands:
  { "type": "seek",   "fraction": 0.0..1.0 }
  { "type": "pause" }
  { "type": "play"  }
  { "type": "speed",  "value": 1.0 }

REST:
  GET /health
  GET /api/state          → full current pointcloud
  GET /api/mesh           → full mesh (Delaunay)
  GET /api/ground-truth   → ground truth JSON (if available)
"""
import asyncio
import base64
import json
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import orjson
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from server.profile import FrameProfiler, profiler_from_env


def _dumps(obj) -> bytes:
    """orjson-backed JSON encoder; returns raw bytes for ws.send_bytes."""
    return orjson.dumps(obj, option=orjson.OPT_SERIALIZE_NUMPY)

app = FastAPI(title="Artifishial Intelligence v3.4 (stream+restart)")

# ── Session state (injected by main.py) ──────────────────────────────────────

_world_state     = None   # WorldState
_replay_ctrl     = None   # ReplayController | None
_stream_ctrl     = None   # StreamController | None  (live/stream restart)
_ground_truth    = None   # GroundTruth | None
_duration_s: Optional[float] = None
_inference_engine = None   # InferenceEngine | None

def set_session(world_state, replay_ctrl=None,
                ground_truth=None, duration_s=None,
                stream_ctrl=None, inference_engine=None):
    global _world_state, _replay_ctrl, _ground_truth, _duration_s, _stream_ctrl, _inference_engine
    _world_state  = world_state
    _replay_ctrl  = replay_ctrl
    _ground_truth = ground_truth
    _duration_s   = duration_s
    _stream_ctrl  = stream_ctrl
    _inference_engine = inference_engine


# ── WebSocket manager ─────────────────────────────────────────────────────────

class _Manager:
    def __init__(self):
        self._clients: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self._clients.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self._clients:
            self._clients.remove(ws)

_manager = _Manager()
_last_mesh_ts: float = 0.0

# Mesh cache — only rebuild + resend when the floor-point set has grown.
# Delaunay over 100k+ points takes hundreds of ms; this collapses cost in
# steady state to a near-free floor-count probe.
_mesh_cache: Optional[dict] = None
_mesh_cache_floor_n: int = -1


def _floor_count(view) -> int:
    """Count of floor observations in a (possibly sliced) WorldState view."""
    d = view._data
    if d is None or len(d) == 0:
        return 0
    return int((d[:, 7] > 0.5).sum())  # _IS_FLOOR == 7


def _mesh_needs_rebuild(view) -> bool:
    """Cheap probe — true iff the floor point count grew since last build."""
    return _floor_count(view) != _mesh_cache_floor_n


def _rebuild_mesh(view) -> Optional[dict]:
    """Run Delaunay (the expensive part) and update the cache."""
    global _mesh_cache, _mesh_cache_floor_n
    floor_n = _floor_count(view)
    mesh = view.to_mesh(min_points=4, as_arrays=True)
    _mesh_cache = mesh
    _mesh_cache_floor_n = floor_n
    return mesh


def _get_latest_floor_obs(world_state, ts):
    """Find last observation with ts <= ts and is_floor == 1.0."""
    d = world_state._data
    if d is None or len(d) == 0:
        return None
    # Find last floor point at or before ts
    # _IS_FLOOR == 7, _TS == 0
    mask = (d[:, 0] <= ts) & (d[:, 7] > 0.5)
    if not np.any(mask):
        return None
    idx = np.where(mask)[0][-1]
    row = d[idx]
    from types import SimpleNamespace
    return SimpleNamespace(
        east_m=row[1], north_m=row[2], depth_m=row[3],
        speed_kts=row[4], heading_deg=row[5], confidence=row[6]
    )


def _build_update(current_ts: Optional[float] = None,
                  last_floor_n: int = 0,
                  prof: Optional[FrameProfiler] = None) -> tuple[dict, int]:
    """
    Build a map_update payload and return (payload, new_last_floor_n).

    last_floor_n is the count of floor observations the client already
    has buffered; the pointcloud is built as a delta beyond that point.

    `prof` is an optional FrameProfiler — when enabled, .stage() is
    called between each logical step so per-stage timings can be
    aggregated by the caller.
    """
    from rendering.pointcloud import build_pointcloud_delta, extract_boat

    if _world_state is None:
        return ({"type": "map_update", "pointcloud": {}, "boat": {}}, 0)

    ts = current_ts if current_ts is not None else _world_state.latest_ts()

    # Slice once and reuse the view for pointcloud, boat, and mesh.
    view = _world_state.state_at(ts) if ts is not None else _world_state
    if prof: prof.stage("state_at")

    # Delta pointcloud: only NEW floor points + the (small) decayed fish set.
    pc = build_pointcloud_delta(view, current_ts=ts, last_floor_n=last_floor_n)
    new_floor_n = pc["floor_total"]
    if prof: prof.stage("pointcloud_build")

    boat = extract_boat(view)
    if prof: prof.stage("extract_boat")

    payload: dict = {
        "type":       "map_update",
        "ts":         ts,
        "duration_s": _duration_s,
        "pointcloud": pc,
        "boat":       boat,
    }
    if _replay_ctrl is not None:
        payload["replay"] = {
            "position_s": _replay_ctrl.position_s,
            "duration_s": _replay_ctrl.duration_s,
            "paused":     _replay_ctrl.paused,
            "fraction":   (_replay_ctrl.position_s / _replay_ctrl.duration_s)
                          if _replay_ctrl.duration_s else 0.0,
        }

    echo = _world_state.echo_at(ts) if ts is not None else None
    if echo:
        payload["echo"] = base64.b64encode(echo).decode("ascii")
    if prof: prof.stage("echo_lookup")

    if _inference_engine is not None:
        latest_obs = _get_latest_floor_obs(_world_state, ts)
        if latest_obs is not None:
            fwd = _world_state.forward_scan_at(ts) if ts is not None else None
            preds = _inference_engine.push(latest_obs, fwd)
            if preds is not None:
                payload["catch_predictions"] = preds
    if prof: prof.stage("inference")

    if _ground_truth is not None and ts is not None:
        payload["fish_positions"] = [
            {
                "east_m":   round(s["east_m"]  + s.get("amp_e", 0) * math.sin(s.get("freq", 0) * ts + s.get("phase", 0)), 2),
                "north_m":  round(s["north_m"] + s.get("amp_n", 0) * math.cos(s.get("freq", 0) * ts + s.get("phase", 0) + 0.5), 2),
                "depth_m":  s["depth_m"],
                "radius_m": s["radius_m"],
                "species":  s.get("species", ""),
            }
            for s in _ground_truth.fish_schools
        ]
    if prof: prof.stage("ground_truth")

    fwd = _world_state.forward_scan_at(ts) if ts is not None else None
    if fwd:
        payload["forward_scan"] = base64.b64encode(fwd).decode("ascii")
    if prof: prof.stage("fwd_scan_lookup")

    payload["_view"] = view  # internal: passed to mesh stage, stripped before send
    return payload, new_floor_n


async def _dispatch(msg: dict):
    t = msg.get("type")
    if t == "stream_restart" and _stream_ctrl is not None:
        _stream_ctrl.request_restart()
        return
    if _replay_ctrl is None:
        return
    if t == "seek":
        await _replay_ctrl.seek(float(msg.get("fraction", 0)))
    elif t == "pause":
        _replay_ctrl.pause()
    elif t == "play":
        _replay_ctrl.play()
    elif t == "speed":
        _replay_ctrl.set_speed(float(msg.get("value", 1.0)))


@app.websocket("/ws/state")
async def ws_state(ws: WebSocket):
    global _last_mesh_ts
    await _manager.connect(ws)

    # Send session info immediately
    await ws.send_bytes(_dumps({
        "type":             "session_info",
        "duration_s":       _duration_s,
        "has_ground_truth": _ground_truth is not None,
        "stream_restart":   _stream_ctrl is not None,
    }))

    if _ground_truth is not None:
        await ws.send_bytes(_dumps({
            "type": "ground_truth",
            "data": {
                "fish_schools": _ground_truth.fish_schools,
                "floor_grid":   _ground_truth.floor_grid,
            },
        }))

    # Drain inbound commands on a separate task so the send loop never
    # pays a per-iteration receive timeout (was ~50ms wasted per frame).
    async def _recv_loop():
        try:
            while True:
                data = await ws.receive_text()
                try:
                    await _dispatch(json.loads(data))
                except Exception:
                    pass
        except WebSocketDisconnect:
            pass

    recv_task = asyncio.create_task(_recv_loop())
    prof = profiler_from_env()

    try:
        last_tick    = time.monotonic()
        last_floor_n = 0   # per-connection cursor for delta pointcloud
        mesh_task: Optional[asyncio.Task] = None  # in-flight Delaunay rebuild
        while True:
            ts  = _replay_ctrl.position_s if _replay_ctrl is not None else None
            prof.start_frame(ts)

            payload, last_floor_n = _build_update(ts, last_floor_n, prof=prof)
            view    = payload.pop("_view", None)

            # Mesh: throttle to every 2s AND skip when floor-count unchanged.
            # Build in a background thread so Delaunay's ~50–150 ms doesn't
            # block this iteration; attach the result to whichever frame
            # is being built when the task finishes.
            now = time.time()
            should_consider = (view is not None
                               and now - _last_mesh_ts >= 2.0
                               and _world_state is not None
                               and mesh_task is None)
            do_rebuild = should_consider and _mesh_needs_rebuild(view)
            prof.stage("mesh_check")

            if do_rebuild:
                # Capture the current view; the worker will read view._data
                # which is a numpy slice into the world buffer. WorldState
                # is append-only so reading concurrently is safe.
                mesh_task = asyncio.create_task(
                    asyncio.to_thread(_rebuild_mesh, view))
                _last_mesh_ts = now
            elif should_consider:
                _last_mesh_ts = now

            if mesh_task is not None and mesh_task.done():
                mesh = mesh_task.result()
                mesh_task = None
                if mesh is not None:
                    payload["mesh"] = mesh
                    prof.note_mesh_built()
                prof.stage("mesh_build")

            encoded = _dumps(payload)
            prof.stage("json_encode")

            await ws.send_bytes(encoded)
            prof.stage("ws_send")

            await asyncio.sleep(0.1)
            prof.stage("sleep")

            prof.end_frame(payload_bytes=len(encoded))

            # Advance replay clock by ACTUAL elapsed wall-clock time,
            # not a hardcoded 0.1s. Without this the displayed time falls
            # further behind real time as iteration cost grows.
            new_tick = time.monotonic()
            if _replay_ctrl is not None:
                _replay_ctrl.advance(new_tick - last_tick)
            last_tick = new_tick

    except WebSocketDisconnect:
        _manager.disconnect(ws)
    finally:
        recv_task.cancel()


# ── REST endpoints ────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "ts": time.time()}


@app.get("/api/state")
def api_state():
    if _world_state is None:
        return JSONResponse({"error": "no session"}, status_code=503)
    ts = _replay_ctrl.position_s if _replay_ctrl is not None else None
    from rendering.pointcloud import build_pointcloud_payload
    return build_pointcloud_payload(_world_state, ts)


@app.get("/api/mesh")
def api_mesh():
    if _world_state is None:
        return JSONResponse({"error": "no session"}, status_code=503)
    from rendering.mesh import build_mesh
    ts   = _replay_ctrl.position_s if _replay_ctrl is not None else None
    mesh = build_mesh(_world_state, ts)
    return mesh if mesh is not None else JSONResponse(
        {"error": "insufficient data"}, status_code=204)


@app.get("/api/ground-truth")
def api_ground_truth():
    if _ground_truth is None:
        return JSONResponse({"error": "no ground truth in this mode"},
                            status_code=404)
    return {
        "fish_schools": _ground_truth.fish_schools,
        "floor_grid":   _ground_truth.floor_grid,
    }


# ── Static frontend ───────────────────────────────────────────────────────────

_frontend = Path(__file__).parent / "frontend"
app.mount("/static", StaticFiles(directory=str(_frontend)), name="static")

@app.get("/")
def index():
    return FileResponse(str(_frontend / "index.html"))
