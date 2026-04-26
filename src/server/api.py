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
    "mesh": { "vertices": [...], "faces": [...] } | null  (every 5s)
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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Artifishial Intelligence v3.3 (cone+perf+fe)")

# ── Session state (injected by main.py) ──────────────────────────────────────

_world_state     = None   # WorldState
_replay_ctrl     = None   # ReplayController | None
_ground_truth    = None   # GroundTruth | None
_duration_s: Optional[float] = None

def set_session(world_state, replay_ctrl=None,
                ground_truth=None, duration_s=None):
    global _world_state, _replay_ctrl, _ground_truth, _duration_s
    _world_state  = world_state
    _replay_ctrl  = replay_ctrl
    _ground_truth = ground_truth
    _duration_s   = duration_s


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


def _build_update(current_ts: Optional[float] = None) -> dict:
    from rendering.pointcloud import build_pointcloud_payload, extract_boat

    if _world_state is None:
        return {"type": "map_update", "pointcloud": {}, "boat": {}}

    ts   = current_ts if current_ts is not None else _world_state.latest_ts()
    pc   = build_pointcloud_payload(_world_state, current_ts)
    boat = extract_boat(_world_state.state_at(ts) if ts is not None else _world_state)

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

    fwd = _world_state.forward_scan_at(ts) if ts is not None else None
    if fwd:
        payload["forward_scan"] = base64.b64encode(fwd).decode("ascii")

    return payload


async def _dispatch(msg: dict):
    if _replay_ctrl is None:
        return
    t = msg.get("type")
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
    await ws.send_text(json.dumps({
        "type":             "session_info",
        "duration_s":       _duration_s,
        "has_ground_truth": _ground_truth is not None,
    }))

    if _ground_truth is not None:
        await ws.send_text(json.dumps({
            "type": "ground_truth",
            "data": {
                "fish_schools": _ground_truth.fish_schools,
                "floor_grid":   _ground_truth.floor_grid,
            },
        }))

    try:
        while True:
            # Non-blocking receive — drain any inbound command
            try:
                data = await asyncio.wait_for(ws.receive_text(), timeout=0.05)
                await _dispatch(json.loads(data))
            except asyncio.TimeoutError:
                pass

            now = time.time()
            ts  = _replay_ctrl.position_s if _replay_ctrl is not None else None
            payload = _build_update(ts)

            # Include mesh every 5 seconds
            if now - _last_mesh_ts >= 0.5 and _world_state is not None:
                from rendering.mesh import build_mesh
                mesh = build_mesh(_world_state, ts)
                if mesh is not None:
                    payload["mesh"] = mesh
                _last_mesh_ts = now

            await ws.send_text(json.dumps(payload))
            await asyncio.sleep(0.1)
            if _replay_ctrl is not None:
                _replay_ctrl.advance(0.1)

    except WebSocketDisconnect:
        _manager.disconnect(ws)


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
