import math
import pytest
from processing.world_state import WorldState
from ticks.models import Observation


# ── Helpers ───────────────────────────────────────────────────────────────────

def _obs(ts, east=0.0, north=0.0, depth=15.0, conf=0.8,
         is_floor=True, echo=None, fwd=None):
    return Observation(ts=ts, east_m=east, north_m=north, depth_m=depth,
                       confidence=conf, heading_deg=0.0, speed_kts=3.5,
                       is_floor=is_floor, echo=echo, forward_scan=fwd)


# ── Basic mutation ────────────────────────────────────────────────────────────

def test_empty_state_len():
    assert len(WorldState()) == 0


def test_empty_pointcloud():
    pc = WorldState().to_pointcloud()
    assert pc["x"] == []


def test_add_increases_len():
    ws = WorldState()
    ws.add(_obs(1.0))
    ws.add(_obs(2.0))
    assert len(ws) == 2


def test_reset_clears_all():
    ws = WorldState()
    ws.add(_obs(1.0, echo=b"\xaa" * 512))
    ws.reset()
    assert len(ws) == 0
    assert ws.echo_at(2.0) is None
    assert ws.forward_scan_at(2.0) is None


# ── Time slicing ──────────────────────────────────────────────────────────────

def test_state_at_returns_subset():
    ws = WorldState()
    for i in range(5):
        ws.add(_obs(float(i)))
    assert len(ws.state_at(2.5)) == 3


def test_state_at_excludes_future():
    ws = WorldState()
    ws.add(_obs(10.0))
    assert len(ws.state_at(5.0)) == 0


def test_state_at_includes_exact_ts():
    ws = WorldState()
    ws.add(_obs(5.0))
    assert len(ws.state_at(5.0)) == 1


# ── Echo / forward scan lookup ────────────────────────────────────────────────

def test_echo_at_returns_most_recent():
    ws = WorldState()
    ws.add(_obs(1.0, echo=b"\xaa" * 512))
    ws.add(_obs(2.0, echo=b"\xbb" * 512))
    assert ws.echo_at(1.5) == b"\xaa" * 512
    assert ws.echo_at(2.0) == b"\xbb" * 512


def test_echo_at_before_first_returns_none():
    ws = WorldState()
    ws.add(_obs(1.0, echo=b"\xaa" * 512))
    assert ws.echo_at(0.5) is None


def test_echo_only_stored_for_floor_obs():
    ws = WorldState()
    ws.add(_obs(1.0, is_floor=False, echo=b"\xaa" * 512))
    assert ws.echo_at(2.0) is None


def test_forward_scan_at_returns_most_recent():
    ws = WorldState()
    fwd1, fwd2 = bytes(7680), bytes(range(256)) * 30
    ws.add(_obs(1.0, fwd=fwd1))
    ws.add(_obs(2.0, fwd=fwd2))
    assert ws.forward_scan_at(1.5) == fwd1
    assert ws.forward_scan_at(2.5) == fwd2


def test_forward_scan_at_before_first_returns_none():
    ws = WorldState()
    ws.add(_obs(1.0, fwd=bytes(7680)))
    assert ws.forward_scan_at(0.5) is None


# ── Confidence decay ──────────────────────────────────────────────────────────

def test_floor_obs_confidence_not_decayed():
    """Floor observations must be unaffected by time-based decay."""
    ws = WorldState()
    ws.add(_obs(0.0, conf=0.9, is_floor=True))
    pc = ws.to_pointcloud(current_ts=1000.0)
    assert len(pc["x"]) == 1
    assert pc["confidence"][0] == pytest.approx(0.9, abs=0.01)


def test_fish_obs_confidence_decays():
    """Fish observation confidence must decrease after ~one half-life (12s)."""
    ws = WorldState()
    ws.add(_obs(0.0, conf=0.65, is_floor=False))
    pc = ws.to_pointcloud(current_ts=12.0)
    # After 12s (one half-life) effective conf should be roughly halved
    if len(pc["x"]) > 0:
        assert pc["confidence"][0] < 0.40


def test_fish_obs_culled_when_ancient():
    """Fish observations too old to matter (< 2% conf) must be removed."""
    ws = WorldState()
    ws.add(_obs(0.0, conf=0.65, is_floor=False))
    pc = ws.to_pointcloud(current_ts=300.0)
    assert len(pc["x"]) == 0


def test_floor_obs_never_culled():
    ws = WorldState()
    ws.add(_obs(0.0, conf=0.8, is_floor=True))
    pc = ws.to_pointcloud(current_ts=300.0)
    assert len(pc["x"]) == 1


def test_fresh_fish_obs_not_culled():
    ws = WorldState()
    ws.add(_obs(0.0, conf=0.65, is_floor=False))
    pc = ws.to_pointcloud(current_ts=1.0)  # 1s old
    assert len(pc["x"]) == 1


def test_no_current_ts_no_decay():
    """Without current_ts, fish obs are returned at original confidence."""
    ws = WorldState()
    ws.add(_obs(0.0, conf=0.5, is_floor=False))
    pc = ws.to_pointcloud()
    assert len(pc["x"]) == 1
    assert pc["confidence"][0] == pytest.approx(0.5, abs=0.01)


# ── Point cloud export ────────────────────────────────────────────────────────

def test_pointcloud_is_floor_flags():
    ws = WorldState()
    ws.add(_obs(1.0, is_floor=True))
    ws.add(_obs(2.0, is_floor=False))
    pc = ws.to_pointcloud(current_ts=3.0)
    assert True  in pc["is_floor"]


def test_pointcloud_floor_only_flag():
    ws = WorldState()
    ws.add(_obs(1.0, is_floor=True))
    ws.add(_obs(2.0, is_floor=False))
    pc = ws.to_pointcloud(floor_only=True)
    assert all(pc["is_floor"])


def test_pointcloud_lengths_consistent():
    ws = WorldState()
    for i in range(5):
        ws.add(_obs(float(i), east=float(i)))
    pc = ws.to_pointcloud()
    n = len(pc["x"])
    for key in ("y", "depth", "confidence", "ts", "heading", "speed_kts", "is_floor"):
        assert len(pc[key]) == n


# ── Mesh ──────────────────────────────────────────────────────────────────────

def test_mesh_needs_min_points():
    ws = WorldState()
    for i in range(3):
        ws.add(_obs(float(i), east=float(i)))
    assert ws.to_mesh(min_points=4) is None


def test_mesh_valid_triangulation():
    ws = WorldState()
    positions = [(0, 0), (10, 0), (0, 10), (10, 10), (5, 5), (20, 5)]
    for i, (e, n) in enumerate(positions):
        ws.add(_obs(float(i), east=float(e), north=float(n)))
    mesh = ws.to_mesh(min_points=4)
    assert mesh is not None
    assert len(mesh["vertices"]) == len(positions)
    assert len(mesh["faces"]) > 0
    # Each face references valid vertex indices
    n_verts = len(mesh["vertices"])
    for face in mesh["faces"]:
        assert all(0 <= idx < n_verts for idx in face)


def test_mesh_excludes_fish_obs():
    """Mesh must only triangulate floor observations."""
    ws = WorldState()
    # Add floor observations at known positions
    for e, n in [(0, 0), (10, 0), (0, 10), (10, 10)]:
        ws.add(_obs(0.0, east=float(e), north=float(n), is_floor=True))
    # Add a fish observation at a different position
    ws.add(_obs(1.0, east=99.0, north=99.0, is_floor=False))
    mesh = ws.to_mesh()
    verts = mesh["vertices"]
    # No vertex should be at the fish echo position
    assert all(not (abs(v[0] - 99.0) < 0.5 and abs(v[1] - 99.0) < 0.5)
               for v in verts)
