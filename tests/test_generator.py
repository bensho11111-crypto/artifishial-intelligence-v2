import math
import pytest
from synthetic.generator import FishSchool, generate, SONAR_HZ, GPS_HZ, ECHO_SIZE
from synthetic.forward_scan import N_BEAMS, N_RANGE, N_AZIMUTH


# ── FishSchool movement ───────────────────────────────────────────────────────

def test_fish_school_stationary_when_no_motion():
    s = FishSchool(10.0, 20.0, 5.0, 8.0, 0.9, "bass")
    for t in [0, 10, 60, 120]:
        s2 = s.at(t)
        assert s2.east_m  == pytest.approx(10.0)
        assert s2.north_m == pytest.approx(20.0)


def test_fish_school_moves_with_params():
    s = FishSchool(0.0, 0.0, 5.0, 8.0, 0.9, "bass",
                   amp_e=10.0, amp_n=8.0, freq=0.1, phase=0.0)
    s30 = s.at(30.0)
    assert s30.east_m  != pytest.approx(0.0, abs=0.1)


def test_fish_school_movement_periodic():
    s = FishSchool(0.0, 0.0, 5.0, 8.0, 0.9, "bass",
                   amp_e=10.0, amp_n=8.0, freq=0.1, phase=0.0)
    period = 2 * math.pi / 0.1
    s0 = s.at(0.0)
    s1 = s.at(period)
    assert s1.east_m  == pytest.approx(s0.east_m,  abs=0.01)
    assert s1.north_m == pytest.approx(s0.north_m, abs=0.01)


def test_fish_school_at_preserves_static_fields():
    s = FishSchool(5.0, 5.0, 15.0, 10.0, 0.7, "trout",
                   amp_e=5.0, amp_n=5.0, freq=0.05, phase=0.0)
    s2 = s.at(30.0)
    assert s2.depth_m  == s.depth_m
    assert s2.radius_m == s.radius_m
    assert s2.density  == s.density
    assert s2.species  == s.species


# ── Session generation ────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def short_session():
    return generate(duration_s=10.0, seed=1)


def test_generate_tick_count(short_session):
    expected = int(10.0 * SONAR_HZ)
    assert len(short_session.ticks) == expected


def test_generate_gps_rate(short_session):
    gps_ticks = [t for t in short_session.ticks if t.gps is not None]
    expected  = int(10.0 * GPS_HZ)
    assert len(gps_ticks) == expected


def test_generate_all_sonar_have_echo(short_session):
    for tick in short_session.ticks:
        if tick.sonar is not None:
            assert tick.sonar.echo is not None
            assert len(tick.sonar.echo) == ECHO_SIZE


def test_generate_gps_ticks_have_forward_scan(short_session):
    for tick in short_session.ticks:
        if tick.gps is not None and tick.sonar is not None:
            assert tick.sonar.forward_scan is not None
            assert len(tick.sonar.forward_scan) == N_AZIMUTH * N_BEAMS * N_RANGE


def test_generate_non_gps_ticks_no_forward_scan(short_session):
    for tick in short_session.ticks:
        if tick.gps is None and tick.sonar is not None:
            assert tick.sonar.forward_scan is None


def test_generate_timestamps_monotonic(short_session):
    ts = [t.ts for t in short_session.ticks]
    assert ts == sorted(ts)


def test_generate_deterministic():
    s1 = generate(duration_s=5.0, seed=7)
    s2 = generate(duration_s=5.0, seed=7)
    assert s1.ticks[0].sonar.echo == s2.ticks[0].sonar.echo


def test_generate_different_seeds_differ():
    s1 = generate(duration_s=5.0, seed=1)
    s2 = generate(duration_s=5.0, seed=2)
    assert s1.ticks[0].sonar.echo != s2.ticks[0].sonar.echo


def test_generate_forward_scans_change_over_time():
    """Forward scans at different times must differ as fish schools move."""
    session = generate(duration_s=60.0, seed=42)
    gps = [t for t in session.ticks if t.gps and t.sonar and t.sonar.forward_scan]
    assert len(gps) >= 2
    assert gps[0].sonar.forward_scan != gps[-1].sonar.forward_scan


def test_ground_truth_has_movement_params():
    session = generate(duration_s=5.0, seed=42)
    gt = session.to_ground_truth()
    for school in gt["fish_schools"]:
        for key in ("amp_e", "amp_n", "freq", "phase"):
            assert key in school, f"Missing movement key '{key}' in ground truth school"


def test_ground_truth_has_four_schools():
    session = generate(duration_s=5.0, seed=42)
    gt = session.to_ground_truth()
    assert len(gt["fish_schools"]) == 4


def test_ground_truth_floor_grid_present():
    session = generate(duration_s=5.0, seed=42)
    gt = session.to_ground_truth()
    assert "floor_grid" in gt
    assert gt["floor_grid"] is not None
