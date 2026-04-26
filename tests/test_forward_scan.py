import math
import pytest
from synthetic.forward_scan import (
    generate, N_BEAMS, N_RANGE, MAX_RANGE_M,
    BEAM_MIN_DEG, BEAM_MAX_DEG,
)
from synthetic.floor import FloorModel
from synthetic.generator import FishSchool


@pytest.fixture(scope="module")
def floor():
    return FloorModel(seed=42)


def test_output_length(floor):
    scan = generate(0.0, 0.0, 0.0, floor, [])
    assert len(scan) == N_BEAMS * N_RANGE


def test_output_is_bytes(floor):
    scan = generate(0.0, 0.0, 0.0, floor, [])
    assert isinstance(scan, bytes)


def test_all_values_in_byte_range(floor):
    scan = generate(0.0, 0.0, 0.0, floor, [])
    assert all(0 <= b <= 255 for b in scan)


def test_floor_return_in_expected_range(floor):
    """
    For steep beams (>45°) the floor return must appear at a range bin
    consistent with the floor depth at the boat position.
    """
    from processing.fusion import _FWD_FLOOR_THRESH
    scan   = generate(0.0, 0.0, 0.0, floor, [])
    step_m = MAX_RANGE_M / N_RANGE

    for b in range(N_BEAMS):
        theta_deg = BEAM_MIN_DEG + b * (BEAM_MAX_DEG - BEAM_MIN_DEG) / (N_BEAMS - 1)
        if theta_deg < 45:
            continue  # shallow beams may not hit floor within scan range
        theta     = math.radians(theta_deg)
        sin_t     = math.sin(theta)
        floor_d   = floor.depth_at(0.0, 0.0)
        exp_range = floor_d / sin_t
        if exp_range > MAX_RANGE_M:
            continue

        exp_ri = int(exp_range / step_m)
        base   = b * N_RANGE
        # Check a window of ±5 bins around expected position for a strong return
        window = range(max(0, exp_ri - 5), min(N_RANGE, exp_ri + 6))
        has_floor = any(scan[base + ri] >= _FWD_FLOOR_THRESH for ri in window)
        assert has_floor, (
            f"Beam {b} ({theta_deg:.0f}°): no floor return near expected ri={exp_ri} "
            f"(floor_d={floor_d:.1f}m, exp_range={exp_range:.1f}m)"
        )


def test_fish_school_increases_amplitude(floor):
    """A fish school within beam range must raise amplitude above background."""
    school    = FishSchool(5.0, 0.0, 4.0, 8.0, 0.9, "bass")
    scan_base = generate(0.0, 0.0, 0.0, floor, [])
    scan_fish = generate(0.0, 0.0, 0.0, floor, [school])
    # At least some bins must have higher amplitude with the school present
    increased = sum(1 for a, b in zip(scan_fish, scan_base) if a > b)
    assert increased > 0


def test_floor_blocks_further_returns(floor):
    """
    Once the floor is hit for a beam, no returns should appear at deeper range.
    Verified: no bin after the first floor peak should have high amplitude.
    """
    from processing.fusion import _FWD_FLOOR_THRESH
    scan = generate(0.0, 0.0, 0.0, floor, [])

    for b in range(N_BEAMS):
        base      = b * N_RANGE
        floor_hit = -1
        for ri in range(N_RANGE):
            if scan[base + ri] >= _FWD_FLOOR_THRESH:
                floor_hit = ri
                break
        if floor_hit < 0:
            continue
        # Everything at range > floor_hit + a small spread should be noise
        for ri in range(floor_hit + 8, N_RANGE):
            assert scan[base + ri] < _FWD_FLOOR_THRESH, (
                f"Beam {b}: high amplitude at ri={ri} after floor hit at {floor_hit}"
            )


def test_deterministic_with_same_rng(floor):
    import random
    rng1 = random.Random(0)
    rng2 = random.Random(0)
    s1 = generate(0.0, 0.0, 0.0, floor, [], rng=rng1)
    s2 = generate(0.0, 0.0, 0.0, floor, [], rng=rng2)
    assert s1 == s2


def test_heading_affects_output(floor):
    """Different headings should give different scans (floor topology varies)."""
    s0   = generate(0.0, 0.0,   0.0, floor, [])
    s90  = generate(0.0, 0.0,  90.0, floor, [])
    s180 = generate(0.0, 0.0, 180.0, floor, [])
    # At least two must differ
    assert not (s0 == s90 == s180)
