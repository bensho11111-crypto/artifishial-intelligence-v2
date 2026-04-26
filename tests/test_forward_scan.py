import math
import pytest
from synthetic.forward_scan import (
    generate, N_BEAMS, N_RANGE, MAX_RANGE_M,
    BEAM_MIN_DEG, BEAM_MAX_DEG,
    N_AZIMUTH, AZIMUTH_HALF_DEG,
)
from synthetic.floor import FloorModel
from synthetic.generator import FishSchool


@pytest.fixture(scope="module")
def floor():
    return FloorModel(seed=42)


def _slab(scan: bytes, az: int) -> bytes:
    """Return the (beam × range) slab for one azimuth slice."""
    slab_size = N_BEAMS * N_RANGE
    return scan[az * slab_size:(az + 1) * slab_size]


def _centre_az() -> int:
    return N_AZIMUTH // 2


def test_output_length(floor):
    scan = generate(0.0, 0.0, 0.0, floor, [])
    assert len(scan) == N_AZIMUTH * N_BEAMS * N_RANGE


def test_output_is_bytes(floor):
    scan = generate(0.0, 0.0, 0.0, floor, [])
    assert isinstance(scan, bytes)


def test_all_values_in_byte_range(floor):
    scan = generate(0.0, 0.0, 0.0, floor, [])
    assert all(0 <= b <= 255 for b in scan)


def test_floor_return_in_expected_range(floor):
    """
    For steep beams (>45°) the floor return must appear at a range bin
    consistent with the floor depth at the boat position. Test the centre
    azimuth slice (boresight).
    """
    from processing.fusion import _FWD_FLOOR_THRESH
    scan   = generate(0.0, 0.0, 0.0, floor, [])
    slab   = _slab(scan, _centre_az())
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
        window = range(max(0, exp_ri - 5), min(N_RANGE, exp_ri + 6))
        has_floor = any(slab[base + ri] >= _FWD_FLOOR_THRESH for ri in window)
        assert has_floor, (
            f"Beam {b} ({theta_deg:.0f}°): no floor return near expected ri={exp_ri} "
            f"(floor_d={floor_d:.1f}m, exp_range={exp_range:.1f}m)"
        )


def test_fish_school_increases_amplitude(floor):
    """A fish school within beam range must raise amplitude above background."""
    school    = FishSchool(5.0, 0.0, 4.0, 8.0, 0.9, "bass")
    scan_base = generate(0.0, 0.0, 0.0, floor, [])
    scan_fish = generate(0.0, 0.0, 0.0, floor, [school])
    increased = sum(1 for a, b in zip(scan_fish, scan_base) if a > b)
    assert increased > 0


def test_floor_blocks_further_returns(floor):
    """
    Once the floor is hit for a beam, no high-amplitude returns should appear
    at deeper range bins. Check every azimuth slice.
    """
    from processing.fusion import _FWD_FLOOR_THRESH
    scan = generate(0.0, 0.0, 0.0, floor, [])

    for a in range(N_AZIMUTH):
        slab = _slab(scan, a)
        for b in range(N_BEAMS):
            base      = b * N_RANGE
            floor_hit = -1
            for ri in range(N_RANGE):
                if slab[base + ri] >= _FWD_FLOOR_THRESH:
                    floor_hit = ri
                    break
            if floor_hit < 0:
                continue
            for ri in range(floor_hit + 8, N_RANGE):
                assert slab[base + ri] < _FWD_FLOOR_THRESH, (
                    f"Az {a} Beam {b}: high amp at ri={ri} after floor hit at {floor_hit}"
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
    assert not (s0 == s90 == s180)


def test_off_axis_school_appears_in_off_axis_slice(floor):
    """
    A school placed well to starboard should raise amplitude in the
    starboard-side azimuth slices but not in the port-side slices.

    Heading = 0° means forward = +north. A school at (east=+15, north=+15) sits
    ~45° to starboard — outside the centre boresight but inside the 60° FOV.
    """
    school = FishSchool(east_m=8.0, north_m=12.0, depth_m=4.0,
                        radius_m=4.0, density=0.95, species="bass")
    scan_base = generate(0.0, 0.0, 0.0, floor, [])
    scan_fish = generate(0.0, 0.0, 0.0, floor, [school])

    # Total amplitude lift per azimuth slice
    slab_size = N_BEAMS * N_RANGE
    lift = []
    for a in range(N_AZIMUTH):
        s_base = scan_base[a*slab_size:(a+1)*slab_size]
        s_fish = scan_fish[a*slab_size:(a+1)*slab_size]
        lift.append(sum(max(0, x - y) for x, y in zip(s_fish, s_base)))

    starboard_max = max(lift[N_AZIMUTH // 2:])
    port_max      = max(lift[:N_AZIMUTH // 2])
    assert starboard_max > port_max, (
        f"Off-axis school should boost starboard slices more than port. "
        f"starboard_max={starboard_max}, port_max={port_max}"
    )
