import pytest
from synthetic.forward_scan import (
    generate, N_FORWARD, N_DEPTH, MAX_FORWARD_M, MAX_DEPTH_M,
)
from synthetic.floor import FloorModel
from synthetic.generator import FishSchool


@pytest.fixture(scope="module")
def floor():
    return FloorModel(seed=42)


def test_output_length(floor):
    scan = generate(0.0, 0.0, 0.0, floor, [])
    assert len(scan) == N_DEPTH * N_FORWARD


def test_output_is_bytes(floor):
    scan = generate(0.0, 0.0, 0.0, floor, [])
    assert isinstance(scan, bytes)


def test_all_values_in_byte_range(floor):
    scan = generate(0.0, 0.0, 0.0, floor, [])
    assert all(0 <= b <= 255 for b in scan)


def test_floor_return_present(floor):
    """
    A floor return must appear somewhere in the image (bright pixels at depth).
    The floor at (0,0) is ~14m deep, which falls within MAX_DEPTH_M=25m,
    so at least some beams should hit it.
    """
    scan = generate(0.0, 0.0, 0.0, floor, [])
    # At least one pixel should be above the background noise level
    max_amp = max(scan)
    assert max_amp > 100, f"No strong returns found (max={max_amp})"


def test_floor_return_at_correct_depth(floor):
    """
    Floor returns must appear in the correct depth rows of the image.
    Floor at (0,0) ≈ 14m deep; within MAX_DEPTH_M=25m → row ≈ 14/25*80 = 45.
    """
    scan = generate(0.0, 0.0, 0.0, floor, [])
    floor_d = floor.depth_at(0.0, 0.0)
    if floor_d > MAX_DEPTH_M:
        pytest.skip("floor depth exceeds display range")

    expected_row = int(floor_d / MAX_DEPTH_M * N_DEPTH)
    # Search a ±10-row window for a strong return
    window_rows = range(max(0, expected_row - 10), min(N_DEPTH, expected_row + 11))
    max_in_window = max(
        scan[r * N_FORWARD + c]
        for r in window_rows
        for c in range(N_FORWARD)
    )
    assert max_in_window > 80, (
        f"No strong return near expected floor row {expected_row} "
        f"(floor_d={floor_d:.1f}m, max_in_window={max_in_window})"
    )


def test_fish_school_increases_amplitude(floor):
    """A fish school within range must raise amplitude above background."""
    school    = FishSchool(0.0, 5.0, 4.0, 8.0, 0.9, "bass")
    scan_base = generate(0.0, 0.0, 0.0, floor, [])
    scan_fish = generate(0.0, 0.0, 0.0, floor, [school])
    increased = sum(1 for a, b in zip(scan_fish, scan_base) if a > b)
    assert increased > 0


def test_lateral_fish_visible(floor):
    """
    A fish school to the side (not directly ahead) should still appear
    in the image due to the horizontal azimuthal spread.
    """
    # Place school 10m to the right of the heading direction
    # heading=0 → north; right = east. School at east=10, north=5.
    school_side = FishSchool(10.0, 5.0, 4.0, 8.0, 0.9, "bass")
    school_fwd  = FishSchool( 0.0, 5.0, 4.0, 8.0, 0.9, "bass")

    scan_side = generate(0.0, 0.0, 0.0, floor, [school_side])
    scan_fwd  = generate(0.0, 0.0, 0.0, floor, [school_fwd])
    scan_none = generate(0.0, 0.0, 0.0, floor, [])

    # Lateral school should produce more signal than empty scan
    diff_side = sum(max(0, a - b) for a, b in zip(scan_side, scan_none))
    assert diff_side > 0, "Lateral school produced no extra signal (azimuthal spread missing)"


def test_heading_affects_output(floor):
    """Different headings should give different scans (floor topology varies)."""
    s0   = generate(0.0, 0.0,   0.0, floor, [])
    s90  = generate(0.0, 0.0,  90.0, floor, [])
    s180 = generate(0.0, 0.0, 180.0, floor, [])
    assert not (s0 == s90 == s180)


def test_deterministic_with_same_rng(floor):
    import random
    rng1 = random.Random(0)
    rng2 = random.Random(0)
    s1 = generate(0.0, 0.0, 0.0, floor, [], rng=rng1)
    s2 = generate(0.0, 0.0, 0.0, floor, [], rng=rng2)
    assert s1 == s2


def test_image_has_volume_reverberation(floor):
    """Near-surface rows should be brighter on average than deep rows (reverberation)."""
    scan = generate(0.0, 0.0, 0.0, floor, [])
    # Average amplitude in first 5 rows (near surface) vs last 5 rows (deep)
    near_avg = sum(scan[r * N_FORWARD + c] for r in range(5) for c in range(N_FORWARD)) / (5 * N_FORWARD)
    deep_avg = sum(scan[r * N_FORWARD + c] for r in range(N_DEPTH - 5, N_DEPTH) for c in range(N_FORWARD)) / (5 * N_FORWARD)
    assert near_avg >= deep_avg, (
        f"Surface noise ({near_avg:.1f}) should be >= deep noise ({deep_avg:.1f})"
    )
