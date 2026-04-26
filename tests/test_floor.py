import pytest
from synthetic.floor import FloorModel


@pytest.fixture(scope="module")
def floor():
    return FloorModel(seed=42)


def test_depth_in_bounds(floor):
    # Allow float32 rounding (e.g. 10.9 stored as 10.8999...)
    eps = 0.01
    for e, n in [(0, 0), (100, 50), (-200, -200), (240, 240), (-240, 240)]:
        d = floor.depth_at(e, n)
        assert FloorModel.MIN_DEPTH_M - eps <= d <= FloorModel.MAX_DEPTH_M + eps, (
            f"depth {d} out of bounds at E={e} N={n}"
        )


def test_deterministic():
    f1 = FloorModel(seed=99)
    f2 = FloorModel(seed=99)
    assert f1.depth_at(50, 30) == f2.depth_at(50, 30)


def test_different_seeds_differ():
    f1 = FloorModel(seed=1)
    f2 = FloorModel(seed=2)
    assert f1.depth_at(50, 30) != f2.depth_at(50, 30)


def test_oob_clamps_not_raises(floor):
    # Positions well outside the 500m grid should clamp, not raise
    eps = 0.01
    d = floor.depth_at(99999, 99999)
    assert FloorModel.MIN_DEPTH_M - eps <= d <= FloorModel.MAX_DEPTH_M + eps


def test_sample_grid_dimensions(floor):
    grid = floor.sample_grid(step=1)
    n = floor._n
    assert grid["rows"] == n
    assert grid["cols"] == n
    assert len(grid["depth_m"]) == n
    assert len(grid["depth_m"][0]) == n


def test_sample_grid_coarser_step(floor):
    grid = floor.sample_grid(step=5)
    n = floor._n // 5
    assert grid["rows"] == n
    assert grid["cols"] == n
    assert grid["cell_size_m"] == pytest.approx(floor.CELL_SIZE_M * 5)


def test_sample_grid_transposition(floor):
    """
    After the transpose fix, depth_m[r][c] must equal depth_at at the
    corresponding (east, north) world position for that cell.
    """
    grid = floor.sample_grid(step=1)
    cell = floor.CELL_SIZE_M
    origin_e = -floor._half
    origin_n = -floor._half

    for r, c in [(0, 0), (10, 20), (50, 100), (100, 50), (200, 10)]:
        expected_e = origin_e + c * cell
        expected_n = origin_n + r * cell
        from_grid  = grid["depth_m"][r][c]
        from_fn    = round(floor.depth_at(expected_e, expected_n), 2)
        assert abs(from_grid - from_fn) < 0.01, (
            f"Transposition mismatch at r={r} c={c}: "
            f"grid={from_grid:.3f} depth_at={from_fn:.3f}"
        )


def test_sample_grid_origin_correct(floor):
    grid = floor.sample_grid(step=1)
    assert grid["origin_east_m"]  == pytest.approx(-floor._half)
    assert grid["origin_north_m"] == pytest.approx(-floor._half)
