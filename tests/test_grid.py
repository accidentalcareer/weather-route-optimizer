"""Tests for backend/routing/grid.py, using small synthetic grids."""

import numpy as np
import pytest
import xarray as xr

from routing.grid import GeoGrid, haversine_km

# 4 x 5 grid at 0.25 deg, latitude ascending (south -> north), like the GFS subset.
LATS = [10.0, 10.25, 10.5, 10.75]
LONS = [70.0, 70.25, 70.5, 70.75, 71.0]


@pytest.fixture
def grid():
    ds = xr.Dataset(coords={"latitude": LATS, "longitude": LONS})
    return GeoGrid.from_dataset(ds)


# --- construction and validation ----------------------------------------

def test_shape_and_coordinates_preserved(grid):
    assert grid.shape == (4, 5)
    assert grid.n_nodes == 20
    np.testing.assert_array_equal(grid.lats, LATS)
    np.testing.assert_array_equal(grid.lons, LONS)


def test_empty_grid_rejected():
    with pytest.raises(ValueError, match="empty"):
        GeoGrid(np.array([]), np.array(LONS))


def test_missing_coordinate_rejected():
    ds = xr.Dataset(coords={"latitude": LATS})
    with pytest.raises(ValueError, match="longitude"):
        GeoGrid.from_dataset(ds)


def test_non_monotonic_rejected():
    with pytest.raises(ValueError, match="monotonic"):
        GeoGrid(np.array([10.0, 10.5, 10.25]), np.array(LONS))


# --- index <-> lat/lon --------------------------------------------------

def test_index_to_latlon(grid):
    assert grid.index_to_latlon((0, 0)) == (10.0, 70.0)
    assert grid.index_to_latlon((3, 4)) == (10.75, 71.0)
    assert grid.index_to_latlon((2, 1)) == (10.5, 70.25)


def test_latlon_to_index_roundtrip(grid):
    for i in range(4):
        for j in range(5):
            assert grid.latlon_to_index(*grid.index_to_latlon((i, j))) == (i, j)


def test_latlon_to_index_rejects_off_grid_point(grid):
    with pytest.raises(KeyError):
        grid.latlon_to_index(10.1, 70.0)


# --- bounds -------------------------------------------------------------

@pytest.mark.parametrize("node", [(-1, 0), (0, -1), (4, 0), (0, 5), (10, 10)])
def test_out_of_bounds_nodes(grid, node):
    assert not grid.in_bounds(node)
    with pytest.raises(IndexError):
        grid.index_to_latlon(node)
    with pytest.raises(IndexError):
        grid.neighbors(node)


# --- neighbours ---------------------------------------------------------

@pytest.mark.parametrize("corner", [(0, 0), (0, 4), (3, 0), (3, 4)])
def test_corner_has_three_neighbors(grid, corner):
    assert len(grid.neighbors(corner)) == 3


def test_edge_node_has_five_neighbors(grid):
    assert len(grid.neighbors((0, 2))) == 5


def test_interior_node_has_eight_neighbors(grid):
    nbs = grid.neighbors((1, 2))
    assert len(nbs) == 8
    assert len(set(nbs)) == 8
    assert (1, 2) not in nbs
    assert set(nbs) == {(1 + di, 2 + dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)} - {(1, 2)}


def test_all_neighbors_in_bounds(grid):
    for i in range(4):
        for j in range(5):
            assert all(grid.in_bounds(nb) for nb in grid.neighbors((i, j)))


def test_neighbor_count_matches_formula(grid):
    total = sum(len(grid.neighbors((i, j))) for i in range(4) for j in range(5))
    assert total == grid.n_directed_edges()


def test_compass_directions_follow_geography(grid):
    nb = grid.neighbors_by_direction((1, 2))
    lat0, lon0 = grid.index_to_latlon((1, 2))
    assert grid.index_to_latlon(nb["N"])[0] > lat0
    assert grid.index_to_latlon(nb["S"])[0] < lat0
    assert grid.index_to_latlon(nb["E"])[1] > lon0
    assert grid.index_to_latlon(nb["W"])[1] < lon0


def test_compass_directions_with_descending_latitude():
    # Same region stored north -> south, as in a full GFS file.
    g = GeoGrid(np.array(LATS[::-1]), np.array(LONS))
    nb = g.neighbors_by_direction((1, 2))
    assert g.index_to_latlon(nb["N"])[0] > g.index_to_latlon((1, 2))[0]


# --- distance -----------------------------------------------------------

def test_haversine_known_values():
    assert haversine_km(0, 0, 0, 0) == 0.0
    # One degree of latitude ~ 111.2 km on a sphere of radius 6371 km.
    assert haversine_km(0, 0, 1, 0) == pytest.approx(111.195, abs=0.01)
    # Same whether longitude is written as -90 or 270.
    assert haversine_km(10, -90, 11, -89) == pytest.approx(haversine_km(10, 270, 11, 271))


def test_distances_positive_for_distinct_neighbors(grid):
    for i in range(4):
        for j in range(5):
            for nb in grid.neighbors((i, j)):
                assert grid.distance_km((i, j), nb) > 0
    assert grid.distance_km((1, 1), (1, 1)) == 0.0


def test_diagonal_distance_is_geographic(grid):
    node = (1, 2)
    nb = grid.neighbors_by_direction(node)
    north = grid.distance_km(node, nb["N"])
    east = grid.distance_km(node, nb["E"])
    diag = grid.distance_km(node, nb["NE"])

    # East-west spacing shrinks with cos(lat); at ~10 deg it is a bit shorter.
    assert east < north
    assert east == pytest.approx(north * np.cos(np.radians(10.25)), rel=1e-3)
    # Diagonal is longer than either side, close to the planar hypotenuse,
    # and clearly not just equal to an orthogonal step.
    assert diag > north and diag > east
    assert diag == pytest.approx(np.hypot(north, east), rel=1e-3)
    assert diag == pytest.approx(
        haversine_km(*grid.index_to_latlon(node), *grid.index_to_latlon(nb["NE"]))
    )


# --- nearest node -------------------------------------------------------

def test_nearest_node_exact_and_offset(grid):
    assert grid.nearest_node(10.5, 70.25) == (2, 1)
    assert grid.nearest_node(10.55, 70.20) == (2, 1)
    assert grid.nearest_node(10.70, 70.90) == (3, 4)


def test_nearest_node_just_outside_edge_snaps(grid):
    # Within half a cell beyond the edge -> snaps to the edge node.
    assert grid.nearest_node(9.9, 69.9) == (0, 0)


def test_nearest_node_far_outside_rejected(grid):
    with pytest.raises(ValueError, match="outside grid extent"):
        grid.nearest_node(20.0, 70.0)
    with pytest.raises(ValueError, match="outside grid extent"):
        grid.nearest_node(10.0, 80.0)


def test_nearest_node_handles_longitude_convention():
    g = GeoGrid(np.array(LATS), np.array([359.5, 359.75]))  # 0..360 grid
    assert g.nearest_node(10.0, -0.25) == (0, 1)


def test_nearest_node_returns_valid_node(grid):
    rng = np.random.default_rng(0)
    for lat, lon in zip(rng.uniform(10.0, 10.75, 50), rng.uniform(70.0, 71.0, 50)):
        node = grid.nearest_node(lat, lon)
        assert grid.in_bounds(node)
        best = min(
            ((i, j) for i in range(4) for j in range(5)),
            key=lambda n: haversine_km(lat, lon, *grid.index_to_latlon(n)),
        )
        assert node == best
