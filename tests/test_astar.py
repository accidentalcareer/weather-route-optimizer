"""Tests for backend/routing/astar.py, using small synthetic grids."""

import math

import numpy as np
import pytest

from routing.astar import RouteNotFoundError, astar, path_length_km, route_cost
from routing.cost import distance_cost, weather_cost
from routing.grid import GeoGrid


def make_grid(n_lat, n_lon, hazard=None, lat0=10.0, lon0=70.0, step=0.25):
    lats = lat0 + step * np.arange(n_lat)
    lons = lon0 + step * np.arange(n_lon)
    if hazard is None:
        hazard = np.zeros((n_lat, n_lon))
    return GeoGrid(lats, lons, hazard=hazard)


def is_connected_route(grid, nodes):
    return all(b in grid.neighbors(a) for a, b in zip(nodes, nodes[1:]))


def brute_force_min_cost(grid, start, goal, edge_cost):
    """Minimum cost over every simple path, by exhaustive DFS (tiny grids only)."""
    best = math.inf

    def dfs(node, cost, visited):
        nonlocal best
        if cost >= best:
            return
        if node == goal:
            best = cost
            return
        for nb in grid.neighbors(node):
            if nb not in visited:
                visited.add(nb)
                dfs(nb, cost + edge_cost(grid, node, nb), visited)
                visited.remove(nb)

    dfs(start, 0.0, {start})
    return best


# --- validation ---------------------------------------------------------

def test_start_equals_goal_rejected():
    g = make_grid(3, 3)
    with pytest.raises(ValueError, match="must differ"):
        astar(g, (1, 1), (1, 1), distance_cost)


@pytest.mark.parametrize("start, goal", [((-1, 0), (1, 1)), ((0, 0), (3, 0)), ((0, 0), (0, 9))])
def test_invalid_nodes_rejected(start, goal):
    with pytest.raises(IndexError):
        astar(make_grid(3, 3), start, goal, distance_cost)


@pytest.mark.parametrize("bad", [0.0, -1.0, math.inf, math.nan])
def test_invalid_edge_cost_rejected(bad):
    with pytest.raises(ValueError, match="finite and positive"):
        astar(make_grid(3, 3), (0, 0), (2, 2), lambda g, a, b: bad)


@pytest.mark.parametrize("bad", [-1.0, math.inf, math.nan])
def test_invalid_heuristic_rejected(bad):
    with pytest.raises(ValueError, match="finite and non-negative"):
        astar(make_grid(3, 3), (0, 0), (2, 2), distance_cost, heuristic=lambda g, n, t: bad)


def test_no_route_raises():
    # Column 2 is impassable (edge cost None), cutting the grid in two.
    g = make_grid(4, 5)

    def blocked(grid, a, b):
        return None if 2 in (a[1], b[1]) else distance_cost(grid, a, b)

    with pytest.raises(RouteNotFoundError, match="No route"):
        astar(g, (0, 0), (3, 4), blocked)


# --- reconstruction and length ------------------------------------------

def test_straight_route_reconstruction():
    g = make_grid(3, 5)
    r = astar(g, (1, 0), (1, 4), distance_cost)
    assert r.nodes == [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4)]
    assert r.start == (1, 0) and r.goal == (1, 4)
    assert r.nodes[0] == r.start and r.nodes[-1] == r.goal
    assert r.n_nodes == 5
    assert r.nodes_expanded >= 1


def test_diagonal_route():
    g = make_grid(4, 4)
    r = astar(g, (0, 0), (3, 3), distance_cost)
    assert r.nodes == [(0, 0), (1, 1), (2, 2), (3, 3)]


def test_path_length_is_sum_of_haversine_steps():
    g = make_grid(4, 6)
    r = astar(g, (0, 0), (3, 5), distance_cost)
    expected = sum(g.distance_km(a, b) for a, b in zip(r.nodes, r.nodes[1:]))
    assert r.path_length_km == pytest.approx(expected)
    assert path_length_km(g, r.nodes) == pytest.approx(expected)
    # Distance-only cost is the path length itself.
    assert r.total_cost == pytest.approx(r.path_length_km)
    assert is_connected_route(g, r.nodes)


def test_route_cost_matches_total_cost():
    rng = np.random.default_rng(1)
    g = make_grid(5, 5, hazard=rng.uniform(0, 1, (5, 5)))
    r = astar(g, (0, 0), (4, 4), weather_cost)
    assert route_cost(g, r.nodes, weather_cost) == pytest.approx(r.total_cost)


# --- optimality ---------------------------------------------------------

def test_minimum_distance_matches_brute_force():
    g = make_grid(3, 4)
    for start, goal in [((0, 0), (2, 3)), ((2, 0), (0, 3)), ((0, 1), (2, 2))]:
        r = astar(g, start, goal, distance_cost)
        assert r.total_cost == pytest.approx(brute_force_min_cost(g, start, goal, distance_cost))


def test_weather_route_matches_brute_force():
    rng = np.random.default_rng(42)
    g = make_grid(3, 4, hazard=rng.uniform(0, 1, (3, 4)))
    for start, goal in [((0, 0), (2, 3)), ((2, 0), (0, 3)), ((1, 0), (1, 3))]:
        r = astar(g, start, goal, weather_cost)
        assert r.total_cost == pytest.approx(brute_force_min_cost(g, start, goal, weather_cost))
        assert is_connected_route(g, r.nodes)


def test_hand_verified_route():
    # 2x3 grid, unit-cost orthogonal and sqrt(2) diagonal edges (planar costs,
    # not geographic, so the answer can be checked by hand).
    g = make_grid(2, 3)
    planar = lambda grid, a, b: math.hypot(a[0] - b[0], a[1] - b[1])  # noqa: E731
    zero_h = lambda grid, n, t: 0.0  # noqa: E731
    r = astar(g, (0, 0), (1, 2), planar, heuristic=zero_h)
    # Optimal: one diagonal + one straight step = sqrt(2) + 1.
    assert r.total_cost == pytest.approx(math.sqrt(2) + 1)
    assert r.n_nodes == 3
    assert r.nodes[0] == (0, 0) and r.nodes[-1] == (1, 2)


# --- weather avoidance --------------------------------------------------

def test_weather_route_avoids_high_hazard_corridor():
    # 3 x 5 grid. The direct row-1 corridor between start and goal is near
    # hazard 1.0; the rows above and below are calm.
    hazard = np.zeros((3, 5))
    hazard[1, 1:4] = 0.99
    g = make_grid(3, 5, hazard=hazard)
    start, goal = (1, 0), (1, 4)
    corridor = {(1, 1), (1, 2), (1, 3)}

    dist = astar(g, start, goal, distance_cost)
    wx = astar(g, start, goal, weather_cost)

    # Distance-only takes the straight, high-hazard line.
    assert dist.nodes == [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4)]
    # Weather-aware detours around it through calm cells.
    assert not corridor & set(wx.nodes)
    assert is_connected_route(g, wx.nodes)
    # The detour is longer geographically but much cheaper in weather cost.
    assert wx.path_length_km > dist.path_length_km
    assert wx.path_length_km < 1.5 * dist.path_length_km
    assert wx.total_cost < route_cost(g, dist.nodes, weather_cost)
    # In calm air the weather cost equals the distance.
    assert wx.total_cost == pytest.approx(wx.path_length_km)


def test_uniform_hazard_gives_same_route_as_distance():
    g = make_grid(4, 6, hazard=np.full((4, 6), 0.4))
    dist = astar(g, (0, 0), (3, 5), distance_cost)
    wx = astar(g, (0, 0), (3, 5), weather_cost)
    assert wx.path_length_km == pytest.approx(dist.path_length_km)
    assert wx.total_cost == pytest.approx(dist.total_cost * (1 + 4 * 0.4))


# --- route_cost ---------------------------------------------------------

def test_route_cost_rejects_non_neighbour_steps():
    g = make_grid(3, 3)
    with pytest.raises(ValueError, match="not between neighbouring nodes"):
        route_cost(g, [(0, 0), (2, 2)], distance_cost)


def test_route_cost_rejects_impassable_edge():
    g = make_grid(3, 3)
    with pytest.raises(ValueError, match="impassable"):
        route_cost(g, [(0, 0), (0, 1)], lambda grid, a, b: None)
