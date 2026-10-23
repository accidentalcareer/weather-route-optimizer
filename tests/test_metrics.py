"""Tests for backend/routing/metrics.py, using small synthetic grids."""

import json
import math

import numpy as np
import pytest

from routing.astar import astar
from routing.cost import distance_cost, weather_cost
from routing.grid import GeoGrid
from routing.metrics import FUEL_PROXY_NOTE, compare_routes, validate_route


def test_metrics_module_imports():
    from routing import metrics  # noqa: F401


def make_grid(hazard):
    hazard = np.asarray(hazard, dtype=float)
    lats = 10.0 + 0.25 * np.arange(hazard.shape[0])
    lons = 70.0 + 0.25 * np.arange(hazard.shape[1])
    return GeoGrid(lats, lons, hazard=hazard)


@pytest.fixture
def corridor():
    """3x5 grid: severe corridor on row 1 between calm rows (as in test_astar)."""
    hazard = np.zeros((3, 5))
    hazard[1, 1:4] = 0.9  # SEVERE
    g = make_grid(hazard)
    dist = astar(g, (1, 0), (1, 4), distance_cost)
    wx = astar(g, (1, 0), (1, 4), weather_cost)
    return g, dist, wx


def legs(g, nodes):
    return [g.distance_km(a, b) for a, b in zip(nodes, nodes[1:])]


# --- distance, percentages, fuel proxy ----------------------------------

def test_extra_distance_and_percent(corridor):
    g, dist, wx = corridor
    c = compare_routes(g, dist, wx)
    assert c.distance_route_km == pytest.approx(sum(legs(g, dist.nodes)))
    assert c.weather_route_km == pytest.approx(sum(legs(g, wx.nodes)))
    assert c.extra_distance_km == pytest.approx(c.weather_route_km - c.distance_route_km)
    assert c.extra_distance_km > 0  # deliberately longer weather route
    assert c.extra_distance_percent == pytest.approx(
        100 * c.extra_distance_km / c.distance_route_km
    )


def test_fuel_proxy(corridor):
    g, dist, wx = corridor
    c = compare_routes(g, dist, wx)
    expected = (c.weather_route_km - c.distance_route_km) / c.distance_route_km * 100
    assert c.estimated_fuel_delta_percent == pytest.approx(expected)
    assert c.fuel_proxy_note == FUEL_PROXY_NOTE
    assert "not an operational" in c.fuel_proxy_note.lower()


# --- weather cost -------------------------------------------------------

def test_weather_cost_comparison(corridor):
    g, dist, wx = corridor
    c = compare_routes(g, dist, wx)

    # Distance route: straight through the corridor. Edge hazards are
    # (0 + .9)/2, .9, .9, (.9 + 0)/2 -> multipliers 2.8, 4.6, 4.6, 2.8.
    d_legs = legs(g, dist.nodes)
    expected = sum(km * m for km, m in zip(d_legs, [2.8, 4.6, 4.6, 2.8]))
    assert c.distance_route_weather_cost == pytest.approx(expected)

    # Weather route is fully calm: weather cost equals its length.
    assert c.weather_route_weather_cost == pytest.approx(c.weather_route_km)
    assert c.weather_route_weather_cost == pytest.approx(wx.total_cost)

    assert c.weather_cost_difference == pytest.approx(
        c.distance_route_weather_cost - c.weather_route_weather_cost
    )
    assert c.weather_cost_reduction_percent == pytest.approx(
        100 * c.weather_cost_difference / c.distance_route_weather_cost
    )
    assert c.weather_cost_reduction_percent > 0


# --- hazard -------------------------------------------------------------

def test_hazard_counts_and_avoidance(corridor):
    g, dist, wx = corridor
    c = compare_routes(g, dist, wx)
    assert c.distance_route.category_counts == {"LOW": 2, "MODERATE": 0, "HIGH": 0, "SEVERE": 3}
    assert c.weather_route.category_counts["SEVERE"] == 0
    assert sum(c.weather_route.category_counts.values()) == len(wx.nodes)
    assert c.severe_cells_avoided == 3
    assert c.high_hazard_cells_avoided == 3


def test_average_and_max_hazard():
    hazard = np.array([[0.1, 0.3, 0.6, 0.8]])
    g = make_grid(np.vstack([hazard, np.zeros((1, 4))]))
    route = [(0, 0), (0, 1), (0, 2), (0, 3)]
    calm = [(0, 0), (1, 1), (1, 2), (0, 3)]
    c = compare_routes(g, route, calm)
    assert c.distance_route.average_hazard == pytest.approx(0.45)
    assert c.distance_route.maximum_hazard == pytest.approx(0.8)
    assert c.distance_route.category_counts == {"LOW": 1, "MODERATE": 1, "HIGH": 1, "SEVERE": 1}
    # Calm route visits (0,0)=0.1, two zeros, (0,3)=0.8.
    assert c.weather_route.average_hazard == pytest.approx(0.225)
    assert c.average_hazard_reduction == pytest.approx(0.225)
    assert c.maximum_hazard_reduction == pytest.approx(0.0)
    assert c.high_hazard_cells_avoided == 1  # (0,2) HIGH avoided; (0,3) SEVERE shared
    assert c.severe_cells_avoided == 0


def test_negative_avoidance_is_reported():
    # Swap roles: the "weather" route is the hazardous one.
    hazard = np.zeros((3, 5))
    hazard[1, 1:4] = 0.9
    g = make_grid(hazard)
    straight = [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4)]
    detour = [(1, 0), (0, 1), (0, 2), (0, 3), (1, 4)]
    c = compare_routes(g, detour, straight)
    assert c.severe_cells_avoided == -3
    assert c.weather_cost_reduction_percent < 0
    assert "crosses 3 more severe-hazard cells" in c.explanation
    assert "increases weather-weighted cost" in c.explanation


# --- identical routes, explanation, output ------------------------------

def test_identical_routes():
    g = make_grid(np.full((3, 4), 0.3))
    r = astar(g, (0, 0), (2, 3), distance_cost)
    c = compare_routes(g, r, r)
    assert c.routes_identical
    assert c.extra_distance_km == 0.0
    assert c.extra_distance_percent == 0.0
    assert c.weather_cost_difference == 0.0
    assert c.weather_cost_reduction_percent == 0.0
    assert c.severe_cells_avoided == 0
    assert c.estimated_fuel_delta_percent == 0.0
    assert "identical" in c.explanation


def test_explanation_uses_computed_values(corridor):
    g, dist, wx = corridor
    c = compare_routes(g, dist, wx)
    assert not c.routes_identical
    assert f"adds {c.extra_distance_km:.1f} km ({c.extra_distance_percent:.1f}%)" in c.explanation
    assert f"by {c.weather_cost_reduction_percent:.1f}%" in c.explanation
    assert "avoids 3 severe-hazard cells" in c.explanation


def test_summary_and_dict_are_json_ready(corridor):
    g, dist, wx = corridor
    c = compare_routes(g, dist, wx)
    s = c.summary()
    for key in ("extra_distance_km", "extra_distance_percent", "weather_cost_reduction_percent",
                "severe_cells_avoided", "estimated_fuel_delta_percent", "explanation"):
        assert key in s
    d = json.loads(json.dumps(c.to_dict()))
    assert d["start"] == [1, 0] and d["goal"] == [1, 4]
    assert d["summary"]["severe_cells_avoided"] == 3


def test_metrics_are_finite(corridor):
    g, dist, wx = corridor
    d = compare_routes(g, dist, wx).to_dict()
    for k, v in d.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            assert math.isfinite(v), k


# --- validation ---------------------------------------------------------

@pytest.fixture
def calm_grid():
    return make_grid(np.zeros((3, 4)))


def test_route_too_short(calm_grid):
    with pytest.raises(ValueError, match="at least two nodes"):
        validate_route(calm_grid, [(0, 0)])


def test_route_node_off_grid(calm_grid):
    with pytest.raises(ValueError, match="outside grid"):
        validate_route(calm_grid, [(0, 0), (0, 1), (0, 9)])


def test_route_not_neighbours(calm_grid):
    with pytest.raises(ValueError, match="not neighbours"):
        validate_route(calm_grid, [(0, 0), (0, 2)])


def test_route_repeated_node(calm_grid):
    # A node is not its own neighbour, so zero-length legs are rejected.
    with pytest.raises(ValueError, match="not neighbours"):
        validate_route(calm_grid, [(0, 0), (0, 0), (0, 1)])


def test_invalid_hazard_rejected():
    hazard = np.zeros((3, 4))
    hazard[0, 1] = np.nan
    g = make_grid(hazard)
    with pytest.raises(ValueError, match="hazard"):
        validate_route(g, [(0, 0), (0, 1)])


def test_mismatched_endpoints(calm_grid):
    with pytest.raises(ValueError, match="share start and goal"):
        compare_routes(calm_grid, [(0, 0), (0, 1)], [(0, 0), (1, 1)])


def test_route_from_different_grid_rejected():
    # Same shape, different coordinates: the stored length no longer matches.
    other = GeoGrid(30.0 + np.arange(3) * 0.25, 70.0 + np.arange(4) * 0.25, hazard=np.zeros((3, 4)))
    here = make_grid(np.zeros((3, 4)))
    r = astar(other, (0, 0), (2, 3), distance_cost)
    with pytest.raises(ValueError, match="different grid"):
        compare_routes(here, r, r)
