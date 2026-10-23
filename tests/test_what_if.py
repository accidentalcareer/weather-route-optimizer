"""Tests for What-if Mode: the hazard-penalty cost factory, the what-if
comparison and the ``hazard_penalty`` field of POST /api/route.

Unit tests use small synthetic grids; API tests use the real processed GFS
hazard files and are skipped when those are absent.
"""

import dataclasses
import json
import math

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import app
from routing import service
from routing.astar import astar, route_cost
from routing.cost import (
    DEFAULT_HAZARD_PENALTY,
    distance_cost,
    edge_cost_details,
    make_weather_cost,
    validate_hazard_penalty,
    weather_cost,
)
from routing.grid import GeoGrid, load_hazard_grid
from routing.metrics import compare_routes, compare_what_if, explain_what_if

client = TestClient(app)

DELHI = {"lat": 28.6139, "lon": 77.2090}
MUMBAI = {"lat": 19.0760, "lon": 72.8777}
F006 = "2026-10-22T18:00:00Z"
F024 = "2026-10-23T12:00:00Z"


def wall_grid() -> GeoGrid:
    """7x9 grid with a moderately hazardous wall the route can go around.

    The wall (hazard 0.3) fills column 4 in rows 0-4, so the direct west-east
    path along row 3 crosses it; the calm crossing is a two-row detour north.
    Crossing adds about 2 * 0.15 * K edge lengths of cost, the detour about
    1.7, so K = 4 crosses the wall and K = 10 goes around it.
    """
    hazard = np.zeros((7, 9))
    hazard[:5, 4] = 0.3
    return GeoGrid(np.arange(20.0, 21.75, 0.25), np.arange(75.0, 77.25, 0.25), hazard)


START, GOAL = (3, 0), (3, 8)


# --- cost factory -------------------------------------------------------

def test_default_penalty_is_the_existing_weather_cost():
    assert DEFAULT_HAZARD_PENALTY == 4.0
    assert make_weather_cost() is weather_cost
    assert make_weather_cost(4) is weather_cost
    assert make_weather_cost(4.0) is weather_cost


@pytest.mark.parametrize("k", [0.0, 0.5, 2.0, 7.5, 10.0])
def test_scenario_cost_formula(k):
    grid = wall_grid()
    cost = make_weather_cost(k)
    for a, b in [((2, 3), (2, 4)), ((3, 4), (4, 5)), ((6, 3), (6, 4)), ((0, 0), (1, 1))]:
        d = edge_cost_details(grid, a, b)
        assert cost(grid, a, b) == pytest.approx(d["distance_km"] * (1 + k * d["edge_hazard"]), rel=1e-15)


def test_k4_scenario_formula_matches_weather_cost_numerically():
    """Even computed through the general formula, K = 4 equals weather_cost."""
    grid = wall_grid()
    for a in [(2, 3), (3, 4), (5, 4)]:
        for b in grid.neighbors(a):
            d = edge_cost_details(grid, a, b)
            assert d["distance_km"] * (1 + 4.0 * d["edge_hazard"]) == weather_cost(grid, a, b)


def test_k0_cost_equals_distance_cost():
    grid = wall_grid()
    cost = make_weather_cost(0)
    for a in [(2, 3), (3, 4), (0, 0)]:
        for b in grid.neighbors(a):
            assert cost(grid, a, b) == distance_cost(grid, a, b)


@pytest.mark.parametrize("bad", [-0.1, -1, 10.0001, 11, math.nan, math.inf, -math.inf, True, "4", None])
def test_invalid_penalty_rejected(bad):
    with pytest.raises(ValueError):
        validate_hazard_penalty(bad)
    with pytest.raises(ValueError):
        make_weather_cost(bad)


@pytest.mark.parametrize("k", [0, 0.0, 10, 10.0, 4, 3.25])
def test_valid_penalty_accepted(k):
    assert validate_hazard_penalty(k) == float(k)


# --- routing with the factory (same astar) ------------------------------

def test_k0_route_is_the_distance_route():
    grid = wall_grid()
    d = astar(grid, START, GOAL, distance_cost)
    w0 = astar(grid, START, GOAL, make_weather_cost(0))
    assert w0.nodes == d.nodes
    assert w0.total_cost == pytest.approx(d.total_cost)


def test_higher_penalty_detours_around_the_wall():
    grid = wall_grid()
    normal = astar(grid, START, GOAL, weather_cost)
    strict = astar(grid, START, GOAL, make_weather_cost(10))
    assert any(grid.hazard[n] > 0 for n in normal.nodes)    # K=4 crosses the wall
    assert all(grid.hazard[n] == 0 for n in strict.nodes)   # K=10 goes around it
    assert strict.path_length_km > normal.path_length_km


def test_route_trade_off_is_monotonic_in_k():
    """cost_K = length + K * hazard_exposure, so as K grows the optimal route's
    exposure can only fall and its length can only rise."""
    rng = np.random.default_rng(7)
    grid = GeoGrid(np.arange(10.0, 15.0, 0.25), np.arange(70.0, 75.0, 0.25), rng.random((20, 20)))
    start, goal = (0, 0), (19, 19)
    exposure = lambda r: route_cost(grid, r.nodes, make_weather_cost(1)) - r.path_length_km  # noqa: E731
    previous = None
    for k in [0, 1, 2, 4, 6, 8, 10]:
        r = astar(grid, start, goal, make_weather_cost(k))
        if previous is not None:
            assert exposure(r) <= exposure(previous) + 1e-9
            assert r.path_length_km >= previous.path_length_km - 1e-9
        previous = r


def test_scenario_routing_does_not_modify_hazard():
    grid = wall_grid()
    before = grid.hazard.copy()
    astar(grid, START, GOAL, make_weather_cost(10))
    astar(grid, START, GOAL, make_weather_cost(0))
    np.testing.assert_array_equal(grid.hazard, before)


# --- what-if comparison -------------------------------------------------

def test_compare_what_if_identical_routes():
    grid = wall_grid()
    r = astar(grid, START, GOAL, weather_cost)
    c = compare_what_if(grid, r, r, 4.0)
    assert c.routes_identical
    assert c.extra_distance_km == 0 and c.weather_cost_difference == 0
    assert c.severe_cells_avoided == 0 and c.high_hazard_cells_avoided == 0
    assert c.explanation == (
        "With hazard sensitivity K=4.0, the what-if route is identical to the "
        "normal weather-aware route (K=4.0)."
    )


def test_compare_what_if_detour_metrics_and_explanation():
    grid = wall_grid()
    normal = astar(grid, START, GOAL, weather_cost)
    strict = astar(grid, START, GOAL, make_weather_cost(10))
    c = compare_what_if(grid, normal, strict, 10)
    assert not c.routes_identical
    assert c.extra_distance_km == pytest.approx(strict.path_length_km - normal.path_length_km)
    assert c.baseline_route.weather_cost == pytest.approx(route_cost(grid, normal.nodes, weather_cost))
    assert c.what_if_route.weather_cost == pytest.approx(route_cost(grid, strict.nodes, weather_cost))
    assert c.what_if_route.average_hazard == 0.0
    assert c.average_hazard_reduction_percent == pytest.approx(100.0)
    assert c.explanation == (
        f"With hazard sensitivity K=10.0, the what-if route adds {c.extra_distance_km:.1f} km "
        f"({c.extra_distance_percent:.1f}%) compared with the normal weather-aware route "
        "while reducing average hazard by 100.0%."
    )


def test_compare_what_if_lower_k_explanation():
    grid = wall_grid()
    normal = astar(grid, START, GOAL, make_weather_cost(10))  # pretend baseline avoids the wall
    loose = astar(grid, START, GOAL, make_weather_cost(0))
    c = compare_what_if(grid, normal, loose, 0)
    assert c.extra_distance_km < 0
    assert c.baseline_route.average_hazard == 0 < c.what_if_route.average_hazard
    assert c.explanation == (
        f"With hazard sensitivity K=0.0, the what-if route saves {-c.extra_distance_km:.1f} km "
        f"({-c.extra_distance_percent:.1f}%) compared with the normal weather-aware route "
        f"while raising average hazard from 0 to {c.what_if_route.average_hazard:.3f}."
    )


@pytest.mark.parametrize("extra_km, extra_pct, base_avg, new_avg, red_pct, expected", [
    (12.0, 1.0, 0.4, 0.3, 25.0, "adds 12.0 km (1.0%) compared with the normal weather-aware route "
                                "while reducing average hazard by 25.0%."),
    (-8.0, -0.7, 0.4, 0.5, -25.0, "saves 8.0 km (0.7%) compared with the normal weather-aware route "
                                  "while increasing average hazard by 25.0%."),
    (-8.0, -0.7, 0.4, 0.3, 25.0, "saves 8.0 km (0.7%) compared with the normal weather-aware route, "
                                 "reducing average hazard by 25.0%."),
    (12.0, 1.0, 0.4, 0.5, -25.0, "adds 12.0 km (1.0%) compared with the normal weather-aware route, "
                                 "increasing average hazard by 25.0%."),
    (0.01, 0.0, 0.4, 0.4, -0.01, "has about the same length compared with the normal weather-aware "
                                 "route, with about the same average hazard."),
    (-8.0, -0.7, 0.0, 0.2, 0.0, "saves 8.0 km (0.7%) compared with the normal weather-aware route "
                                "while raising average hazard from 0 to 0.200."),
])
def test_explain_what_if_branches(extra_km, extra_pct, base_avg, new_avg, red_pct, expected):
    grid = wall_grid()
    r = astar(grid, START, GOAL, weather_cost)
    c = compare_what_if(grid, r, r, 6)
    c = dataclasses.replace(
        c, routes_identical=False, extra_distance_km=extra_km, extra_distance_percent=extra_pct,
        baseline_route=dataclasses.replace(c.baseline_route, average_hazard=base_avg),
        what_if_route=dataclasses.replace(c.what_if_route, average_hazard=new_avg),
        average_hazard_reduction_percent=red_pct,
    )
    assert explain_what_if(c) == f"With hazard sensitivity K=6.0, the what-if route {expected}"


def test_compare_what_if_requires_shared_endpoints():
    grid = wall_grid()
    a = astar(grid, (3, 0), (3, 8), weather_cost)
    b = astar(grid, (2, 0), (3, 8), weather_cost)
    with pytest.raises(ValueError, match="share start and goal"):
        compare_what_if(grid, a, b, 4)


# --- service ------------------------------------------------------------

def has_real_data():
    try:
        return service.default_hazard_path().is_file()
    except service.WeatherDataError:
        return False


needs_data = pytest.mark.skipif(not has_real_data(), reason="processed hazard files not available")


@pytest.fixture(autouse=True)
def reset_cache():
    service.clear_hazard_cache()
    yield
    service.clear_hazard_cache()


@pytest.mark.parametrize("bad", [math.nan, math.inf, -1.0, 10.5])
def test_service_rejects_invalid_penalty_before_loading(bad, monkeypatch):
    monkeypatch.setattr(service, "load_snapshot", lambda p: pytest.fail("grid loaded"))
    with pytest.raises(service.InvalidHazardPenaltyError):
        service.calculate_route(DELHI["lat"], DELHI["lon"], MUMBAI["lat"], MUMBAI["lon"],
                                hazard_penalty=bad)


# --- API ----------------------------------------------------------------

def post(**extra):
    return client.post("/api/route", json={"origin": DELHI, "destination": MUMBAI, **extra})


def without_what_if(body):
    return {k: v for k, v in body.items() if k != "what_if"}


def direct(valid_time, k):
    """Routes computed straight from the hazard file with the routing layers."""
    path = service.resolve_forecast(valid_time).path
    grid, _ = load_hazard_grid(path)
    s = grid.nearest_node(DELHI["lat"], DELHI["lon"])
    t = grid.nearest_node(MUMBAI["lat"], MUMBAI["lon"])
    return grid, astar(grid, s, t, distance_cost), astar(grid, s, t, weather_cost), astar(
        grid, s, t, make_weather_cost(k))


@pytest.fixture(scope="module")
def normal_body():
    if not has_real_data():
        pytest.skip("processed hazard files not available")
    service.clear_hazard_cache()
    r = post()
    assert r.status_code == 200, r.text
    return r.json()


@needs_data
def test_omitted_penalty_reports_disabled_what_if(normal_body):
    assert normal_body["what_if"] == {
        "enabled": False, "hazard_penalty": 4.0,
        "baseline_route": None, "what_if_route": None, "comparison": None,
    }


@needs_data
@pytest.mark.parametrize("k", [0, 4, 7.5, 10])
def test_what_if_leaves_existing_sections_unchanged(normal_body, k):
    body = post(hazard_penalty=k).json()
    assert without_what_if(body) == without_what_if(normal_body)
    assert body["what_if"]["enabled"] is True
    assert body["what_if"]["hazard_penalty"] == float(k)


@needs_data
def test_k4_reproduces_the_weather_route(normal_body):
    w = post(hazard_penalty=4).json()["what_if"]
    route = w["what_if_route"]
    for key in ("coordinates", "node_count", "length_km", "total_cost", "nodes_expanded",
                "average_hazard", "maximum_hazard", "category_counts"):
        assert route[key] == normal_body["weather_route"][key], key
    assert w["comparison"]["routes_identical"] is True
    assert w["comparison"]["extra_distance_km"] == 0
    assert w["baseline_route"]["weather_cost"] == pytest.approx(
        normal_body["comparison"]["weather_route_weather_cost"])


@needs_data
def test_k0_what_if_route_is_the_distance_route(normal_body):
    w = post(hazard_penalty=0).json()["what_if"]
    assert w["what_if_route"]["coordinates"] == normal_body["distance_route"]["coordinates"]
    assert w["what_if_route"]["total_cost"] == pytest.approx(normal_body["distance_route"]["length_km"])
    assert w["what_if_route"]["weather_cost"] == pytest.approx(
        normal_body["comparison"]["distance_route_weather_cost"])
    c = w["comparison"]
    assert c["extra_distance_km"] == pytest.approx(
        normal_body["distance_route"]["length_km"] - normal_body["weather_route"]["length_km"])
    assert "K=0.0" in c["explanation"]


@needs_data
@pytest.mark.parametrize("valid_time", [F006, F024])
@pytest.mark.parametrize("k", [0, 10])
def test_what_if_matches_direct_python_on_selected_forecast(valid_time, k):
    body = post(forecast_valid_time=valid_time, hazard_penalty=k).json()
    assert pd.Timestamp(body["forecast"]["valid_time"]) == pd.Timestamp(valid_time)
    grid, d, w, x = direct(valid_time, k)
    route = body["what_if"]["what_if_route"]
    assert [(p["lat"], p["lon"]) for p in route["coordinates"]] == [grid.index_to_latlon(n) for n in x.nodes]
    assert route["total_cost"] == pytest.approx(x.total_cost)
    assert route["nodes_expanded"] == x.nodes_expanded
    c = compare_what_if(grid, w, x, k)
    api_c = body["what_if"]["comparison"]
    assert api_c["average_hazard_reduction_percent"] == pytest.approx(c.average_hazard_reduction_percent)
    assert api_c["high_hazard_cells_avoided"] == c.high_hazard_cells_avoided
    assert api_c["explanation"] == c.explanation
    # The normal sections still come from the unchanged pipeline.
    assert body["distance_route"]["total_cost"] == pytest.approx(d.total_cost)
    assert body["comparison"]["explanation"] == compare_routes(grid, d, w).explanation


@needs_data
def test_what_if_f024_differs_from_f006():
    a = post(forecast_valid_time=F006, hazard_penalty=0).json()["what_if"]
    b = post(forecast_valid_time=F024, hazard_penalty=0).json()["what_if"]
    assert a["baseline_route"] != b["baseline_route"]


@needs_data
def test_what_if_keeps_one_grid_per_forecast(monkeypatch):
    calls = []
    real = service.load_snapshot
    monkeypatch.setattr(service, "load_snapshot", lambda p: calls.append(p.name) or real(p))
    for vt, k in [(F006, 0), (F024, 10), (F006, 8), (F024, None), (F006, None), (F024, 0)]:
        extra = {"forecast_valid_time": vt}
        if k is not None:
            extra["hazard_penalty"] = k
        r = post(**extra)
        assert r.status_code == 200
        assert pd.Timestamp(r.json()["forecast"]["valid_time"]) == pd.Timestamp(vt)
    assert sorted(calls) == ["gfs.t12z.pgrb2.0p25.f006.hazard.nc", "gfs.t12z.pgrb2.0p25.f024.hazard.nc"]


@needs_data
def test_what_if_with_unknown_forecast_still_forecast_not_found():
    r = post(forecast_valid_time="2026-10-24T00:00:00Z", hazard_penalty=8)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "forecast_not_found"


@needs_data
def test_what_if_response_serializes_strictly():
    body = post(hazard_penalty=10).json()
    json.loads(json.dumps(body, allow_nan=False))
    route = body["what_if"]["what_if_route"]
    assert all(set(p) == {"lat", "lon"} for p in route["coordinates"])
    assert sum(route["category_counts"].values()) == route["node_count"]
    assert set(body["what_if"]["comparison"]) == {
        "routes_identical", "extra_distance_km", "extra_distance_percent",
        "weather_cost_difference", "weather_cost_reduction_percent",
        "average_hazard_reduction", "average_hazard_reduction_percent",
        "maximum_hazard_reduction", "severe_cells_avoided", "high_hazard_cells_avoided",
        "explanation",
    }


RAW = '{"origin":{"lat":28.6139,"lon":77.209},"destination":{"lat":19.076,"lon":72.8777},"hazard_penalty":%s}'


@pytest.mark.parametrize("raw", ["-0.1", "-5", "10.0001", "11", "NaN", "Infinity", "-Infinity",
                                 "1e400", "true", "\"4\"", "[4]"])
def test_api_rejects_invalid_penalty(raw):
    r = client.post("/api/route", content=RAW % raw, headers={"Content-Type": "application/json"})
    assert r.status_code == 422, r.text
    err = r.json()["error"]
    assert err["code"] == "invalid_request"
    assert err["details"][0]["field"] == "hazard_penalty"
    assert "Traceback" not in r.text


@needs_data
@pytest.mark.parametrize("raw", ["0", "10", "4", "2.5", "null"])
def test_api_accepts_boundary_penalties(raw):
    r = client.post("/api/route", content=RAW % raw, headers={"Content-Type": "application/json"})
    assert r.status_code == 200, r.text
    assert r.json()["what_if"]["enabled"] is (raw != "null")


def test_openapi_documents_hazard_penalty():
    spec = client.get("/openapi.json").json()
    schemas = spec["components"]["schemas"]
    prop = schemas["RouteRequest"]["properties"]["hazard_penalty"]
    number = next(s for s in prop["anyOf"] if s.get("type") == "number")
    assert (number["minimum"], number["maximum"]) == (0.0, 10.0)
    assert "what_if" in schemas["RouteResponse"]["required"]
    for model in ("WhatIfInfo", "WhatIfRouteInfo", "WhatIfComparisonModel", "RouteMetricsInfo"):
        assert model in schemas
