"""Tests for the FastAPI app, GET /api/forecasts and POST /api/route.

Route tests use the real processed hazard file and are skipped if it is
absent (backend/data/processed is git-ignored).
"""

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from fastapi.testclient import TestClient

from app import app
from routing import service
from routing.astar import RouteNotFoundError, astar
from routing.cost import distance_cost, weather_cost
from routing.grid import load_hazard_grid
from routing.metrics import compare_routes

client = TestClient(app)

DELHI = {"lat": 28.6139, "lon": 77.2090}
MUMBAI = {"lat": 19.0760, "lon": 72.8777}


def has_hazard_file():
    try:
        return service.default_hazard_path().is_file()
    except service.WeatherDataError:
        return False


needs_data = pytest.mark.skipif(not has_hazard_file(), reason="processed hazard file not available")


@pytest.fixture(autouse=True)
def reset_cache():
    """Each test starts with (and leaves) an empty grid cache."""
    service.clear_hazard_cache()
    yield
    service.clear_hazard_cache()


def post_route(origin=DELHI, destination=MUMBAI):
    return client.post("/api/route", json={"origin": origin, "destination": destination})


def assert_error(response, status, code):
    assert response.status_code == status, response.text
    body = response.json()
    assert body["error"]["code"] == code
    assert body["error"]["message"]
    assert "Traceback" not in response.text
    return body["error"]


# --- health -------------------------------------------------------------

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "weather-route-optimizer"}


# --- successful route ---------------------------------------------------

@pytest.fixture(scope="module")
def delhi_mumbai():
    if not has_hazard_file():
        pytest.skip("processed hazard file not available")
    service.clear_hazard_cache()
    response = post_route()
    assert response.status_code == 200, response.text
    return response.json()


@needs_data
def test_response_contains_both_routes(delhi_mumbai):
    for key in ("distance_route", "weather_route"):
        r = delhi_mumbai[key]
        assert r["node_count"] == len(r["coordinates"]) >= 2
        assert r["length_km"] > 0 and r["total_cost"] > 0 and r["nodes_expanded"] > 0
        assert set(r["category_counts"]) == {"LOW", "MODERATE", "HIGH", "SEVERE"}
        assert sum(r["category_counts"].values()) == r["node_count"]


@needs_data
def test_routes_start_and_end_at_snapped_points(delhi_mumbai):
    for key in ("distance_route", "weather_route"):
        coords = delhi_mumbai[key]["coordinates"]
        assert coords[0] == delhi_mumbai["origin"]["snapped"]
        assert coords[-1] == delhi_mumbai["destination"]["snapped"]


@needs_data
def test_coordinates_are_plain_lat_lon_objects(delhi_mumbai):
    for c in delhi_mumbai["weather_route"]["coordinates"]:
        assert set(c) == {"lat", "lon"}
        assert isinstance(c["lat"], float) and isinstance(c["lon"], float)
    # Round-trips through strict JSON (no NaN/Infinity, no NumPy types).
    json.loads(json.dumps(delhi_mumbai, allow_nan=False))


@needs_data
def test_snapped_coordinates_returned(delhi_mumbai):
    o = delhi_mumbai["origin"]
    assert o["requested"] == DELHI
    assert o["snapped"] == {"lat": 28.5, "lon": 77.25}
    assert 0 < o["snap_distance_km"] < 20
    assert delhi_mumbai["destination"]["requested"] == MUMBAI
    assert delhi_mumbai["destination"]["snapped"] == {"lat": 19.0, "lon": 73.0}


@needs_data
def test_forecast_valid_time_returned(delhi_mumbai):
    f = delhi_mumbai["forecast"]
    assert pd.Timestamp(f["valid_time"]) == pd.Timestamp("2026-10-22T18:00:00Z")
    assert f["source_file"].endswith(".hazard.nc")
    assert f["grid_bounds"] == {"lat_min": 5.0, "lat_max": 35.0, "lon_min": 65.0, "lon_max": 95.0}


@needs_data
def test_comparison_metrics_present(delhi_mumbai):
    c = delhi_mumbai["comparison"]
    for key in ("extra_distance_km", "extra_distance_percent", "weather_cost_difference",
                "weather_cost_reduction_percent", "average_hazard_distance_route",
                "average_hazard_weather_route", "maximum_hazard_distance_route",
                "maximum_hazard_weather_route", "severe_cells_avoided",
                "high_hazard_cells_avoided", "estimated_fuel_delta_percent", "explanation"):
        assert key in c
    assert "not an" in c["fuel_proxy_note"].lower()


@needs_data
def test_api_matches_direct_python(delhi_mumbai):
    grid, _ = load_hazard_grid(service.default_hazard_path())
    s = grid.nearest_node(DELHI["lat"], DELHI["lon"])
    t = grid.nearest_node(MUMBAI["lat"], MUMBAI["lon"])
    d, w = astar(grid, s, t, distance_cost), astar(grid, s, t, weather_cost)
    c = compare_routes(grid, d, w)

    api_d, api_w, api_c = (delhi_mumbai[k] for k in ("distance_route", "weather_route", "comparison"))
    assert api_d["length_km"] == pytest.approx(d.path_length_km)
    assert api_w["length_km"] == pytest.approx(w.path_length_km)
    assert api_d["total_cost"] == pytest.approx(d.total_cost)
    assert api_w["total_cost"] == pytest.approx(w.total_cost)
    assert api_d["nodes_expanded"] == d.nodes_expanded
    assert api_w["nodes_expanded"] == w.nodes_expanded
    assert [(p["lat"], p["lon"]) for p in api_w["coordinates"]] == [grid.index_to_latlon(n) for n in w.nodes]
    assert api_c["weather_cost_reduction_percent"] == pytest.approx(c.weather_cost_reduction_percent)
    assert api_c["severe_cells_avoided"] == c.severe_cells_avoided
    assert api_c["explanation"] == c.explanation


@needs_data
def test_grid_loaded_once_across_requests(monkeypatch):
    calls = []
    real = service.load_snapshot
    monkeypatch.setattr(service, "load_snapshot", lambda p: calls.append(p) or real(p))
    assert post_route().status_code == 200
    assert post_route(MUMBAI, DELHI).status_code == 200
    assert len(calls) == 1


# --- forecast selection (real data) -------------------------------------

REAL_FORECASTS = [
    ("2026-10-22T18:00:00Z", 6.0, "gfs.t12z.pgrb2.0p25.f006.hazard.nc"),
    ("2026-10-23T00:00:00Z", 12.0, "gfs.t12z.pgrb2.0p25.f012.hazard.nc"),
    ("2026-10-23T06:00:00Z", 18.0, "gfs.t12z.pgrb2.0p25.f018.hazard.nc"),
    ("2026-10-23T12:00:00Z", 24.0, "gfs.t12z.pgrb2.0p25.f024.hazard.nc"),
]


def all_real_forecasts_present():
    names = {p.name for p in service.processed_dir().glob("*.hazard.nc")}
    return {name for *_, name in REAL_FORECASTS} <= names


needs_all_forecasts = pytest.mark.skipif(
    not all_real_forecasts_present(), reason="the four GFS 12Z hazard files are not available"
)


@needs_all_forecasts
def test_forecasts_endpoint_lists_real_files():
    response = client.get("/api/forecasts")
    assert response.status_code == 200, response.text
    body = response.json()
    assert pd.Timestamp(body["default_valid_time"]) == pd.Timestamp("2026-10-22T18:00:00Z")
    got = [(pd.Timestamp(f["valid_time"]), f["step_hours"], f["source_file"]) for f in body["forecasts"]]
    assert got == [(pd.Timestamp(t), step, name) for t, step, name in REAL_FORECASTS]
    for f in body["forecasts"]:
        assert pd.Timestamp(f["reference_time"]) == pd.Timestamp("2026-10-22T12:00:00Z")
        assert f["grib_file"] == f["source_file"].removesuffix(".hazard.nc")
    assert [f["is_default"] for f in body["forecasts"]] == [True, False, False, False]


@needs_all_forecasts
@pytest.mark.parametrize("valid_time, step, name", REAL_FORECASTS)
def test_route_with_selected_real_forecast(valid_time, step, name):
    response = client.post("/api/route", json={
        "origin": DELHI, "destination": MUMBAI, "forecast_valid_time": valid_time,
    })
    assert response.status_code == 200, response.text
    f = response.json()["forecast"]
    assert pd.Timestamp(f["valid_time"]) == pd.Timestamp(valid_time)
    assert f["source_file"] == name
    assert f["step_hours"] == step


@needs_all_forecasts
def test_selected_forecast_matches_direct_python():
    """A non-default forecast is routed on its own grid, through the same A*/metrics."""
    valid_time, _, name = REAL_FORECASTS[2]
    body = client.post("/api/route", json={
        "origin": DELHI, "destination": MUMBAI, "forecast_valid_time": valid_time,
    }).json()
    grid, _ = load_hazard_grid(service.processed_dir() / name)
    s = grid.nearest_node(DELHI["lat"], DELHI["lon"])
    t = grid.nearest_node(MUMBAI["lat"], MUMBAI["lon"])
    w = astar(grid, s, t, weather_cost)
    c = compare_routes(grid, astar(grid, s, t, distance_cost), w)
    assert body["weather_route"]["total_cost"] == pytest.approx(w.total_cost)
    assert [(p["lat"], p["lon"]) for p in body["weather_route"]["coordinates"]] == [
        grid.index_to_latlon(n) for n in w.nodes]
    assert body["comparison"]["weather_cost_reduction_percent"] == pytest.approx(
        c.weather_cost_reduction_percent)


@needs_all_forecasts
def test_each_forecast_loaded_once(monkeypatch):
    calls = []
    real = service.load_snapshot
    monkeypatch.setattr(service, "load_snapshot", lambda p: calls.append(p.name) or real(p))
    for valid_time, *_ in REAL_FORECASTS * 2:
        assert client.post("/api/route", json={
            "origin": DELHI, "destination": MUMBAI, "forecast_valid_time": valid_time,
        }).status_code == 200
    assert post_route().status_code == 200  # default shares the f006 entry
    assert sorted(calls) == sorted(name for *_, name in REAL_FORECASTS)


@needs_all_forecasts
def test_unknown_real_forecast_rejected():
    response = client.post("/api/route", json={
        "origin": DELHI, "destination": MUMBAI, "forecast_valid_time": "2026-10-22T19:00:00Z",
    })
    err = assert_error(response, 422, "forecast_not_found")
    assert len(err["details"]) == 4


# --- request validation -------------------------------------------------

@pytest.mark.parametrize("payload", [
    {},
    {"origin": DELHI},
    {"origin": {"lat": 28.6}, "destination": MUMBAI},
    {"origin": {"lat": "north", "lon": 77.2}, "destination": MUMBAI},
    {"origin": DELHI, "destination": MUMBAI, "departure_time": "now"},
])
def test_malformed_request_rejected(payload):
    assert_error(client.post("/api/route", json=payload), 422, "invalid_request")


@pytest.mark.parametrize("lat", [90.01, -91, 1e9])
def test_invalid_latitude(lat):
    err = assert_error(post_route(origin={"lat": lat, "lon": 77.2}), 422, "invalid_request")
    assert err["details"][0]["field"] == "origin.lat"


@pytest.mark.parametrize("lon", [180.5, -181, 360])
def test_invalid_longitude(lon):
    err = assert_error(post_route(destination={"lat": 19.0, "lon": lon}), 422, "invalid_request")
    assert err["details"][0]["field"] == "destination.lon"


# --- locations outside the grid -----------------------------------------

@needs_data
def test_origin_outside_grid():
    err = assert_error(post_route(origin={"lat": 51.5, "lon": -0.12}), 422, "outside_grid")
    assert err["message"].startswith("Origin")
    assert "forecast area" in err["message"]


@needs_data
def test_destination_outside_grid():
    err = assert_error(post_route(destination={"lat": 19.0, "lon": 100.0}), 422, "outside_grid")
    assert err["message"].startswith("Destination")


@needs_data
def test_same_grid_point_rejected():
    assert_error(post_route(DELHI, {"lat": 28.62, "lon": 77.21}), 422, "same_grid_point")


# --- routing and data errors --------------------------------------------

@needs_data
def test_route_not_found(monkeypatch):
    def no_route(*args, **kwargs):
        raise RouteNotFoundError("No route from (94, 49) to (56, 32)")

    monkeypatch.setattr(service, "astar", no_route)
    assert_error(post_route(), 422, "route_not_found")


def test_missing_hazard_file(monkeypatch, tmp_path):
    monkeypatch.setenv(service.HAZARD_FILE_ENV, str(tmp_path / "missing.hazard.nc"))
    err = assert_error(post_route(), 503, "weather_data_unavailable")
    assert "not found" in err["message"]


def test_invalid_weather_data(monkeypatch, tmp_path):
    hazard = np.full((3, 3), 0.5)
    hazard[1, 1] = np.nan
    ds = xr.Dataset(
        {"hazard_score": (("latitude", "longitude"), hazard)},
        coords={"latitude": [20.0, 20.25, 20.5], "longitude": [75.0, 75.25, 75.5],
                "valid_time": pd.Timestamp("2030-01-01T06:00")},
    )
    path = tmp_path / "bad.hazard.nc"
    ds.to_netcdf(path, engine="scipy")
    monkeypatch.setenv(service.HAZARD_FILE_ENV, str(path))
    err = assert_error(post_route({"lat": 20.0, "lon": 75.0}, {"lat": 20.5, "lon": 75.5}),
                       503, "weather_data_unavailable")
    assert "non-finite" in err["message"]


@needs_data
def test_unexpected_error_hides_internals(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(service, "compare_routes", boom)
    err = assert_error(post_route(), 500, "internal_error")
    assert "secret" not in err["message"]


# --- cache refresh, CORS, docs ------------------------------------------

@needs_data
def test_refresh_replaces_snapshot():
    first = service.get_hazard_snapshot()
    second = service.refresh_hazard_grid()
    assert second is not first
    assert service.get_hazard_snapshot() is second


def test_failed_refresh_keeps_previous_snapshot(tmp_path):
    if not has_hazard_file():
        pytest.skip("processed hazard file not available")
    before = service.get_hazard_snapshot()
    with pytest.raises(service.WeatherDataError):
        service.refresh_hazard_grid(tmp_path / "nope.hazard.nc")
    assert service.get_hazard_snapshot() is before


def test_cors_allows_dev_frontend():
    response = client.options(
        "/api/route",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"},
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_cors_rejects_unknown_origin():
    response = client.options(
        "/api/route",
        headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in response.headers


def test_openapi_documents_route_endpoint():
    spec = client.get("/openapi.json").json()
    op = spec["paths"]["/api/route"]["post"]
    assert op["requestBody"]["content"]["application/json"]["schema"]["$ref"].endswith("/RouteRequest")
    assert op["responses"]["200"]["content"]["application/json"]["schema"]["$ref"].endswith("/RouteResponse")
    assert {"422", "503", "500"} <= set(op["responses"])
    for model in ("RouteRequest", "RouteResponse", "RouteInfo", "RouteComparisonModel", "Coordinate"):
        assert model in spec["components"]["schemas"]
    assert "forecast_valid_time" in spec["components"]["schemas"]["RouteRequest"]["properties"]


def test_openapi_documents_forecasts_endpoint():
    spec = client.get("/openapi.json").json()
    op = spec["paths"]["/api/forecasts"]["get"]
    assert op["responses"]["200"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "/ForecastListResponse")
    assert "503" in op["responses"]
