"""Tests for backend/routing/cost.py, using small synthetic grids."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from routing.cost import distance_cost, edge_cost_details, edge_multiplier, weather_cost
from routing.grid import GeoGrid, haversine_km
from weather import hazard_model

LATS = np.array([10.0, 10.25, 10.5])
LONS = np.array([70.0, 70.25, 70.5, 70.75])


def make_grid(hazard=None):
    if hazard is None:
        # Distinct value per node: 0.0, 0.05, ..., 0.55.
        hazard = np.arange(12).reshape(3, 4) * 0.05
    return GeoGrid(LATS, LONS, hazard=hazard)


# --- distance cost ------------------------------------------------------

def test_distance_cost_equals_haversine():
    g = make_grid()
    for b in [(0, 1), (1, 0), (1, 1)]:
        expected = haversine_km(*g.index_to_latlon((0, 0)), *g.index_to_latlon(b))
        assert distance_cost(g, (0, 0), b) == pytest.approx(expected)


def test_distance_cost_ignores_weather():
    calm = make_grid(np.zeros((3, 4)))
    stormy = make_grid(np.ones((3, 4)))
    no_weather = GeoGrid(LATS, LONS)
    for b in [(0, 1), (1, 1), (2, 3)]:
        d = distance_cost(no_weather, (1, 2), b)
        assert distance_cost(calm, (1, 2), b) == d
        assert distance_cost(stormy, (1, 2), b) == d


# --- weather cost -------------------------------------------------------

def test_edge_hazard_is_mean_of_endpoints():
    g = make_grid()
    d = edge_cost_details(g, (0, 1), (1, 2))
    assert d["hazard_a"] == pytest.approx(0.05)
    assert d["hazard_b"] == pytest.approx(0.30)
    assert d["edge_hazard"] == pytest.approx(0.175)


@pytest.mark.parametrize("hazard, mult", [(0.0, 1.0), (0.5, 3.0), (1.0, 5.0), (0.25, 2.0)])
def test_edge_multiplier(hazard, mult):
    assert edge_multiplier(hazard) == pytest.approx(mult)


@pytest.mark.parametrize("value, mult", [(0.0, 1.0), (0.5, 3.0), (1.0, 5.0)])
def test_uniform_hazard_multiplier_and_cost(value, mult):
    g = make_grid(np.full((3, 4), value))
    d = edge_cost_details(g, (1, 1), (1, 2))
    assert d["weather_multiplier"] == pytest.approx(mult)
    assert d["weather_cost_km"] == pytest.approx(mult * d["distance_km"])
    if value == 0.0:
        assert weather_cost(g, (1, 1), (1, 2)) == distance_cost(g, (1, 1), (1, 2))


def test_weather_cost_calculation():
    g = make_grid()
    d = edge_cost_details(g, (0, 0), (1, 1))  # hazards 0.0 and 0.25
    assert d["edge_hazard"] == pytest.approx(0.125)
    assert d["weather_multiplier"] == pytest.approx(1.5)
    assert d["weather_cost_km"] == pytest.approx(1.5 * d["distance_km"])
    assert weather_cost(g, (0, 0), (1, 1)) == pytest.approx(d["weather_cost_km"])


def test_details_contain_required_keys():
    d = edge_cost_details(make_grid(), (0, 0), (0, 1))
    for key in ("distance_km", "hazard_a", "hazard_b", "edge_hazard",
                "weather_multiplier", "weather_cost_km"):
        assert key in d


def test_weather_cost_at_least_distance_cost():
    rng = np.random.default_rng(0)
    g = make_grid(rng.uniform(0, 1, (3, 4)))
    for i in range(3):
        for j in range(4):
            for nb in g.neighbors((i, j)):
                assert weather_cost(g, (i, j), nb) >= distance_cost(g, (i, j), nb)


def test_cost_is_symmetric():
    g = make_grid()
    ab = edge_cost_details(g, (0, 1), (1, 2))
    ba = edge_cost_details(g, (1, 2), (0, 1))
    assert ab["hazard_a"] == ba["hazard_b"] and ab["hazard_b"] == ba["hazard_a"]
    assert ab["edge_hazard"] == pytest.approx(ba["edge_hazard"])
    assert ab["distance_km"] == pytest.approx(ba["distance_km"])
    assert ab["weather_cost_km"] == pytest.approx(ba["weather_cost_km"])


# --- validation ---------------------------------------------------------

@pytest.mark.parametrize("bad", [(-1, 0), (3, 0), (0, 4), (99, 99)])
def test_invalid_node_raises(bad):
    g = make_grid()
    with pytest.raises(IndexError, match="Invalid edge"):
        distance_cost(g, (0, 0), bad)
    with pytest.raises(IndexError, match="Invalid edge"):
        weather_cost(g, bad, (0, 0))


def test_same_node_raises():
    g = make_grid()
    with pytest.raises(ValueError, match="distinct"):
        distance_cost(g, (1, 1), (1, 1))
    with pytest.raises(ValueError, match="distinct"):
        weather_cost(g, (1, 1), (1, 1))


@pytest.mark.parametrize("bad_value, msg", [(np.nan, "not finite"), (np.inf, "not finite"),
                                            (1.2, "outside"), (-0.1, "outside")])
def test_bad_hazard_values_raise(bad_value, msg):
    hazard = np.zeros((3, 4))
    hazard[0, 1] = bad_value
    g = make_grid(hazard)
    with pytest.raises(ValueError, match=msg):
        weather_cost(g, (0, 0), (0, 1))


def test_weather_cost_requires_hazard_field():
    g = GeoGrid(LATS, LONS)
    with pytest.raises(ValueError, match="no hazard field"):
        weather_cost(g, (0, 0), (0, 1))


def test_hazard_shape_must_match_grid():
    with pytest.raises(ValueError, match="does not match grid"):
        GeoGrid(LATS, LONS, hazard=np.zeros((4, 3)))


# --- multiplier consistency with the hazard model -----------------------

def test_multiplier_constants_are_single_source():
    from routing import cost

    assert cost.MULTIPLIER_MIN is hazard_model.MULTIPLIER_MIN
    assert cost.MULTIPLIER_RANGE is hazard_model.MULTIPLIER_RANGE
    assert cost.weather_multiplier is hazard_model.weather_multiplier


def test_edge_multiplier_matches_hazard_model_exactly():
    for h in np.linspace(0.0, 1.0, 101):
        assert edge_multiplier(h) == hazard_model.weather_multiplier(h)


def test_edge_multiplier_matches_stored_cell_multiplier():
    # Run the real hazard pipeline on a small processed dataset, then check
    # each edge's multiplier equals the mean of its endpoints' stored
    # weather_multiplier (they agree because the formula is linear).
    rng = np.random.default_rng(3)
    coords = {"latitude": LATS, "longitude": LONS, "valid_time": pd.Timestamp("2030-01-01")}
    dims = ("latitude", "longitude")
    ds = xr.Dataset(
        {
            "wind_speed_300": (dims, rng.uniform(0, 50, (3, 4))),
            "wind_shear": (dims, rng.uniform(0, 25, (3, 4))),
            "rh_mean": (dims, rng.uniform(0, 100, (3, 4))),
        },
        coords=coords,
    )
    out = hazard_model.compute_hazard(ds)
    g = GeoGrid.from_dataset(out)
    cell_mult = out["weather_multiplier"].values
    for i in range(3):
        for j in range(4):
            for nb in g.neighbors((i, j)):
                d = edge_cost_details(g, (i, j), nb)
                assert d["weather_multiplier"] == pytest.approx(
                    (cell_mult[i, j] + cell_mult[nb]) / 2, rel=1e-12
                )
