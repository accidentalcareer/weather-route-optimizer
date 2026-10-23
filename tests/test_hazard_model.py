"""Tests for backend/weather/hazard_model.py, using small synthetic datasets."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from weather.hazard_model import (
    OUTPUT_VARIABLES,
    categorize,
    category_labels,
    compute_hazard,
    normalize_rh,
    robust_minmax,
    weather_multiplier,
    weighted_hazard,
)

LAT = [10.0, 10.25, 10.5, 10.75]
LON = [70.0, 70.25, 70.5, 70.75, 71.0]


def grid(values):
    return xr.DataArray(
        np.asarray(values, dtype=float).reshape(len(LAT), len(LON)),
        dims=("latitude", "longitude"),
        coords={"latitude": LAT, "longitude": LON},
    )


def make_processed():
    """4x5 processed dataset; values 0..19 make percentiles easy to reason about."""
    ramp = np.arange(20.0)
    ds = xr.Dataset(
        {
            "wind_speed_300": grid(ramp * 2),     # 0..38 m/s
            "wind_shear": grid(ramp),             # 0..19 m/s
            "rh_mean": grid(ramp * 5),            # 0..95 %
        },
        coords={"valid_time": pd.Timestamp("2030-01-01T06:00")},
    )
    return ds


# --- normalization -------------------------------------------------------

def test_robust_minmax_with_fixed_bounds():
    norm, bounds = robust_minmax(grid(np.arange(20.0)), bounds=(5.0, 15.0))
    assert bounds == (5.0, 15.0)
    assert norm.sel(latitude=10.0, longitude=70.0) == 0.0     # 0 -> clipped to 0
    assert norm.sel(latitude=10.25, longitude=70.0) == 0.0    # 5 -> 0
    assert norm.sel(latitude=10.5, longitude=70.0) == 0.5     # 10 -> 0.5
    assert norm.sel(latitude=10.75, longitude=71.0) == 1.0    # 19 -> clipped to 1


def test_robust_minmax_uses_percentiles():
    da = grid(np.arange(20.0))
    norm, (lo, hi) = robust_minmax(da)
    assert lo == pytest.approx(np.quantile(np.arange(20.0), 0.05))
    assert hi == pytest.approx(np.quantile(np.arange(20.0), 0.95))
    assert float(norm.min()) == 0.0 and float(norm.max()) == 1.0


def test_robust_minmax_constant_field_is_zero():
    norm, _ = robust_minmax(grid(np.full(20, 7.0)))
    assert float(norm.max()) == 0.0


def test_normalize_rh_physical_range():
    rh = grid(np.linspace(0, 101, 20))  # includes a slightly supersaturated value
    norm = normalize_rh(rh)
    assert norm.sel(latitude=10.0, longitude=70.0) == 0.0
    assert float(norm.max()) == 1.0
    assert normalize_rh(grid(np.full(20, 50.0))).values == pytest.approx(0.5)


# --- weighted score, categories, multiplier ------------------------------

def test_weighted_hazard():
    w, s, r = grid(np.full(20, 1.0)), grid(np.full(20, 0.5)), grid(np.full(20, 0.25))
    # 0.30*1 + 0.50*0.5 + 0.20*0.25 = 0.60
    assert weighted_hazard(w, s, r).values == pytest.approx(0.60)
    ones = grid(np.ones(20))
    assert weighted_hazard(ones, ones, ones).values == pytest.approx(1.0)


def test_weights_must_sum_to_one():
    ones = grid(np.ones(20))
    with pytest.raises(ValueError, match="sum to 1"):
        weighted_hazard(ones, ones, ones, weights={"wind": 0.5, "shear": 0.5, "rh": 0.5})


@pytest.mark.parametrize(
    "score, label",
    [
        (0.0, "LOW"), (0.2499, "LOW"),
        (0.25, "MODERATE"), (0.4999, "MODERATE"),
        (0.50, "HIGH"), (0.7499, "HIGH"),
        (0.75, "SEVERE"), (1.0, "SEVERE"),
    ],
)
def test_category_boundaries(score, label):
    da = xr.DataArray([score], dims="x")
    assert category_labels(categorize(da)).item() == label


def test_nan_score_gets_missing_category():
    cats = categorize(xr.DataArray([np.nan, 0.1], dims="x"))
    assert cats.values.tolist() == [-1, 0]
    assert category_labels(cats).values.tolist() == ["MISSING", "LOW"]


@pytest.mark.parametrize("score, mult", [(0.0, 1.0), (0.5, 3.0), (1.0, 5.0), (0.25, 2.0)])
def test_weather_multiplier(score, mult):
    assert weather_multiplier(xr.DataArray(score)).item() == pytest.approx(mult)


# --- full pipeline -------------------------------------------------------

def test_compute_hazard_end_to_end():
    ds = make_processed()
    out = compute_hazard(ds)

    for name in OUTPUT_VARIABLES:
        assert out[name].dims == ("latitude", "longitude")
        assert out[name].shape == (4, 5)
    np.testing.assert_array_equal(out["latitude"], LAT)
    np.testing.assert_array_equal(out["longitude"], LON)
    assert out["valid_time"].values == np.datetime64("2030-01-01T06:00")

    for name in ("normalized_wind", "normalized_shear", "normalized_rh", "hazard_score"):
        assert 0.0 <= float(out[name].min()) and float(out[name].max()) <= 1.0
    assert 1.0 <= float(out["weather_multiplier"].min())
    assert float(out["weather_multiplier"].max()) <= 5.0

    # Recompute one cell by hand from the stored bounds.
    cell = {"latitude": 10.5, "longitude": 70.5}  # ramp index 12
    wb = (out["normalized_wind"].attrs["lower_bound"], out["normalized_wind"].attrs["upper_bound"])
    sb = (out["normalized_shear"].attrs["lower_bound"], out["normalized_shear"].attrs["upper_bound"])
    nw = np.clip((24.0 - wb[0]) / (wb[1] - wb[0]), 0, 1)
    ns = np.clip((12.0 - sb[0]) / (sb[1] - sb[0]), 0, 1)
    nr = 60.0 / 100
    expected = 0.30 * nw + 0.50 * ns + 0.20 * nr
    assert out["hazard_score"].sel(cell) == pytest.approx(expected)
    assert out["weather_multiplier"].sel(cell) == pytest.approx(1 + 4 * expected)


def test_compute_hazard_extreme_cells():
    out = compute_hazard(make_processed())
    calm = {"latitude": 10.0, "longitude": 70.0}     # all inputs at minimum
    rough = {"latitude": 10.75, "longitude": 71.0}   # all inputs at maximum
    assert out["hazard_score"].sel(calm) == pytest.approx(0.0)
    assert category_labels(out["hazard_category"]).sel(calm) == "LOW"
    # wind and shear clip to 1, RH = 95 % -> 0.30 + 0.50 + 0.20 * 0.95 = 0.99
    assert out["hazard_score"].sel(rough) == pytest.approx(0.99)
    assert category_labels(out["hazard_category"]).sel(rough) == "SEVERE"


def test_missing_input_rejected():
    with pytest.raises(ValueError, match="Missing required input variables"):
        compute_hazard(make_processed().drop_vars("wind_shear"))


def test_input_nans_propagate_without_new_ones():
    ds = make_processed()
    ds["rh_mean"].loc[{"latitude": 10.0, "longitude": 70.0}] = np.nan
    out = compute_hazard(ds)
    assert int(out["hazard_score"].isnull().sum()) == 1
    assert int((out["hazard_category"] == -1).sum()) == 1
