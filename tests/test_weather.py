"""Tests for backend/weather preprocessing, using small synthetic datasets."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from weather import grib_reader, hazard_model, preprocessing  # noqa: F401
from weather.preprocessing import OUTPUT_VARIABLES, compute_derived_fields


def make_dataset(levels=(400.0, 300.0)):
    """A 2x3 grid where every value is easy to check by hand.

    Levels are listed 400-first (as cfgrib returns them) to make sure the
    code selects by value, not position.
    """
    lat = [10.0, 10.25]
    lon = [70.0, 70.25, 70.5]
    shape = (len(levels), len(lat), len(lon))
    values = {
        300.0: {"u": 3.0, "v": 4.0, "r": 80.0},   # speed 5
        400.0: {"u": 0.0, "v": 0.0, "r": 40.0},   # speed 0
    }
    data = {
        name: (
            ("isobaricInhPa", "latitude", "longitude"),
            np.stack([np.full(shape[1:], values[lev][name]) for lev in levels]),
        )
        for name in ("u", "v", "r")
    }
    ds = xr.Dataset(
        data,
        coords={
            "isobaricInhPa": list(levels),
            "latitude": lat,
            "longitude": lon,
            "time": pd.Timestamp("2030-01-01T00:00"),
            "valid_time": pd.Timestamp("2030-01-01T06:00"),
        },
    )
    # Vary one cell so the test isn't only checking constants.
    ds["u"].loc[{"isobaricInhPa": 300.0, "latitude": 10.25, "longitude": 70.5}] = 6.0
    ds["v"].loc[{"isobaricInhPa": 400.0, "latitude": 10.25, "longitude": 70.5}] = 12.0
    return ds


def test_derived_fields_values():
    out = compute_derived_fields(make_dataset())

    assert set(OUTPUT_VARIABLES) <= set(out.data_vars)
    for name in OUTPUT_VARIABLES:
        assert out[name].dims == ("latitude", "longitude")
        assert out[name].shape == (2, 3)

    corner = {"latitude": 10.0, "longitude": 70.0}
    assert out["wind_speed_300"].sel(corner) == pytest.approx(5.0)
    assert out["wind_speed_400"].sel(corner) == pytest.approx(0.0)
    assert out["wind_shear"].sel(corner) == pytest.approx(5.0)
    assert out["rh_mean"].sel(corner) == pytest.approx(60.0)

    # Varied cell: 300 hPa (6, 4), 400 hPa (0, 12).
    cell = {"latitude": 10.25, "longitude": 70.5}
    assert out["wind_speed_300"].sel(cell) == pytest.approx(np.hypot(6, 4))
    assert out["wind_speed_400"].sel(cell) == pytest.approx(12.0)
    assert out["wind_shear"].sel(cell) == pytest.approx(np.hypot(6, -8))  # = 10

    assert out["valid_time"].values == np.datetime64("2030-01-01T06:00")
    np.testing.assert_array_equal(out["latitude"], [10.0, 10.25])
    np.testing.assert_array_equal(out["longitude"], [70.0, 70.25, 70.5])


def test_level_order_does_not_matter():
    a = compute_derived_fields(make_dataset(levels=(400.0, 300.0)))
    b = compute_derived_fields(make_dataset(levels=(300.0, 400.0)))
    xr.testing.assert_allclose(a, b)


def test_missing_variable_rejected():
    with pytest.raises(ValueError, match="Missing required variables"):
        compute_derived_fields(make_dataset().drop_vars("r"))


def test_missing_level_rejected():
    ds = make_dataset().sel(isobaricInhPa=[300.0])
    with pytest.raises(ValueError, match="400 hPa not found"):
        compute_derived_fields(ds)


def test_input_nans_are_allowed_but_not_new_ones():
    ds = make_dataset()
    ds["r"].loc[{"isobaricInhPa": 300.0, "latitude": 10.0, "longitude": 70.0}] = np.nan
    out = compute_derived_fields(ds)
    assert np.isnan(out["rh_mean"].sel(latitude=10.0, longitude=70.0))
    assert int(out["rh_mean"].isnull().sum()) == 1
