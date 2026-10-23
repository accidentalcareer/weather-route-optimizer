"""Derive per-grid-cell upper-air wind and humidity fields from GFS data.

Input is a pressure-level Dataset from ``grib_reader.load_gfs_pressure_levels``
with variables ``u``, ``v`` (m/s) and ``r`` (%) on dims
``(isobaricInhPa, latitude, longitude)``.

Output is a Dataset on the original ``(latitude, longitude)`` grid with:

    u_300, v_300, u_400, v_400, rh_300, rh_400   raw fields at each level
    wind_speed_300, wind_speed_400               horizontal wind speed
    wind_shear                                   vertical shear, 300-400 hPa
    rh_mean                                      mean RH of the two levels

This layer only derives physical fields; it does not score hazards.
"""

from pathlib import Path

import numpy as np
import xarray as xr

from weather.grib_reader import (
    REQUIRED_VARIABLES,
    SPATIAL_DIMS,
    load_gfs_pressure_levels,
    processed_dir,
    select_level,
)

UPPER_LEVEL_HPA = 300
LOWER_LEVEL_HPA = 400

OUTPUT_VARIABLES = (
    "u_300", "v_300", "u_400", "v_400", "rh_300", "rh_400",
    "wind_speed_300", "wind_speed_400", "wind_shear", "rh_mean",
)


def validate_input(ds: xr.Dataset) -> None:
    """Check that ``ds`` has the variables, levels and coordinates we need."""
    missing = [v for v in REQUIRED_VARIABLES if v not in ds.data_vars]
    if missing:
        raise ValueError(f"Missing required variables: {missing}")
    for coord in (*SPATIAL_DIMS, "valid_time"):
        if coord not in ds.coords:
            raise ValueError(f"Missing coordinate: {coord}")
    if "isobaricInhPa" not in ds.coords:
        raise ValueError("Missing pressure-level coordinate: isobaricInhPa")
    levels = ds["isobaricInhPa"].values.tolist()
    for level in (UPPER_LEVEL_HPA, LOWER_LEVEL_HPA):
        if level not in levels:
            raise ValueError(f"Pressure level {level} hPa not found; available: {levels}")


def wind_speed(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    """Horizontal wind speed |V| = sqrt(u^2 + v^2), in the units of u and v."""
    return np.hypot(u, v)


def vertical_wind_shear(
    u_upper: xr.DataArray, v_upper: xr.DataArray,
    u_lower: xr.DataArray, v_lower: xr.DataArray,
) -> xr.DataArray:
    """Magnitude of the wind-vector difference between two levels.

    shear = sqrt((u_upper - u_lower)^2 + (v_upper - v_lower)^2)

    This is the bulk shear across the layer (m/s), not divided by layer depth.
    Using the vector difference (rather than the difference of speeds) also
    captures direction changes, which matter for clear-air turbulence.
    """
    return np.hypot(u_upper - u_lower, v_upper - v_lower)


def compute_derived_fields(ds: xr.Dataset) -> xr.Dataset:
    """Compute wind speed, shear and mean RH for every grid cell.

    Args:
        ds: pressure-level Dataset with ``u``, ``v``, ``r`` at 300 and 400 hPa.

    Returns:
        Dataset with ``OUTPUT_VARIABLES`` on dims ``(latitude, longitude)``,
        keeping the ``latitude``, ``longitude`` and ``valid_time`` coordinates.
    """
    validate_input(ds)

    # Select by coordinate value so the result doesn't depend on level ordering.
    upper = select_level(ds, UPPER_LEVEL_HPA)
    lower = select_level(ds, LOWER_LEVEL_HPA)

    out = xr.Dataset(
        {
            "u_300": upper["u"],
            "v_300": upper["v"],
            "u_400": lower["u"],
            "v_400": lower["v"],
            "rh_300": upper["r"],
            "rh_400": lower["r"],
            # A/B. Wind speed at each level.
            "wind_speed_300": wind_speed(upper["u"], upper["v"]),
            "wind_speed_400": wind_speed(lower["u"], lower["v"]),
            # C. Bulk vertical wind shear between 300 and 400 hPa.
            "wind_shear": vertical_wind_shear(upper["u"], upper["v"], lower["u"], lower["v"]),
            # D. Simple arithmetic mean of the two levels' relative humidity.
            "rh_mean": (upper["r"] + lower["r"]) / 2,
        }
    )

    units = {
        "u_300": "m s-1", "v_300": "m s-1", "u_400": "m s-1", "v_400": "m s-1",
        "rh_300": "%", "rh_400": "%",
        "wind_speed_300": "m s-1", "wind_speed_400": "m s-1",
        "wind_shear": "m s-1", "rh_mean": "%",
    }
    for name, unit in units.items():
        out[name].attrs = {"units": unit}
    out["wind_shear"].attrs["long_name"] = "Bulk vertical wind shear, 300-400 hPa"
    out.attrs = dict(ds.attrs)

    validate_output(out, ds)
    return out


def validate_output(out: xr.Dataset, source: xr.Dataset) -> None:
    """Check the derived fields keep the input grid and introduce no new NaNs."""
    expected_shape = tuple(source.sizes[d] for d in SPATIAL_DIMS)
    # Any cell that was NaN in an input field may legitimately be NaN in the output.
    input_nan = source[list(REQUIRED_VARIABLES)].to_array().isnull().any(
        dim=["variable", "isobaricInhPa"]
    )
    for name in OUTPUT_VARIABLES:
        da = out[name]
        if da.dims != SPATIAL_DIMS or da.shape != expected_shape:
            raise ValueError(
                f"{name}: expected dims {SPATIAL_DIMS} shape {expected_shape}, "
                f"got {da.dims} {da.shape}"
            )
        new_nan = int((da.isnull() & ~input_nan).sum())
        if new_nan:
            raise ValueError(f"{name}: {new_nan} NaN values introduced by preprocessing")


def save_processed(ds: xr.Dataset, path=None) -> Path:
    """Write the processed Dataset to NetCDF and return the file path.

    Defaults to ``<processed_dir>/<source_file>.nc``.
    """
    if path is None:
        path = processed_dir() / f"{ds.attrs.get('source_file', 'gfs_processed')}.nc"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # scipy writes NetCDF3 without extra dependencies.
    ds.to_netcdf(path, engine="scipy")
    return path


def preprocess_file(grib_path, save: bool = False) -> xr.Dataset:
    """Load a GFS GRIB2 file and return its derived fields, optionally saving them."""
    out = compute_derived_fields(load_gfs_pressure_levels(grib_path))
    if save:
        save_processed(out)
    return out
