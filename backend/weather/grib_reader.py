"""Read regional NOAA GFS GRIB2 files into in-memory xarray datasets.

The reader keeps only the isobaric (pressure-level) hypercube, which is where
GFS stores upper-air wind and humidity. Filtering on ``typeOfLevel`` means the
same function works on a full GFS ``pgrb2`` file, which mixes many level types
that cfgrib cannot merge into a single dataset, as well as on regional subsets
like ``gfs.t12z.pgrb2.0p25.f006``.

All selection is done by coordinate value (e.g. ``isobaricInhPa=300``), never by
array position, because level and latitude ordering differ between files.
"""

import os
from pathlib import Path

import xarray as xr

BACKEND_DIR = Path(__file__).resolve().parents[1]

# GRIB shortNames needed by preprocessing: u/v wind components and relative humidity.
REQUIRED_VARIABLES = ("u", "v", "r")
LEVEL_DIM = "isobaricInhPa"
SPATIAL_DIMS = ("latitude", "longitude")


def _data_dir(env_var: str, default: str) -> Path:
    """Resolve a data directory from the environment, relative to backend/."""
    path = Path(os.getenv(env_var, default))
    return path if path.is_absolute() else BACKEND_DIR / path


def raw_dir() -> Path:
    return _data_dir("WEATHER_RAW_DIR", "data/raw")


def processed_dir() -> Path:
    return _data_dir("WEATHER_PROCESSED_DIR", "data/processed")


def load_gfs_pressure_levels(path, variables=REQUIRED_VARIABLES) -> xr.Dataset:
    """Load pressure-level fields from a GFS GRIB2 file.

    Args:
        path: GRIB2 file path. A bare filename is looked up in ``raw_dir()``.
        variables: GRIB shortNames to keep.

    Returns:
        A loaded Dataset with dims ``(isobaricInhPa, latitude, longitude)`` and
        the ``time`` (reference), ``step`` and ``valid_time`` coordinates.

    Raises:
        FileNotFoundError: the file does not exist.
        ValueError: a requested variable or a required coordinate is missing.
    """
    path = Path(path)
    if not path.is_file() and not path.is_absolute():
        path = raw_dir() / path
    if not path.is_file():
        raise FileNotFoundError(f"GRIB file not found: {path}")

    ds = xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={
            "indexpath": "",  # don't write .idx sidecar files into data/raw
            "filter_by_keys": {"typeOfLevel": LEVEL_DIM},
        },
    )

    missing = [v for v in variables if v not in ds.data_vars]
    if missing:
        raise ValueError(
            f"{path.name}: missing variables {missing}; found {list(ds.data_vars)}"
        )
    missing_coords = [c for c in (LEVEL_DIM, *SPATIAL_DIMS, "valid_time") if c not in ds.coords]
    if missing_coords:
        raise ValueError(f"{path.name}: missing coordinates {missing_coords}")

    # Load into memory so the file handle is released and later maths is eager.
    ds = ds[list(variables)].load()
    ds.attrs["source_file"] = path.name
    return ds


def select_level(ds: xr.Dataset, level_hpa: float) -> xr.Dataset:
    """Return the 2D (latitude, longitude) slice at one pressure level.

    Raises:
        ValueError: the level is not present in the dataset.
    """
    available = ds[LEVEL_DIM].values.tolist()
    if level_hpa not in available:
        raise ValueError(f"Pressure level {level_hpa} hPa not found; available: {available}")
    # Drop the now-scalar level coordinate so slices from different levels combine cleanly.
    return ds.sel({LEVEL_DIM: level_hpa}).drop_vars(LEVEL_DIM)
