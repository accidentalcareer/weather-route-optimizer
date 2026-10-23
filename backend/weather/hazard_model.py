"""Map preprocessed weather fields to a per-cell experimental hazard score.

This is an experimental academic routing cost, NOT an operational aviation
turbulence forecast. It contains no aviation safety thresholds: every input is
rescaled from the data itself (or from its physical range), so the score ranks
cells within one forecast as relatively calmer or rougher.

Pipeline, per grid cell:

    normalized_wind  = robust_minmax(wind_speed_300)   # 5th-95th percentile
    normalized_shear = robust_minmax(wind_shear)       # 5th-95th percentile
    normalized_rh    = rh_mean / 100                   # physical 0-100 % range

    hazard_score       = 0.30 * normalized_wind
                       + 0.50 * normalized_shear
                       + 0.20 * normalized_rh          # in [0, 1]
    hazard_category    = LOW / MODERATE / HIGH / SEVERE (see categorize)
    weather_multiplier = 1 + 4 * hazard_score          # in [1, 5]

Input is the Dataset from ``preprocessing.compute_derived_fields`` (or the
NetCDF it saves).
"""

from pathlib import Path

import numpy as np
import xarray as xr

from weather.grib_reader import SPATIAL_DIMS, processed_dir

REQUIRED_INPUTS = ("wind_speed_300", "wind_shear", "rh_mean")

WEIGHTS = {"wind": 0.30, "shear": 0.50, "rh": 0.20}

# Percentiles used for robust min-max scaling. Values below the 5th percentile
# map to 0 and above the 95th to 1, so a few extreme cells (e.g. a jet core)
# don't compress everyone else into a narrow band.
LOWER_QUANTILE = 0.05
UPPER_QUANTILE = 0.95

# Category bins. Lower edges are inclusive, upper edges exclusive, except the
# last bin, which includes 1.0:
#   LOW      0.00 <= h < 0.25
#   MODERATE 0.25 <= h < 0.50
#   HIGH     0.50 <= h < 0.75
#   SEVERE   0.75 <= h <= 1.00
CATEGORY_LABELS = ("LOW", "MODERATE", "HIGH", "SEVERE")
CATEGORY_EDGES = (0.25, 0.50, 0.75)
MISSING_CATEGORY = -1  # code for cells whose hazard_score is NaN

# Canonical routing-penalty formula, also used per edge by routing.cost.
MULTIPLIER_MIN = 1.0
MULTIPLIER_RANGE = 4.0  # multiplier = 1 + 4 * hazard -> [1, 5]

TOLERANCE = 1e-9

OUTPUT_VARIABLES = (
    "normalized_wind", "normalized_shear", "normalized_rh",
    "hazard_score", "hazard_category", "weather_multiplier",
)


def robust_minmax(
    da: xr.DataArray,
    lower_q: float = LOWER_QUANTILE,
    upper_q: float = UPPER_QUANTILE,
    bounds: tuple[float, float] | None = None,
) -> tuple[xr.DataArray, tuple[float, float]]:
    """Rescale ``da`` linearly to [0, 1] between two robust bounds, then clip.

    norm = clip((x - lo) / (hi - lo), 0, 1)

    By default ``lo`` and ``hi`` are the ``lower_q`` and ``upper_q``
    percentiles of ``da`` (NaNs ignored). Pass ``bounds=(lo, hi)`` to reuse
    fixed bounds, e.g. to compare several forecast files on one scale.

    If ``hi == lo`` (a constant field) there is no variation to rank, so every
    cell maps to 0.

    Returns:
        The normalized array and the ``(lo, hi)`` bounds actually used.
    """
    if bounds is None:
        lo, hi = (float(q) for q in np.nanquantile(da.values, [lower_q, upper_q]))
    else:
        lo, hi = (float(b) for b in bounds)
    if hi < lo:
        raise ValueError(f"Upper bound {hi} is below lower bound {lo}")
    if hi == lo:
        return xr.zeros_like(da, dtype=float).where(da.notnull()), (lo, hi)
    return ((da - lo) / (hi - lo)).clip(0.0, 1.0), (lo, hi)


def normalize_rh(rh: xr.DataArray) -> xr.DataArray:
    """Scale relative humidity from its physical 0-100 % range to [0, 1].

    RH already has natural bounds, so no data-dependent scaling is needed.
    GFS can report RH marginally above 100 %, so the result is clipped.
    """
    return (rh / 100.0).clip(0.0, 1.0)


def weighted_hazard(
    normalized_wind: xr.DataArray,
    normalized_shear: xr.DataArray,
    normalized_rh: xr.DataArray,
    weights: dict = WEIGHTS,
) -> xr.DataArray:
    """Weighted sum of the normalized features.

    With weights summing to 1 and inputs in [0, 1], the result is in [0, 1];
    it is clipped anyway to absorb floating-point rounding.
    """
    total = sum(weights.values())
    if not np.isclose(total, 1.0):
        raise ValueError(f"Weights must sum to 1, got {total}")
    score = (
        weights["wind"] * normalized_wind
        + weights["shear"] * normalized_shear
        + weights["rh"] * normalized_rh
    )
    return score.clip(0.0, 1.0)


def categorize(hazard_score: xr.DataArray) -> xr.DataArray:
    """Map hazard scores to integer category codes 0-3.

    Codes index ``CATEGORY_LABELS``. A value exactly on an edge goes to the
    higher category (0.25 -> MODERATE, 0.5 -> HIGH, 0.75 -> SEVERE), and 1.0
    is SEVERE. NaN scores get ``MISSING_CATEGORY`` (-1).

    Integer codes (rather than strings) follow the CF ``flag_values`` /
    ``flag_meanings`` convention, so they survive NetCDF round-trips; use
    ``category_labels`` to get the strings.
    """
    # np.digitize with right=False puts x == edge into the upper bin.
    codes = np.digitize(hazard_score.values, CATEGORY_EDGES, right=False)
    codes = np.where(np.isnan(hazard_score.values), MISSING_CATEGORY, codes).astype(np.int8)
    return xr.DataArray(
        codes,
        coords=hazard_score.coords,
        dims=hazard_score.dims,
        attrs={
            "long_name": "Experimental hazard category",
            "flag_values": np.arange(len(CATEGORY_LABELS), dtype=np.int8),
            "flag_meanings": " ".join(CATEGORY_LABELS),
            "missing_value_code": MISSING_CATEGORY,
        },
    )


def category_labels(categories: xr.DataArray) -> xr.DataArray:
    """Convert category codes to their string labels ("MISSING" for -1)."""
    lookup = np.array([*CATEGORY_LABELS, "MISSING"])
    # Index -1 already selects the trailing "MISSING" entry.
    return xr.DataArray(
        lookup[categories.values], coords=categories.coords, dims=categories.dims
    )


def weather_multiplier(hazard_score):
    """Routing cost multiplier: 1 + 4 * hazard (0 -> 1.0, 0.5 -> 3.0, 1 -> 5.0).

    Accepts a DataArray, NumPy array or float and returns the same type.
    """
    return MULTIPLIER_MIN + MULTIPLIER_RANGE * hazard_score


def compute_hazard(ds: xr.Dataset, bounds: dict | None = None) -> xr.Dataset:
    """Add normalized features, hazard score, category and multiplier to ``ds``.

    Args:
        ds: processed Dataset with ``REQUIRED_INPUTS`` on ``(latitude, longitude)``.
        bounds: optional fixed ``{"wind": (lo, hi), "shear": (lo, hi)}``
            normalization bounds; percentiles of ``ds`` are used otherwise.

    Returns:
        A copy of ``ds`` with ``OUTPUT_VARIABLES`` added. The percentile bounds
        used are stored as variable attributes so the scores can be traced.
    """
    validate_input(ds)
    bounds = bounds or {}

    norm_wind, wind_bounds = robust_minmax(ds["wind_speed_300"], bounds=bounds.get("wind"))
    norm_shear, shear_bounds = robust_minmax(ds["wind_shear"], bounds=bounds.get("shear"))
    norm_rh = normalize_rh(ds["rh_mean"])
    score = weighted_hazard(norm_wind, norm_shear, norm_rh)

    out = ds.copy()
    out["normalized_wind"] = norm_wind.assign_attrs(
        source="wind_speed_300", method="robust_minmax",
        lower_bound=wind_bounds[0], upper_bound=wind_bounds[1],
        lower_quantile=LOWER_QUANTILE, upper_quantile=UPPER_QUANTILE,
    )
    out["normalized_shear"] = norm_shear.assign_attrs(
        source="wind_shear", method="robust_minmax",
        lower_bound=shear_bounds[0], upper_bound=shear_bounds[1],
        lower_quantile=LOWER_QUANTILE, upper_quantile=UPPER_QUANTILE,
    )
    out["normalized_rh"] = norm_rh.assign_attrs(
        source="rh_mean", method="physical_range", lower_bound=0.0, upper_bound=100.0,
    )
    out["hazard_score"] = score.assign_attrs(
        long_name="Experimental routing hazard score (not an operational forecast)",
        weight_wind=WEIGHTS["wind"], weight_shear=WEIGHTS["shear"], weight_rh=WEIGHTS["rh"],
    )
    out["hazard_category"] = categorize(score)
    out["weather_multiplier"] = weather_multiplier(score).assign_attrs(
        long_name="Routing cost multiplier, 1 + 4 * hazard_score",
    )

    validate_output(out, ds)
    return out


def validate_input(ds: xr.Dataset) -> None:
    """Check the processed Dataset has the fields and coordinates we need."""
    missing = [v for v in REQUIRED_INPUTS if v not in ds.data_vars]
    if missing:
        raise ValueError(f"Missing required input variables: {missing}")
    for coord in (*SPATIAL_DIMS, "valid_time"):
        if coord not in ds.coords:
            raise ValueError(f"Missing coordinate: {coord}")
    for name in REQUIRED_INPUTS:
        if ds[name].dims != SPATIAL_DIMS:
            raise ValueError(f"{name}: expected dims {SPATIAL_DIMS}, got {ds[name].dims}")


def _check_range(da: xr.DataArray, lo: float, hi: float) -> None:
    vals = da.values
    finite = vals[~np.isnan(vals)]
    if finite.size and (finite.min() < lo - TOLERANCE or finite.max() > hi + TOLERANCE):
        raise ValueError(
            f"{da.name}: values [{finite.min()}, {finite.max()}] outside [{lo}, {hi}]"
        )


def validate_output(out: xr.Dataset, source: xr.Dataset) -> None:
    """Check bounds, grid shape and that no NaNs were introduced."""
    expected_shape = tuple(source.sizes[d] for d in SPATIAL_DIMS)
    # Cells that were NaN in any input may legitimately be NaN in the outputs.
    input_nan = source[list(REQUIRED_INPUTS)].to_array().isnull().any(dim="variable")

    for name in OUTPUT_VARIABLES:
        da = out[name]
        if da.dims != SPATIAL_DIMS or da.shape != expected_shape:
            raise ValueError(
                f"{name}: expected dims {SPATIAL_DIMS} shape {expected_shape}, "
                f"got {da.dims} {da.shape}"
            )
        if name == "hazard_category":
            missing = da == MISSING_CATEGORY
        else:
            missing = da.isnull()
        new_missing = int((missing & ~input_nan).sum())
        if new_missing:
            raise ValueError(f"{name}: {new_missing} missing values introduced")

    for name in ("normalized_wind", "normalized_shear", "normalized_rh", "hazard_score"):
        _check_range(out[name], 0.0, 1.0)
    _check_range(out["weather_multiplier"], MULTIPLIER_MIN, MULTIPLIER_MIN + MULTIPLIER_RANGE)


def hazard_from_processed_file(path, save: bool = False) -> xr.Dataset:
    """Load a processed NetCDF from ``preprocessing.save_processed`` and score it.

    A bare filename is looked up in ``processed_dir()``. With ``save=True`` the
    result is written next to it as ``<stem>.hazard.nc``.
    """
    path = Path(path)
    if not path.is_file() and not path.is_absolute():
        path = processed_dir() / path
    if not path.is_file():
        raise FileNotFoundError(f"Processed file not found: {path}")
    with xr.open_dataset(path, engine="scipy") as ds:
        out = compute_hazard(ds.load())
    if save:
        save_hazard(out, path.with_name(f"{path.stem}.hazard.nc"))
    return out


def save_hazard(ds: xr.Dataset, path) -> Path:
    """Write the hazard Dataset to NetCDF (NetCDF3 via scipy) and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path, engine="scipy")
    return path
