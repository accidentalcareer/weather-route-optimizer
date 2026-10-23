"""Discover the processed hazard forecasts available for routing.

Each ``*.hazard.nc`` file holds one forecast. A forecast is identified by its
``valid_time`` coordinate, read from the NetCDF itself; filenames and file
modification times are never used to decide which forecast a file holds.

Only metadata is read here (no hazard grids are loaded), so a scan is cheap:

    discover_forecasts()  -> [ForecastEntry, ...] sorted by valid time
    default_forecast()    -> the earliest valid time (deterministic)
    find_forecast()       -> the entry for a requested valid time, or None

All times are timezone-aware UTC datetimes.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from weather.grib_reader import processed_dir

logger = logging.getLogger(__name__)

HAZARD_GLOB = "*.hazard.nc"


@dataclass(frozen=True)
class ForecastEntry:
    """Metadata for one processed hazard forecast file."""

    valid_time: datetime
    reference_time: datetime | None   # model run (GFS cycle) time
    step_hours: float | None          # forecast lead time, valid - reference
    path: Path
    grib_file: str | None             # GRIB2 file the forecast was derived from

    @property
    def source_file(self) -> str:
        return self.path.name

    def to_dict(self) -> dict:
        """JSON-ready summary (no local filesystem paths)."""
        return {
            "valid_time": self.valid_time.isoformat(),
            "reference_time": self.reference_time.isoformat() if self.reference_time else None,
            "step_hours": self.step_hours,
            "source_file": self.source_file,
            "grib_file": self.grib_file,
        }


def to_utc(value) -> datetime:
    """A datetime64 / datetime / ISO string as an aware UTC ``datetime``.

    Naive values are taken to be UTC, which is how GFS times are stored.
    """
    ts = pd.Timestamp(value)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.to_pydatetime()


def iso_utc(value: datetime) -> str:
    """ISO-8601 with a ``Z`` suffix, matching how the API serialises datetimes."""
    return to_utc(value).isoformat().replace("+00:00", "Z")


def _step_hours(ds: xr.Dataset, valid: datetime, reference: datetime | None) -> float | None:
    if "step" in ds.coords and np.issubdtype(ds["step"].dtype, np.timedelta64):
        return float(ds["step"].values / np.timedelta64(1, "h"))
    if reference is not None:
        return (valid - reference).total_seconds() / 3600.0
    return None


def forecast_entry_from_dataset(ds: xr.Dataset, path) -> ForecastEntry:
    """Forecast metadata from an open hazard Dataset.

    Raises ``ValueError`` if it has no scalar ``valid_time`` coordinate.
    """
    path = Path(path)
    if "valid_time" not in ds.coords or ds["valid_time"].ndim != 0:
        raise ValueError(f"{path.name} has no valid_time coordinate")
    valid = to_utc(ds["valid_time"].values)
    reference = to_utc(ds["time"].values) if "time" in ds.coords and ds["time"].ndim == 0 else None
    return ForecastEntry(
        valid_time=valid,
        reference_time=reference,
        step_hours=_step_hours(ds, valid, reference),
        path=path,
        grib_file=ds.attrs.get("source_file"),
    )


def read_forecast_metadata(path) -> ForecastEntry:
    """Read a hazard file's forecast metadata without loading its grids.

    Raises ``FileNotFoundError`` if the file is missing and ``ValueError`` if
    it has no scalar ``valid_time`` coordinate.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Hazard file not found: {path.name}")
    with xr.open_dataset(path, engine="scipy") as ds:
        return forecast_entry_from_dataset(ds, path)


def _sort_key(entry: ForecastEntry):
    oldest = datetime.min.replace(tzinfo=timezone.utc)
    return (entry.valid_time, entry.reference_time or oldest, entry.source_file)


def discover_forecasts(directory=None) -> list[ForecastEntry]:
    """All readable hazard forecasts in ``directory``, oldest valid time first.

    Unreadable files are logged and skipped. If several files share a valid
    time, the one from the newest model run is kept (ties: last filename), so
    each valid time maps to exactly one forecast.
    """
    directory = Path(directory) if directory else processed_dir()
    by_valid: dict[datetime, ForecastEntry] = {}
    for path in sorted(directory.glob(HAZARD_GLOB)):
        try:
            entry = read_forecast_metadata(path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("Skipping unreadable hazard file %s: %s", path.name, exc)
            continue
        current = by_valid.get(entry.valid_time)
        if current is None or _sort_key(entry) > _sort_key(current):
            by_valid[entry.valid_time] = entry
    return sorted(by_valid.values(), key=_sort_key)


def default_forecast(entries: list[ForecastEntry]) -> ForecastEntry | None:
    """The forecast used when none is requested: the earliest valid time."""
    return entries[0] if entries else None


def find_forecast(entries: list[ForecastEntry], valid_time) -> ForecastEntry | None:
    """The entry whose valid time equals ``valid_time`` (any UTC-equivalent form)."""
    wanted = to_utc(valid_time)
    return next((e for e in entries if e.valid_time == wanted), None)
