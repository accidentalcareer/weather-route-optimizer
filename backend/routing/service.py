"""High-level route service: lat/lon in, two routes and their comparison out.

This module only orchestrates the existing layers; it holds no routing maths:

    available forecasts -> processed hazard files by valid time
                           (weather.forecast_catalog)
    load_hazard_grid  -> GeoGrid with hazard_score        (routing.grid)
    nearest_node      -> snap origin / destination        (routing.grid)
    astar             -> distance-only and weather routes (routing.astar,
                         with routing.cost.distance_cost / weather_cost)
    compare_routes    -> trade-off metrics                (routing.metrics)
    what-if (optional) -> astar with make_weather_cost(K), compared with the
                         normal weather route by compare_what_if (routing.metrics)

Coordinate convention for everything returned: ``{"lat": float, "lon": float}``
in decimal degrees, latitude north-positive, longitude east-positive in
[-180, 180]. Route coordinates are the grid-point centres, ordered from
origin to destination.

Forecast selection: each processed ``*.hazard.nc`` file is one forecast,
identified by its ``valid_time``. A route request may name a valid time;
otherwise the default forecast is used, which is the earliest valid time
(never the newest file on disk). Setting ``$WEATHER_HAZARD_FILE`` pins the
service to that single file, as before multi-forecast support.

``HazardGridCache`` keeps the forecast list and every grid loaded so far,
keyed by valid time; ``refresh_hazard_grid()`` and ``clear_hazard_cache()``
pick up new files without a restart.
"""

import math
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from routing.astar import RouteNotFoundError, RouteResult, astar
from routing.cost import (
    DEFAULT_HAZARD_PENALTY,
    distance_cost,
    make_weather_cost,
    validate_hazard_penalty,
    weather_cost,
)
from routing.grid import GeoGrid, haversine_km, load_hazard_grid
from routing.metrics import compare_routes, compare_what_if
from weather.forecast_catalog import (
    HAZARD_GLOB,
    ForecastEntry,
    discover_forecasts,
    find_forecast,
    forecast_entry_from_dataset,
    iso_utc,
    read_forecast_metadata,
    to_utc,
)
from weather.grib_reader import processed_dir

# Optional override: path (absolute, or relative to the processed directory)
# of the only hazard NetCDF to serve. Otherwise every *.hazard.nc is offered.
HAZARD_FILE_ENV = "WEATHER_HAZARD_FILE"


class RouteServiceError(Exception):
    """Base class for errors the service reports to callers."""


class InvalidCoordinateError(RouteServiceError, ValueError):
    """A latitude/longitude is not a finite value in the valid range."""


class LocationOutsideGridError(RouteServiceError, ValueError):
    """A location lies outside the forecast grid."""

    def __init__(self, which: str, lat: float, lon: float, bounds: dict):
        self.which = which
        self.bounds = bounds
        super().__init__(
            f"{which.capitalize()} ({lat}, {lon}) is outside the forecast area: "
            f"latitude {bounds['lat_min']} to {bounds['lat_max']}, "
            f"longitude {bounds['lon_min']} to {bounds['lon_max']}."
        )


class SameNodeError(RouteServiceError, ValueError):
    """Origin and destination snap to the same grid point."""


class InvalidHazardPenaltyError(RouteServiceError, ValueError):
    """The what-if hazard penalty is not a finite number in [0, 10]."""


class WeatherDataError(RouteServiceError, RuntimeError):
    """The processed hazard file is missing or unusable."""


class ForecastNotFoundError(RouteServiceError, LookupError):
    """No available forecast has the requested valid time."""

    def __init__(self, requested: datetime, available: list[ForecastEntry]):
        self.requested = requested
        self.available = [e.valid_time for e in available]
        listing = ", ".join(iso_utc(t) for t in self.available) or "none"
        super().__init__(
            f"No forecast is available for valid time {iso_utc(requested)}. "
            f"Available valid times: {listing}."
        )


# --- forecasts and the hazard grid cache --------------------------------

@dataclass(frozen=True)
class HazardGridSnapshot:
    """One loaded forecast: the routing grid plus its metadata."""

    grid: GeoGrid
    forecast: ForecastEntry
    loaded_at: datetime

    @property
    def valid_time(self) -> datetime:
        return self.forecast.valid_time

    @property
    def reference_time(self) -> datetime | None:
        return self.forecast.reference_time

    @property
    def source_file(self) -> str:
        return self.forecast.source_file

    @property
    def path(self) -> Path:
        return self.forecast.path


def _override_path() -> Path | None:
    override = os.getenv(HAZARD_FILE_ENV)
    if not override:
        return None
    path = Path(override)
    return path if path.is_absolute() else processed_dir() / path


def scan_forecasts() -> list[ForecastEntry]:
    """Forecasts that can be served, oldest valid time first.

    With ``$WEATHER_HAZARD_FILE`` set that is just the one file (raising
    ``WeatherDataError`` if it is missing or unreadable); otherwise every
    readable ``*.hazard.nc`` in the processed directory.
    """
    override = _override_path()
    if override is None:
        return discover_forecasts()
    try:
        return [read_forecast_metadata(override)]
    except FileNotFoundError:
        raise WeatherDataError(f"Hazard file not found: {override.name}") from None
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise WeatherDataError(f"Could not read hazard file {override.name}: {exc}") from exc


def _no_forecasts_error() -> WeatherDataError:
    return WeatherDataError(
        f"No processed hazard file ({HAZARD_GLOB}) found in {processed_dir()}. "
        "Run preprocessing and the hazard model first."
    )


def load_snapshot(path: Path) -> HazardGridSnapshot:
    """Load and sanity-check a hazard NetCDF. Raises ``WeatherDataError``."""
    try:
        grid, ds = load_hazard_grid(path)
    except FileNotFoundError as exc:
        raise WeatherDataError(f"Hazard file not found: {Path(path).name}") from exc
    except (OSError, ValueError, KeyError) as exc:
        raise WeatherDataError(f"Could not read hazard file {Path(path).name}: {exc}") from exc

    if grid.hazard is None:
        raise WeatherDataError(f"{Path(path).name} has no hazard_score variable")
    if not np.all(np.isfinite(grid.hazard)):
        raise WeatherDataError(f"{Path(path).name} contains non-finite hazard values")
    if grid.hazard.min() < 0 or grid.hazard.max() > 1:
        raise WeatherDataError(f"{Path(path).name} has hazard values outside [0, 1]")
    try:
        forecast = forecast_entry_from_dataset(ds, path)
    except ValueError as exc:
        raise WeatherDataError(str(exc)) from exc

    return HazardGridSnapshot(grid=grid, forecast=forecast, loaded_at=datetime.now(timezone.utc))


class HazardGridCache:
    """The forecast list plus loaded hazard grids, keyed by valid time.

    Both are filled lazily and kept until refreshed or cleared, so a forecast
    is loaded once however many requests use it. All updates happen under one
    lock, so concurrent requests never load the same file twice.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._forecasts: list[ForecastEntry] | None = None
        self._snapshots: dict[datetime, HazardGridSnapshot] = {}

    def _set_forecasts(self, forecasts: list[ForecastEntry]) -> None:
        """Store a new forecast list and drop grids whose file is no longer in it."""
        paths = {f.path for f in forecasts}
        self._forecasts = forecasts
        self._snapshots = {k: s for k, s in self._snapshots.items() if s.path in paths}

    def forecasts(self, rescan: bool = False) -> list[ForecastEntry]:
        """Available forecasts, oldest first, scanning the disk on first use."""
        with self._lock:
            if self._forecasts is None or rescan:
                self._set_forecasts(scan_forecasts())
            return list(self._forecasts)

    def get(self, forecast: ForecastEntry) -> HazardGridSnapshot:
        """The loaded grid for ``forecast``, loading it on first use."""
        snapshot = self._snapshots.get(forecast.valid_time)
        if snapshot is not None and snapshot.path == forecast.path:
            return snapshot
        with self._lock:
            snapshot = self._snapshots.get(forecast.valid_time)
            if snapshot is None or snapshot.path != forecast.path:
                snapshot = load_snapshot(forecast.path)
                self._snapshots[forecast.valid_time] = snapshot
            return snapshot

    def cached_valid_times(self) -> list[datetime]:
        with self._lock:
            return sorted(self._snapshots)

    def refresh(self, path=None) -> HazardGridSnapshot:
        """Rescan the forecast list and reload ``path`` (default: the default forecast).

        Nothing changes if scanning or loading fails.
        """
        forecasts = scan_forecasts()
        if path is None:
            if not forecasts:
                raise _no_forecasts_error()
            path = forecasts[0].path
        snapshot = load_snapshot(Path(path))
        with self._lock:
            self._set_forecasts(forecasts)
            self._snapshots[snapshot.valid_time] = snapshot
        return snapshot

    def clear(self) -> None:
        """Forget the forecast list and every grid; the next use rescans and reloads."""
        with self._lock:
            self._forecasts = None
            self._snapshots = {}


_cache = HazardGridCache()


def available_forecasts() -> list[ForecastEntry]:
    return _cache.forecasts()


def resolve_forecast(valid_time=None) -> ForecastEntry:
    """The forecast to route on: the one valid at ``valid_time``, or the default.

    The default is the earliest available valid time. An unknown valid time
    triggers one rescan (a file may have been added) before
    ``ForecastNotFoundError`` is raised.
    """
    forecasts = _cache.forecasts()
    if valid_time is None:
        if not forecasts:
            raise _no_forecasts_error()
        return forecasts[0]

    wanted = to_utc(valid_time)
    entry = find_forecast(forecasts, wanted)
    if entry is None:
        forecasts = _cache.forecasts(rescan=True)
        entry = find_forecast(forecasts, wanted)
    if entry is None:
        raise ForecastNotFoundError(wanted, forecasts)
    return entry


def default_hazard_path() -> Path:
    """Path of the default forecast file. Raises ``WeatherDataError`` if there is none."""
    return resolve_forecast().path


def get_hazard_snapshot(valid_time=None) -> HazardGridSnapshot:
    return _cache.get(resolve_forecast(valid_time))


def refresh_hazard_grid(path=None) -> HazardGridSnapshot:
    return _cache.refresh(path)


def clear_hazard_cache() -> None:
    _cache.clear()


def list_forecasts() -> dict:
    """Available forecasts for the API, oldest first, with the default marked."""
    forecasts = available_forecasts()
    default = forecasts[0] if forecasts else None
    return {
        "default_valid_time": default.valid_time.isoformat() if default else None,
        "forecasts": [{**f.to_dict(), "is_default": f is default} for f in forecasts],
    }



# --- helpers ------------------------------------------------------------

def _latlon(lat: float, lon: float) -> dict:
    """JSON-ready coordinate with longitude in [-180, 180]."""
    lon = ((float(lon) + 180.0) % 360.0) - 180.0
    return {"lat": float(lat), "lon": lon}


def _validate_coordinate(which: str, lat: float, lon: float) -> None:
    if not (isinstance(lat, (int, float)) and math.isfinite(lat) and -90 <= lat <= 90):
        raise InvalidCoordinateError(f"{which.capitalize()} latitude must be between -90 and 90, got {lat}")
    if not (isinstance(lon, (int, float)) and math.isfinite(lon) and -180 <= lon <= 180):
        raise InvalidCoordinateError(f"{which.capitalize()} longitude must be between -180 and 180, got {lon}")


def grid_bounds(grid: GeoGrid) -> dict:
    return {
        "lat_min": float(grid.lats.min()), "lat_max": float(grid.lats.max()),
        "lon_min": float(grid.lons.min()), "lon_max": float(grid.lons.max()),
    }


def _snap(grid: GeoGrid, which: str, lat: float, lon: float):
    try:
        node = grid.nearest_node(lat, lon)
    except ValueError:
        raise LocationOutsideGridError(which, lat, lon, grid_bounds(grid)) from None
    snapped = _latlon(*grid.index_to_latlon(node))
    return node, {
        "requested": _latlon(lat, lon),
        "snapped": snapped,
        "snap_distance_km": float(haversine_km(lat, lon, snapped["lat"], snapped["lon"])),
    }


def _route_info(grid: GeoGrid, route: RouteResult, metrics) -> dict:
    return {
        "coordinates": [_latlon(*grid.index_to_latlon(n)) for n in route.nodes],
        "node_count": route.n_nodes,
        "length_km": route.path_length_km,
        "total_cost": route.total_cost,
        "nodes_expanded": route.nodes_expanded,
        "average_hazard": metrics.average_hazard,
        "maximum_hazard": metrics.maximum_hazard,
        "category_counts": metrics.category_counts,
    }


def _metrics_info(m) -> dict:
    """Per-route figures from ``routing.metrics.RouteMetrics``."""
    return {
        "node_count": m.n_nodes,
        "length_km": m.length_km,
        "weather_cost": m.weather_cost,
        "average_hazard": m.average_hazard,
        "maximum_hazard": m.maximum_hazard,
        "category_counts": m.category_counts,
    }


def _what_if(grid: GeoGrid, start, goal, weather_route: RouteResult, hazard_penalty) -> dict:
    """The ``what_if`` response section; disabled when no penalty was given."""
    if hazard_penalty is None:
        return {
            "enabled": False, "hazard_penalty": DEFAULT_HAZARD_PENALTY,
            "baseline_route": None, "what_if_route": None, "comparison": None,
        }
    cost = make_weather_cost(hazard_penalty)
    # K = 4 is the normal weather cost, so the normal route is the answer.
    route = weather_route if cost is weather_cost else astar(grid, start, goal, cost)
    c = compare_what_if(grid, weather_route, route, hazard_penalty)
    return {
        "enabled": True,
        "hazard_penalty": c.hazard_penalty,
        "baseline_route": _metrics_info(c.baseline_route),
        "what_if_route": {
            **_route_info(grid, route, c.what_if_route),
            "weather_cost": c.what_if_route.weather_cost,
        },
        "comparison": {
            "routes_identical": c.routes_identical,
            "extra_distance_km": c.extra_distance_km,
            "extra_distance_percent": c.extra_distance_percent,
            "weather_cost_difference": c.weather_cost_difference,
            "weather_cost_reduction_percent": c.weather_cost_reduction_percent,
            "average_hazard_reduction": c.average_hazard_reduction,
            "average_hazard_reduction_percent": c.average_hazard_reduction_percent,
            "maximum_hazard_reduction": c.maximum_hazard_reduction,
            "severe_cells_avoided": c.severe_cells_avoided,
            "high_hazard_cells_avoided": c.high_hazard_cells_avoided,
            "explanation": c.explanation,
        },
    }


# --- main entry point ---------------------------------------------------

def calculate_route(origin_lat: float, origin_lon: float,
                    destination_lat: float, destination_lon: float,
                    forecast_valid_time=None, hazard_penalty=None) -> dict:
    """Compute and compare distance-only and weather-aware routes.

    ``forecast_valid_time`` (datetime or ISO-8601 string, UTC) selects the
    forecast; ``None`` uses the default forecast (see ``resolve_forecast``).

    ``hazard_penalty`` (K in [0, 10]) enables what-if mode: a third route is
    planned with multiplier ``1 + K * hazard`` and compared with the normal
    weather-aware route in ``what_if``. The distance-only route, the normal
    weather-aware route and ``comparison`` are the same with or without it.

    Returns a JSON-serialisable dict with ``forecast``, ``origin``,
    ``destination``, ``distance_route``, ``weather_route`` and ``comparison``.

    Raises:
        InvalidCoordinateError, LocationOutsideGridError, SameNodeError,
        InvalidHazardPenaltyError, ForecastNotFoundError, WeatherDataError,
        RouteNotFoundError.
    """
    _validate_coordinate("origin", origin_lat, origin_lon)
    _validate_coordinate("destination", destination_lat, destination_lon)
    if hazard_penalty is not None:
        try:
            hazard_penalty = validate_hazard_penalty(hazard_penalty)
        except ValueError as exc:
            raise InvalidHazardPenaltyError(str(exc)) from None

    snapshot = get_hazard_snapshot(forecast_valid_time)
    grid = snapshot.grid

    start, origin = _snap(grid, "origin", origin_lat, origin_lon)
    goal, destination = _snap(grid, "destination", destination_lat, destination_lon)
    if start == goal:
        raise SameNodeError(
            "Origin and destination are closer than one grid cell and map to the same "
            f"grid point ({origin['snapped']['lat']}, {origin['snapped']['lon']})."
        )

    distance_route = astar(grid, start, goal, distance_cost)
    weather_route = astar(grid, start, goal, weather_cost)
    c = compare_routes(grid, distance_route, weather_route)

    return {
        "forecast": {**snapshot.forecast.to_dict(), "grid_bounds": grid_bounds(grid)},
        "origin": origin,
        "destination": destination,
        "distance_route": _route_info(grid, distance_route, c.distance_route),
        "weather_route": _route_info(grid, weather_route, c.weather_route),
        "comparison": {
            "routes_identical": c.routes_identical,
            "extra_distance_km": c.extra_distance_km,
            "extra_distance_percent": c.extra_distance_percent,
            "distance_route_weather_cost": c.distance_route_weather_cost,
            "weather_route_weather_cost": c.weather_route_weather_cost,
            "weather_cost_difference": c.weather_cost_difference,
            "weather_cost_reduction_percent": c.weather_cost_reduction_percent,
            "average_hazard_distance_route": c.distance_route.average_hazard,
            "average_hazard_weather_route": c.weather_route.average_hazard,
            "maximum_hazard_distance_route": c.distance_route.maximum_hazard,
            "maximum_hazard_weather_route": c.weather_route.maximum_hazard,
            "severe_cells_avoided": c.severe_cells_avoided,
            "high_hazard_cells_avoided": c.high_hazard_cells_avoided,
            "estimated_fuel_delta_percent": c.estimated_fuel_delta_percent,
            "fuel_proxy_note": c.fuel_proxy_note,
            "explanation": c.explanation,
        },
        "what_if": _what_if(grid, start, goal, weather_route, hazard_penalty),
    }


__all__ = [
    "calculate_route", "get_hazard_snapshot", "refresh_hazard_grid", "clear_hazard_cache",
    "available_forecasts", "list_forecasts", "resolve_forecast", "default_hazard_path",
    "HazardGridCache", "HazardGridSnapshot", "ForecastEntry",
    "RouteServiceError", "InvalidCoordinateError", "LocationOutsideGridError",
    "SameNodeError", "InvalidHazardPenaltyError", "WeatherDataError", "ForecastNotFoundError", "RouteNotFoundError",
]
