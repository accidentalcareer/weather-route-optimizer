"""API endpoints: request/response models, forecast discovery and route planning.

All routing work happens in ``routing.service``; this module only validates
input, maps service errors to HTTP status codes and documents the schema.

Error responses share one shape::

    {"error": {"code": "outside_grid", "message": "...", "details": [...]}}
"""

import logging
from datetime import datetime

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from routing import service
from routing.cost import HAZARD_PENALTY_MAX, HAZARD_PENALTY_MIN
from weather.forecast_catalog import iso_utc

logger = logging.getLogger(__name__)

router = APIRouter(tags=["routing"])


# --- request ------------------------------------------------------------

class Coordinate(BaseModel):
    """A location in decimal degrees (north- and east-positive)."""

    model_config = ConfigDict(json_schema_extra={"example": {"lat": 28.6139, "lon": 77.2090}})

    lat: float = Field(..., ge=-90, le=90, allow_inf_nan=False, description="Latitude, -90 to 90")
    lon: float = Field(..., ge=-180, le=180, allow_inf_nan=False, description="Longitude, -180 to 180")


class RouteRequest(BaseModel):
    """Origin and destination for a route, and optionally the forecast to use."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "origin": {"lat": 28.6139, "lon": 77.2090},
                "destination": {"lat": 19.0760, "lon": 72.8777},
                "forecast_valid_time": "2026-10-22T18:00:00Z",
                "hazard_penalty": 8.0,
            }
        },
    )

    origin: Coordinate
    destination: Coordinate
    forecast_valid_time: datetime | None = Field(
        default=None,
        description="Valid time (ISO-8601, UTC) of a forecast listed by GET /api/forecasts. "
        "Omit to use the default forecast. A time without an offset is read as UTC.",
    )
    hazard_penalty: float | None = Field(
        default=None,
        ge=HAZARD_PENALTY_MIN,
        le=HAZARD_PENALTY_MAX,
        allow_inf_nan=False,
        strict=True,
        description="What-if hazard penalty K, 0-10: plans an extra weather route with "
        "multiplier 1 + K * hazard_score and compares it with the normal one (K = 4). "
        "Omit for normal routing only.",
    )


# --- response -----------------------------------------------------------

class SnappedLocation(BaseModel):
    requested: Coordinate = Field(description="Location as sent in the request")
    snapped: Coordinate = Field(description="Nearest forecast grid point, where the route starts/ends")
    snap_distance_km: float = Field(description="Great-circle distance from requested to snapped point")


class CategoryCounts(BaseModel):
    """Number of route grid points in each hazard category."""

    LOW: int
    MODERATE: int
    HIGH: int
    SEVERE: int


class RouteInfo(BaseModel):
    coordinates: list[Coordinate] = Field(
        description="Route vertices from origin to destination (grid-point centres)"
    )
    node_count: int
    length_km: float = Field(description="Geographic length of the route")
    total_cost: float = Field(
        description="A* cost of this route under its own edge cost: km for the distance "
        "route, hazard-weighted km for the weather route"
    )
    nodes_expanded: int = Field(description="Grid nodes A* expanded to find this route")
    average_hazard: float = Field(description="Mean hazard_score over route grid points, 0-1")
    maximum_hazard: float = Field(description="Highest hazard_score on the route, 0-1")
    category_counts: CategoryCounts


class RouteComparisonModel(BaseModel):
    routes_identical: bool
    extra_distance_km: float = Field(description="Weather route length minus distance route length")
    extra_distance_percent: float
    distance_route_weather_cost: float
    weather_route_weather_cost: float
    weather_cost_difference: float = Field(
        description="Distance route weather cost minus weather route weather cost"
    )
    weather_cost_reduction_percent: float
    average_hazard_distance_route: float
    average_hazard_weather_route: float
    maximum_hazard_distance_route: float
    maximum_hazard_weather_route: float
    severe_cells_avoided: int = Field(
        description="SEVERE points on the distance route minus on the weather route (may be negative)"
    )
    high_hazard_cells_avoided: int = Field(
        description="HIGH+SEVERE points on the distance route minus on the weather route"
    )
    estimated_fuel_delta_percent: float = Field(
        description="Distance-based fuel proxy only (percentage change in length); "
        "not an aircraft fuel-burn prediction"
    )
    fuel_proxy_note: str
    explanation: str


class RouteMetricsInfo(BaseModel):
    """Per-route figures without coordinates."""

    node_count: int
    length_km: float
    weather_cost: float = Field(description="Weather-weighted cost on the normal (K = 4) scale")
    average_hazard: float
    maximum_hazard: float
    category_counts: CategoryCounts


class WhatIfRouteInfo(RouteInfo):
    """The what-if route. ``total_cost`` is its A* cost under K."""

    weather_cost: float = Field(description="Weather-weighted cost on the normal (K = 4) scale")


class WhatIfComparisonModel(BaseModel):
    """Normal weather-aware route minus what-if route, unless stated otherwise."""

    routes_identical: bool
    extra_distance_km: float = Field(description="What-if length minus normal weather route length")
    extra_distance_percent: float
    weather_cost_difference: float = Field(description="On the normal (K = 4) scale")
    weather_cost_reduction_percent: float
    average_hazard_reduction: float
    average_hazard_reduction_percent: float
    maximum_hazard_reduction: float
    severe_cells_avoided: int
    high_hazard_cells_avoided: int = Field(description="HIGH + SEVERE points")
    explanation: str


class WhatIfInfo(BaseModel):
    """What-if scenario; route fields are null when ``enabled`` is false."""

    enabled: bool = Field(description="True when the request supplied hazard_penalty")
    hazard_penalty: float = Field(description="K used (4 = normal weather routing)")
    baseline_route: RouteMetricsInfo | None = Field(description="Normal weather-aware route")
    what_if_route: WhatIfRouteInfo | None
    comparison: WhatIfComparisonModel | None


class GridBounds(BaseModel):
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


class ForecastSummary(BaseModel):
    valid_time: datetime = Field(description="Forecast valid time (UTC); identifies the forecast")
    reference_time: datetime | None = Field(description="Model run time (UTC)")
    step_hours: float | None = Field(description="Forecast lead time in hours (valid - reference)")
    source_file: str = Field(description="Processed hazard NetCDF file")
    grib_file: str | None = Field(description="GRIB2 file the forecast was derived from")


class ForecastInfo(ForecastSummary):
    """The forecast a route was calculated on."""

    grid_bounds: GridBounds = Field(description="Area within which origin/destination must lie")


class ForecastListItem(ForecastSummary):
    is_default: bool = Field(description="Used when a route request names no forecast")


class ForecastListResponse(BaseModel):
    """Available forecasts, oldest valid time first."""

    default_valid_time: datetime | None = Field(
        description="Valid time of the default forecast; null if none are available"
    )
    forecasts: list[ForecastListItem]


class RouteResponse(BaseModel):
    """Distance-only and weather-aware routes plus their comparison.

    Coordinates are ``{"lat", "lon"}`` objects in decimal degrees, longitude
    in [-180, 180].
    """

    forecast: ForecastInfo
    origin: SnappedLocation
    destination: SnappedLocation
    distance_route: RouteInfo
    weather_route: RouteInfo
    comparison: RouteComparisonModel
    what_if: WhatIfInfo


class ErrorBody(BaseModel):
    code: str
    message: str
    details: list[dict] | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


def error_response(status: int, code: str, message: str, details=None) -> JSONResponse:
    body = {"error": {"code": code, "message": message}}
    if details is not None:
        body["error"]["details"] = details
    return JSONResponse(status_code=status, content=body)


# --- endpoints ----------------------------------------------------------

# (exception type, HTTP status, error code), checked in order.
ERROR_MAP = [
    (service.InvalidCoordinateError, 422, "invalid_coordinate"),
    (service.LocationOutsideGridError, 422, "outside_grid"),
    (service.SameNodeError, 422, "same_grid_point"),
    (service.InvalidHazardPenaltyError, 422, "invalid_hazard_penalty"),
    (service.ForecastNotFoundError, 422, "forecast_not_found"),
    (service.RouteNotFoundError, 422, "route_not_found"),
    (service.WeatherDataError, 503, "weather_data_unavailable"),
]


def service_error_response(exc: Exception, action: str) -> JSONResponse:
    """Map a service exception to the common error shape; hide unexpected ones."""
    for exc_type, status, code in ERROR_MAP:
        if isinstance(exc, exc_type):
            if status >= 500:
                logger.error("%s failed: %s", action.capitalize(), exc)
            details = None
            if isinstance(exc, service.ForecastNotFoundError):
                details = [{"valid_time": iso_utc(t)} for t in exc.available]
            return error_response(status, code, str(exc), details)
    logger.exception("Unexpected error while %s", action)
    return error_response(500, "internal_error", f"An unexpected error occurred while {action}.")


@router.get(
    "/forecasts",
    response_model=ForecastListResponse,
    summary="List the forecasts available for routing",
    description=(
        "One entry per processed hazard file, identified by its valid time and "
        "sorted oldest first. Pass an entry's valid_time as forecast_valid_time "
        "to POST /api/route to plan on that forecast."
    ),
    responses={
        503: {"model": ErrorResponse, "description": "Configured hazard file missing or invalid"},
        500: {"model": ErrorResponse, "description": "Unexpected server error"},
    },
)
def get_forecasts():
    try:
        return service.list_forecasts()
    except Exception as exc:
        return service_error_response(exc, "listing forecasts")


@router.post(
    "/route",
    response_model=RouteResponse,
    summary="Plan distance-only and weather-aware routes",
    description=(
        "Uses the forecast named by forecast_valid_time, or the default forecast "
        "(earliest valid time) when it is omitted. With hazard_penalty K, also plans a "
        "what-if weather route (multiplier 1 + K * hazard) and compares it with the "
        "normal weather-aware route in what_if. "
        "Snaps origin and destination to the nearest forecast grid points, runs A* "
        "twice (shortest distance; distance weighted by the experimental hazard score) "
        "and compares the two routes. The hazard score is an academic routing cost, "
        "not an operational turbulence forecast."
    ),
    responses={
        422: {"model": ErrorResponse, "description": "Invalid request, unknown forecast "
              "valid time, location outside the forecast grid, origin and destination on "
              "the same grid point, or no route"},
        503: {"model": ErrorResponse, "description": "Processed weather data missing or invalid"},
        500: {"model": ErrorResponse, "description": "Unexpected server error"},
    },
)
def plan_route(request: RouteRequest):
    try:
        return service.calculate_route(
            request.origin.lat, request.origin.lon,
            request.destination.lat, request.destination.lon,
            forecast_valid_time=request.forecast_valid_time,
            hazard_penalty=request.hazard_penalty,
        )
    except Exception as exc:
        return service_error_response(exc, "planning the route")
