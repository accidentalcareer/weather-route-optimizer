"""Compare a distance-only route with a weather-aware route.

``compare_routes`` scores both routes on the same grid and returns:

* distance metrics      - geographic length of each route and the detour
* weather-cost metrics  - each route's weather-weighted cost (``route_cost``
                          with ``cost.weather_cost``) and the reduction
* hazard metrics        - mean / max hazard and per-category cell counts
                          along each route, plus cells avoided
* fuel proxy            - extra distance as a percentage. This is a
                          DISTANCE-BASED PROXY ONLY, not an aircraft
                          fuel-burn prediction: the project has no aircraft
                          performance, wind-component or altitude model.
* a summary for display and a plain-language explanation

Cell counts are taken over the grid nodes a route visits, endpoints included.
The hazard model is experimental and the weather cost is a routing score, so
none of these figures are operational aviation guidance.
"""

import math
from dataclasses import asdict, dataclass

import numpy as np
import xarray as xr

from routing.astar import RouteResult, path_length_km, route_cost
from routing.cost import distance_cost, weather_cost
from routing.grid import GeoGrid, Node
from weather.hazard_model import CATEGORY_LABELS, categorize

FUEL_PROXY_NOTE = (
    "Distance-based fuel proxy only: percentage change in route length. "
    "Not an operational aircraft fuel-burn prediction."
)

# A route's stored length must match the grid within this tolerance, which
# catches routes computed on a different grid.
LENGTH_RTOL = 1e-6


@dataclass(frozen=True)
class RouteMetrics:
    """Per-route figures."""

    n_nodes: int
    length_km: float
    weather_cost: float
    average_hazard: float
    maximum_hazard: float
    category_counts: dict  # {"LOW": n, "MODERATE": n, "HIGH": n, "SEVERE": n}


@dataclass(frozen=True)
class RouteComparison:
    """Distance-only vs weather-aware route comparison."""

    start: Node
    goal: Node
    routes_identical: bool
    distance_route: RouteMetrics
    weather_route: RouteMetrics

    # Distance
    distance_route_km: float
    weather_route_km: float
    extra_distance_km: float
    extra_distance_percent: float

    # Weather cost
    distance_route_weather_cost: float
    weather_route_weather_cost: float
    weather_cost_difference: float        # distance-route cost minus weather-route cost
    weather_cost_reduction_percent: float

    # Hazard (positive = fewer such cells on the weather route; may be negative)
    severe_cells_avoided: int
    high_hazard_cells_avoided: int        # HIGH + SEVERE
    average_hazard_reduction: float
    maximum_hazard_reduction: float

    # Fuel proxy
    estimated_fuel_delta_percent: float
    fuel_proxy_note: str

    explanation: str

    def summary(self) -> dict:
        """Headline figures for display."""
        return {
            "extra_distance_km": self.extra_distance_km,
            "extra_distance_percent": self.extra_distance_percent,
            "weather_cost_reduction_percent": self.weather_cost_reduction_percent,
            "severe_cells_avoided": self.severe_cells_avoided,
            "high_hazard_cells_avoided": self.high_hazard_cells_avoided,
            "estimated_fuel_delta_percent": self.estimated_fuel_delta_percent,
            "fuel_proxy_note": self.fuel_proxy_note,
            "routes_identical": self.routes_identical,
            "explanation": self.explanation,
        }

    def to_dict(self) -> dict:
        """Everything as plain JSON-serialisable types (nodes become lists)."""
        out = asdict(self)
        out["start"], out["goal"] = list(self.start), list(self.goal)
        out["summary"] = self.summary()
        return out


# --- validation ---------------------------------------------------------

def _route_nodes(route) -> list[Node]:
    nodes = route.nodes if isinstance(route, RouteResult) else route
    return [tuple(int(x) for x in n) for n in nodes]


def validate_route(grid: GeoGrid, route, name: str = "route") -> list[Node]:
    """Check a route (``RouteResult`` or node list) is a valid walk on ``grid``.

    Returns the nodes as a list of int tuples.
    """
    nodes = _route_nodes(route)
    if len(nodes) < 2:
        raise ValueError(f"{name} must contain at least two nodes, got {len(nodes)}")
    for n in nodes:
        if not grid.in_bounds(n):
            raise ValueError(f"{name} contains node {n} outside grid of shape {grid.shape}")
    for a, b in zip(nodes, nodes[1:]):
        if b not in grid.neighbors(a):
            raise ValueError(f"{name}: consecutive nodes {a} -> {b} are not neighbours")
        if grid.distance_km(a, b) <= 0:
            raise ValueError(f"{name}: non-positive distance between {a} and {b}")

    if grid.hazard is None:
        raise ValueError("Grid has no hazard field")
    hazards = np.array([grid.hazard[n] for n in nodes])
    if not np.all(np.isfinite(hazards)) or hazards.min() < 0 or hazards.max() > 1:
        raise ValueError(f"{name} visits nodes with hazard outside [0, 1] or non-finite")

    # A RouteResult remembers its own length; it must agree with this grid.
    if isinstance(route, RouteResult):
        length = path_length_km(grid, nodes)
        if not math.isclose(route.path_length_km, length, rel_tol=LENGTH_RTOL):
            raise ValueError(
                f"{name}: stored length {route.path_length_km} km does not match "
                f"{length} km on this grid; was it computed on a different grid?"
            )
    return nodes


# --- per-route metrics --------------------------------------------------

def route_metrics(grid: GeoGrid, nodes: list[Node]) -> RouteMetrics:
    """Length, weather cost and hazard exposure of one (validated) route."""
    hazards = np.array([grid.hazard[n] for n in nodes])
    # Same category rule as the hazard model, so counts match hazard_category.
    codes = categorize(xr.DataArray(hazards, dims="node")).values
    counts = {label: int((codes == i).sum()) for i, label in enumerate(CATEGORY_LABELS)}
    return RouteMetrics(
        n_nodes=len(nodes),
        length_km=route_cost(grid, nodes, distance_cost),
        weather_cost=route_cost(grid, nodes, weather_cost),
        average_hazard=float(hazards.mean()),
        maximum_hazard=float(hazards.max()),
        category_counts=counts,
    )


def _percent(part: float, whole: float) -> float:
    """100 * part / whole, or 0 when ``whole`` is 0 (nothing to compare against)."""
    return 100.0 * part / whole if whole else 0.0


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def explain(c: RouteComparison) -> str:
    """One-sentence, data-driven description of the trade-off."""
    if c.routes_identical:
        return (
            "The weather-aware route is identical to the distance-only route, "
            "so there is no distance or weather trade-off for this pair."
        )

    if c.extra_distance_km > 0:
        dist = f"adds {c.extra_distance_km:.1f} km ({c.extra_distance_percent:.1f}%)"
    elif c.extra_distance_km < 0:
        dist = f"is {-c.extra_distance_km:.1f} km ({-c.extra_distance_percent:.1f}%) shorter"
    else:
        dist = "is the same length"

    if c.weather_cost_reduction_percent >= 0:
        cost = f"reduces weather-weighted cost by {c.weather_cost_reduction_percent:.1f}%"
    else:
        cost = f"increases weather-weighted cost by {-c.weather_cost_reduction_percent:.1f}%"

    n = c.severe_cells_avoided
    if n > 0:
        severe = f"avoids {_plural(n, 'severe-hazard cell')}"
    elif n < 0:
        severe = f"crosses {_plural(-n, 'more severe-hazard cell')}"
    else:
        severe = "crosses the same number of severe-hazard cells"

    return f"Weather-aware route {dist} but {cost} and {severe}."


# --- comparison ---------------------------------------------------------

def compare_routes(grid: GeoGrid, distance_route, weather_route) -> RouteComparison:
    """Compare a distance-only route with a weather-aware route on ``grid``.

    Both routes may be ``RouteResult`` objects or lists of ``(i, j)`` nodes.
    They must share start and goal and be valid walks on ``grid``.
    """
    d_nodes = validate_route(grid, distance_route, "distance_route")
    w_nodes = validate_route(grid, weather_route, "weather_route")
    if d_nodes[0] != w_nodes[0] or d_nodes[-1] != w_nodes[-1]:
        raise ValueError(
            f"Routes must share start and goal: distance {d_nodes[0]}->{d_nodes[-1]}, "
            f"weather {w_nodes[0]}->{w_nodes[-1]}"
        )

    dm = route_metrics(grid, d_nodes)
    wm = route_metrics(grid, w_nodes)

    extra_km = wm.length_km - dm.length_km
    cost_diff = dm.weather_cost - wm.weather_cost
    severe = lambda m: m.category_counts["SEVERE"]  # noqa: E731
    high_plus = lambda m: m.category_counts["HIGH"] + m.category_counts["SEVERE"]  # noqa: E731

    comparison = RouteComparison(
        start=d_nodes[0],
        goal=d_nodes[-1],
        routes_identical=d_nodes == w_nodes,
        distance_route=dm,
        weather_route=wm,
        distance_route_km=dm.length_km,
        weather_route_km=wm.length_km,
        extra_distance_km=extra_km,
        extra_distance_percent=_percent(extra_km, dm.length_km),
        distance_route_weather_cost=dm.weather_cost,
        weather_route_weather_cost=wm.weather_cost,
        weather_cost_difference=cost_diff,
        weather_cost_reduction_percent=_percent(cost_diff, dm.weather_cost),
        severe_cells_avoided=severe(dm) - severe(wm),
        high_hazard_cells_avoided=high_plus(dm) - high_plus(wm),
        average_hazard_reduction=dm.average_hazard - wm.average_hazard,
        maximum_hazard_reduction=dm.maximum_hazard - wm.maximum_hazard,
        # Distance-based proxy: same formula as extra_distance_percent.
        estimated_fuel_delta_percent=_percent(extra_km, dm.length_km),
        fuel_proxy_note=FUEL_PROXY_NOTE,
        explanation="",
    )
    _check_finite(comparison)
    # Frozen dataclass: rebuild with the explanation filled in.
    return RouteComparison(**{**comparison.__dict__, "explanation": explain(comparison)})


def _check_finite(c: RouteComparison) -> None:
    values = {k: v for k, v in c.__dict__.items() if isinstance(v, (int, float))}
    for prefix, m in (("distance_route", c.distance_route), ("weather_route", c.weather_route)):
        for k, v in m.__dict__.items():
            if isinstance(v, (int, float)):
                values[f"{prefix}.{k}"] = v
    bad = {k: v for k, v in values.items() if not math.isfinite(v)}
    if bad:
        raise ValueError(f"Non-finite metrics: {bad}")


# --- what-if scenario ---------------------------------------------------

@dataclass(frozen=True)
class WhatIfComparison:
    """Normal weather-aware route vs a route planned with hazard penalty K.

    Both routes are scored with the same metrics as ``compare_routes``, so
    ``weather_cost`` is on the normal (K = 4) scale for both. Differences are
    baseline minus what-if: positive means the what-if route has less.
    """

    hazard_penalty: float
    routes_identical: bool
    baseline_route: RouteMetrics
    what_if_route: RouteMetrics

    extra_distance_km: float              # what-if length minus baseline length
    extra_distance_percent: float
    weather_cost_difference: float        # baseline minus what-if, K = 4 scale
    weather_cost_reduction_percent: float
    average_hazard_reduction: float
    average_hazard_reduction_percent: float
    maximum_hazard_reduction: float
    severe_cells_avoided: int
    high_hazard_cells_avoided: int
    explanation: str


def _k(value: float) -> str:
    return f"{value:.1f}"


def explain_what_if(c: WhatIfComparison) -> str:
    """One-sentence description of the what-if route against the normal one."""
    k = _k(c.hazard_penalty)
    if c.routes_identical:
        return (
            f"With hazard sensitivity K={k}, the what-if route is identical to the "
            "normal weather-aware route (K=4.0)."
        )

    # Differences that round to zero at the displayed precision count as none.
    extra = round(c.extra_distance_km, 1)
    if extra > 0:
        dist = f"adds {c.extra_distance_km:.1f} km ({c.extra_distance_percent:.1f}%)"
    elif extra < 0:
        dist = f"saves {-c.extra_distance_km:.1f} km ({-c.extra_distance_percent:.1f}%)"
    else:
        dist = "has about the same length"

    base, new = c.baseline_route.average_hazard, c.what_if_route.average_hazard
    r = round(c.average_hazard_reduction_percent, 1)
    if base == 0 and round(new, 3) > 0:
        # No percentage of a zero baseline; give the absolute change instead.
        hazard, change = f"raising average hazard from 0 to {new:.3f}", -1
    elif r > 0:
        hazard, change = f"reducing average hazard by {r:.1f}%", 1
    elif r < 0:
        hazard, change = f"increasing average hazard by {-r:.1f}%", -1
    else:
        hazard, change = "with about the same average hazard", 0

    lead = (
        f"With hazard sensitivity K={k}, the what-if route {dist} compared with "
        "the normal weather-aware route"
    )
    # "while" marks a trade-off: longer but calmer, or shorter but rougher.
    if (extra > 0 and change > 0) or (extra < 0 and change < 0):
        return f"{lead} while {hazard}."
    return f"{lead}, {hazard}."


def compare_what_if(grid: GeoGrid, baseline_route, what_if_route,
                    hazard_penalty: float) -> WhatIfComparison:
    """Compare the normal weather-aware route with a what-if route on ``grid``.

    Routes may be ``RouteResult`` objects or node lists sharing start and goal.
    """
    b_nodes = validate_route(grid, baseline_route, "baseline_route")
    w_nodes = validate_route(grid, what_if_route, "what_if_route")
    if b_nodes[0] != w_nodes[0] or b_nodes[-1] != w_nodes[-1]:
        raise ValueError(
            f"Routes must share start and goal: baseline {b_nodes[0]}->{b_nodes[-1]}, "
            f"what-if {w_nodes[0]}->{w_nodes[-1]}"
        )

    bm = route_metrics(grid, b_nodes)
    wm = route_metrics(grid, w_nodes)
    extra_km = wm.length_km - bm.length_km
    cost_diff = bm.weather_cost - wm.weather_cost
    avg_diff = bm.average_hazard - wm.average_hazard
    severe = lambda m: m.category_counts["SEVERE"]  # noqa: E731
    high_plus = lambda m: m.category_counts["HIGH"] + m.category_counts["SEVERE"]  # noqa: E731

    comparison = WhatIfComparison(
        hazard_penalty=float(hazard_penalty),
        routes_identical=b_nodes == w_nodes,
        baseline_route=bm,
        what_if_route=wm,
        extra_distance_km=extra_km,
        extra_distance_percent=_percent(extra_km, bm.length_km),
        weather_cost_difference=cost_diff,
        weather_cost_reduction_percent=_percent(cost_diff, bm.weather_cost),
        average_hazard_reduction=avg_diff,
        average_hazard_reduction_percent=_percent(avg_diff, bm.average_hazard),
        maximum_hazard_reduction=bm.maximum_hazard - wm.maximum_hazard,
        severe_cells_avoided=severe(bm) - severe(wm),
        high_hazard_cells_avoided=high_plus(bm) - high_plus(wm),
        explanation="",
    )
    values = {k: v for k, v in comparison.__dict__.items() if isinstance(v, (int, float))}
    bad = {k: v for k, v in values.items() if not math.isfinite(v)}
    if bad:
        raise ValueError(f"Non-finite what-if metrics: {bad}")
    return WhatIfComparison(**{**comparison.__dict__, "explanation": explain_what_if(comparison)})
