"""Edge cost functions: distance-only and distance + weighted hazard.

For an edge between neighbouring grid nodes ``a`` and ``b``:

    distance_km      = haversine(a, b)                      # geographic length
    edge_hazard      = (hazard_a + hazard_b) / 2            # mean endpoint hazard
    edge_multiplier  = 1 + 4 * edge_hazard                  # in [1, 5]
    weather_cost_km  = distance_km * edge_multiplier        # >= distance_km

The distance-only cost never reads weather data, so it gives the baseline
shortest-path route. The weather cost is a "hazard-inflated distance" in km:
crossing a cell with hazard 0.5 costs as much as flying three times as far
through calm air.

Because the multiplier is linear in hazard, averaging the endpoint hazards and
then converting is the same as averaging the endpoints' ``weather_multiplier``.

What-if scenarios replace the fixed 4 with a hazard penalty K:

    edge_multiplier  = 1 + K * edge_hazard                  # K in [0, 10]

``make_weather_cost(K)`` returns such an edge cost for ``astar``; for K = 4 it
returns ``weather_cost`` itself. Since K >= 0 keeps the multiplier >= 1, the
haversine heuristic stays admissible. The stored hazard_score is never changed.
"""

import math

from routing.grid import GeoGrid, Node
# The multiplier formula is defined once, in the hazard model. The constants
# are re-exported so existing ``routing.cost.MULTIPLIER_*`` imports keep working.
from weather.hazard_model import MULTIPLIER_MIN, MULTIPLIER_RANGE, weather_multiplier

__all__ = [
    "MULTIPLIER_MIN", "MULTIPLIER_RANGE",
    "HAZARD_PENALTY_MIN", "HAZARD_PENALTY_MAX", "DEFAULT_HAZARD_PENALTY",
    "distance_cost", "weather_cost", "edge_cost_details", "edge_multiplier",
    "make_weather_cost", "validate_hazard_penalty",
]

# What-if hazard penalty K (multiplier = 1 + K * hazard). The default is the
# normal routing formula's factor.
HAZARD_PENALTY_MIN = 0.0
HAZARD_PENALTY_MAX = 10.0
DEFAULT_HAZARD_PENALTY = MULTIPLIER_RANGE


def _validate_edge(grid: GeoGrid, node_a: Node, node_b: Node) -> tuple[Node, Node]:
    """Return both nodes as int tuples, or raise if invalid or identical."""
    try:
        a = grid.validate_node(node_a)
        b = grid.validate_node(node_b)
    except IndexError as exc:
        raise IndexError(f"Invalid edge {node_a} -> {node_b}: {exc}") from None
    if a == b:
        raise ValueError(f"Edge endpoints must be distinct nodes, got {a} twice")
    return a, b


def _hazard_at(grid: GeoGrid, node: Node) -> float:
    """Hazard score at ``node``, checked to be finite and within [0, 1]."""
    if grid.hazard is None:
        raise ValueError(
            "Grid has no hazard field; build it with load_hazard_grid() or "
            "GeoGrid(..., hazard=...)"
        )
    h = float(grid.hazard[node])
    if not math.isfinite(h):
        raise ValueError(f"Hazard at node {node} is not finite ({h})")
    if not 0.0 <= h <= 1.0:
        raise ValueError(f"Hazard at node {node} is {h}, outside [0, 1]")
    return h


def edge_multiplier(edge_hazard: float) -> float:
    """Weather multiplier for an edge: 1 + 4 * edge_hazard.

    Applies ``hazard_model.weather_multiplier`` to the edge's mean hazard.
    """
    return weather_multiplier(edge_hazard)


def distance_cost(grid: GeoGrid, node_a: Node, node_b: Node) -> float:
    """Distance-only edge cost: haversine distance in km. Ignores weather."""
    a, b = _validate_edge(grid, node_a, node_b)
    d = grid.distance_km(a, b)
    if d <= 0:
        raise ValueError(f"Non-positive distance {d} km between {a} and {b}")
    return d


def edge_cost_details(grid: GeoGrid, node_a: Node, node_b: Node) -> dict:
    """All components of the weather-aware edge cost, for explainability.

    Returns a dict with ``node_a``, ``node_b``, ``distance_km``, ``hazard_a``,
    ``hazard_b``, ``edge_hazard``, ``weather_multiplier`` and ``weather_cost_km``.
    """
    a, b = _validate_edge(grid, node_a, node_b)
    distance_km = distance_cost(grid, a, b)
    hazard_a = _hazard_at(grid, a)
    hazard_b = _hazard_at(grid, b)
    edge_hazard = (hazard_a + hazard_b) / 2
    multiplier = edge_multiplier(edge_hazard)
    return {
        "node_a": a,
        "node_b": b,
        "distance_km": distance_km,
        "hazard_a": hazard_a,
        "hazard_b": hazard_b,
        "edge_hazard": edge_hazard,
        "weather_multiplier": multiplier,
        "weather_cost_km": distance_km * multiplier,
    }


def weather_cost(grid: GeoGrid, node_a: Node, node_b: Node) -> float:
    """Weather-aware edge cost in km: distance * (1 + 4 * mean endpoint hazard)."""
    return edge_cost_details(grid, node_a, node_b)["weather_cost_km"]


def validate_hazard_penalty(hazard_penalty) -> float:
    """``hazard_penalty`` as a float, or ``ValueError`` if not finite and in [0, 10]."""
    if isinstance(hazard_penalty, bool) or not isinstance(hazard_penalty, (int, float)):
        raise ValueError(f"Hazard penalty must be a number, got {hazard_penalty!r}")
    k = float(hazard_penalty)
    if not math.isfinite(k) or not HAZARD_PENALTY_MIN <= k <= HAZARD_PENALTY_MAX:
        raise ValueError(
            f"Hazard penalty must be between {HAZARD_PENALTY_MIN:g} and "
            f"{HAZARD_PENALTY_MAX:g}, got {hazard_penalty}"
        )
    return k


def make_weather_cost(hazard_penalty: float = DEFAULT_HAZARD_PENALTY):
    """Weather-aware edge cost with multiplier ``1 + hazard_penalty * edge_hazard``.

    Returns ``weather_cost`` unchanged for the default penalty (4), so normal
    routing is exactly the existing behaviour.
    """
    k = validate_hazard_penalty(hazard_penalty)
    if k == DEFAULT_HAZARD_PENALTY:
        return weather_cost

    def scenario_weather_cost(grid: GeoGrid, node_a: Node, node_b: Node) -> float:
        d = edge_cost_details(grid, node_a, node_b)
        return d["distance_km"] * (MULTIPLIER_MIN + k * d["edge_hazard"])

    scenario_weather_cost.hazard_penalty = k
    return scenario_weather_cost
