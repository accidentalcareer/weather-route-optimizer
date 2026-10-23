"""A* search over the grid, parameterised by an edge cost function.

The same ``astar`` serves both route types: pass ``cost.distance_cost`` for
the shortest geographic route or ``cost.weather_cost`` for the weather-aware
route.

Heuristic and admissibility
---------------------------
The default heuristic is the haversine distance from a node to the goal.

* Distance-only routing: every path from n to the goal is at least as long
  as the great-circle distance between them, so h(n) never overestimates.
* Weather-aware routing: each edge costs distance * multiplier with
  multiplier >= 1, so any path's weather cost is >= its length, which is in
  turn >= the great-circle distance. h(n) is therefore still a lower bound.

The heuristic is also consistent (h(a) <= cost(a, b) + h(b)) in both modes,
by the triangle inequality and cost(a, b) >= distance(a, b). That means each
node is settled at most once and the first time the goal is popped its route
is optimal.

Scaling the heuristic up by the maximum multiplier (5) would NOT be
admissible: a route through calm air has multiplier close to 1, so 5x the
distance could overestimate its true cost.
"""

import heapq
import itertools
import math
from dataclasses import dataclass
from typing import Callable

from routing.grid import GeoGrid, Node

# edge_cost(grid, a, b) -> cost, or None if the edge cannot be traversed.
EdgeCost = Callable[[GeoGrid, Node, Node], float | None]
Heuristic = Callable[[GeoGrid, Node, Node], float]


class RouteNotFoundError(Exception):
    """No traversable route connects start and goal."""


@dataclass(frozen=True)
class RouteResult:
    """Outcome of one A* search.

    ``total_cost`` is in the edge cost's units (km for both built-in costs;
    hazard-inflated km for ``weather_cost``). ``path_length_km`` is always the
    plain geographic length of the route.
    """

    start: Node
    goal: Node
    nodes: list[Node]
    total_cost: float
    path_length_km: float
    nodes_expanded: int

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)


def haversine_heuristic(grid: GeoGrid, node: Node, goal: Node) -> float:
    """Great-circle distance in km from ``node`` to ``goal``."""
    return grid.distance_km(node, goal)


def path_length_km(grid: GeoGrid, nodes: list[Node]) -> float:
    """Sum of haversine distances between consecutive route nodes."""
    return sum(grid.distance_km(a, b) for a, b in zip(nodes, nodes[1:]))


def route_cost(grid: GeoGrid, nodes: list[Node], edge_cost: EdgeCost) -> float:
    """Total cost of an existing route under ``edge_cost``.

    Useful for scoring one route under another cost, e.g. the weather cost of
    the distance-only route.

    Raises:
        ValueError: consecutive nodes are not neighbours, or an edge is
            impassable (``edge_cost`` returned ``None``).
    """
    total = 0.0
    for a, b in zip(nodes, nodes[1:]):
        if b not in grid.neighbors(a):
            raise ValueError(f"Route step {a} -> {b} is not between neighbouring nodes")
        cost = _checked_edge_cost(edge_cost, grid, a, b)
        if cost is None:
            raise ValueError(f"Route step {a} -> {b} is impassable under this edge cost")
        total += cost
    return total


def _checked_edge_cost(edge_cost: EdgeCost, grid: GeoGrid, a: Node, b: Node) -> float | None:
    cost = edge_cost(grid, a, b)
    if cost is None:
        return None
    cost = float(cost)
    if not math.isfinite(cost) or cost <= 0:
        raise ValueError(f"Edge cost {a} -> {b} must be finite and positive, got {cost}")
    return cost


def _checked_heuristic(heuristic: Heuristic, grid: GeoGrid, node: Node, goal: Node) -> float:
    h = float(heuristic(grid, node, goal))
    if not math.isfinite(h) or h < 0:
        raise ValueError(f"Heuristic at {node} must be finite and non-negative, got {h}")
    return h


def _reconstruct(came_from: dict[Node, Node], goal: Node) -> list[Node]:
    """Follow parent links back from ``goal`` and return start -> goal order."""
    nodes = [goal]
    while nodes[-1] in came_from:
        nodes.append(came_from[nodes[-1]])
    nodes.reverse()
    return nodes


def astar(
    grid: GeoGrid,
    start: Node,
    goal: Node,
    edge_cost: EdgeCost,
    heuristic: Heuristic | None = None,
) -> RouteResult:
    """Find the minimum-cost route from ``start`` to ``goal``.

    Args:
        grid: routing grid; neighbours come from ``grid.neighbors``.
        start, goal: grid nodes ``(i, j)``.
        edge_cost: ``edge_cost(grid, a, b)`` returning a finite positive cost,
            or ``None`` to mark the edge as impassable.
        heuristic: ``heuristic(grid, node, goal)``; must not overestimate the
            remaining cost. Defaults to ``haversine_heuristic``.

    Raises:
        IndexError: start or goal is off the grid.
        ValueError: start == goal, or an edge cost / heuristic value is invalid.
        RouteNotFoundError: no traversable route exists.
    """
    start = grid.validate_node(start)
    goal = grid.validate_node(goal)
    if start == goal:
        raise ValueError(f"Start and goal must differ, both are {start}")
    heuristic = heuristic or haversine_heuristic

    g_score: dict[Node, float] = {start: 0.0}
    f_score: dict[Node, float] = {start: _checked_heuristic(heuristic, grid, start, goal)}
    came_from: dict[Node, Node] = {}
    closed: set[Node] = set()

    # Heap entries: (f, tie-breaker, node). Updating a node pushes a new entry;
    # outdated entries are skipped when popped (lazy deletion).
    tie = itertools.count()
    open_heap = [(f_score[start], next(tie), start)]
    nodes_expanded = 0

    while open_heap:
        f, _, node = heapq.heappop(open_heap)
        if node in closed or f > f_score[node]:
            continue
        if node == goal:
            nodes = _reconstruct(came_from, goal)
            return RouteResult(
                start=start,
                goal=goal,
                nodes=nodes,
                total_cost=g_score[goal],
                path_length_km=path_length_km(grid, nodes),
                nodes_expanded=nodes_expanded,
            )

        closed.add(node)
        nodes_expanded += 1

        for nb in grid.neighbors(node):
            if nb in closed:
                continue
            step = _checked_edge_cost(edge_cost, grid, node, nb)
            if step is None:
                continue
            tentative = g_score[node] + step
            if tentative < g_score.get(nb, math.inf):
                came_from[nb] = node
                g_score[nb] = tentative
                f_score[nb] = tentative + _checked_heuristic(heuristic, grid, nb, goal)
                heapq.heappush(open_heap, (f_score[nb], next(tie), nb))

    raise RouteNotFoundError(
        f"No route from {start} to {goal} ({nodes_expanded} nodes expanded)"
    )
