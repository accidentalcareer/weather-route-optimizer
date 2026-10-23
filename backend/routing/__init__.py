"""Routing layer: geographic grid, edge costs, A* search, and route metrics.

Modules and their main entry points:

    routing.grid     GeoGrid, load_hazard_grid, haversine_km
    routing.cost     distance_cost, weather_cost, edge_cost_details
    routing.astar    astar, route_cost, RouteResult, RouteNotFoundError
    routing.metrics  compare_routes, RouteComparison

Import from the submodules, e.g. ``from routing.astar import astar``. Names
are not re-exported here: the ``astar`` function would shadow the ``astar``
module, and importing the package would pull in xarray eagerly.
"""

__all__ = ["grid", "cost", "astar", "metrics"]
