"""Import checks for the canonical backend/routing modules."""


def test_routing_modules_import():
    from routing import astar, cost, grid, metrics  # noqa: F401


def test_routing_entry_points_exist():
    from routing.astar import RouteNotFoundError, RouteResult, astar, route_cost
    from routing.cost import distance_cost, edge_cost_details, weather_cost
    from routing.grid import GeoGrid, haversine_km, load_hazard_grid
    from routing.metrics import RouteComparison, compare_routes

    for obj in (astar, route_cost, distance_cost, weather_cost, edge_cost_details,
                haversine_km, load_hazard_grid, compare_routes):
        assert callable(obj)
    for cls in (RouteResult, RouteNotFoundError, GeoGrid, RouteComparison):
        assert isinstance(cls, type)
