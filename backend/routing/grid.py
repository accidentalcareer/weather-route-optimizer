"""Lightweight geographic grid for routing over a regular lat/lon field.

Each (latitude, longitude) grid point is a node, identified by its integer
index pair ``(i, j)``: ``i`` indexes the latitude coordinate and ``j`` the
longitude coordinate, in the order they appear in the NetCDF. Nodes connect to
their 8 neighbours (N, NE, E, SE, S, SW, W, NW).

No explicit edge list is stored: neighbours are generated on demand from the
index, and edge lengths are computed geographically with the haversine formula,
so diagonal steps (and east-west steps, which shrink with latitude) get their
true great-circle length. For a 121 x 121 grid that is ~14.6k nodes and ~116k
directed edges, which would cost far more memory as a graph object than as two
coordinate arrays.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

from weather.grib_reader import processed_dir

# IUGG mean Earth radius.
EARTH_RADIUS_KM = 6371.0088

Node = tuple[int, int]

# Compass moves as (north, east) steps. These are translated into index
# offsets per grid, because latitude may be stored south->north or north->south.
COMPASS_MOVES = {
    "N": (1, 0), "NE": (1, 1), "E": (0, 1), "SE": (-1, 1),
    "S": (-1, 0), "SW": (-1, -1), "W": (0, -1), "NW": (1, -1),
}


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between points given in degrees.

    a = sin^2(dlat/2) + cos(lat1) * cos(lat2) * sin^2(dlon/2)
    d = 2 R asin(sqrt(a))

    Works on scalars or NumPy arrays. Because dlon enters through sin^2, the
    result is the same whether longitudes use -180..180 or 0..360.
    """
    lat1, lon1, lat2, lon2 = (np.radians(x) for x in (lat1, lon1, lat2, lon2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    # Clip guards against a > 1 from floating-point rounding.
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _step_direction(coord: np.ndarray, name: str) -> int:
    """+1 if ``coord`` increases with index, -1 if it decreases."""
    diffs = np.diff(coord)
    if np.all(diffs > 0):
        return 1
    if np.all(diffs < 0):
        return -1
    raise ValueError(f"{name} coordinate must be strictly monotonic")


@dataclass(frozen=True)
class GeoGrid:
    """A rectilinear lat/lon grid with 8-neighbour connectivity.

    Build it with ``GeoGrid.from_dataset`` or ``load_hazard_grid`` so the
    coordinates come straight from the NetCDF; they are kept in file order.

    ``hazard`` is an optional ``(n_lat, n_lon)`` array of per-node hazard
    scores, indexed by node ``(i, j)``. Distance calculations never use it.
    """

    lats: np.ndarray
    lons: np.ndarray
    hazard: np.ndarray | None = None

    def __post_init__(self):
        lats = np.asarray(self.lats, dtype=float)
        lons = np.asarray(self.lons, dtype=float)
        for name, c in (("latitude", lats), ("longitude", lons)):
            if c.ndim != 1:
                raise ValueError(f"{name} coordinate must be 1-D, got shape {c.shape}")
            if c.size == 0:
                raise ValueError(f"{name} coordinate is empty")
            if not np.all(np.isfinite(c)):
                raise ValueError(f"{name} coordinate contains non-finite values")
        if lats.min() < -90 or lats.max() > 90:
            raise ValueError("latitude values must be within [-90, 90]")
        lat_dir = _step_direction(lats, "latitude") if lats.size > 1 else 1
        lon_dir = _step_direction(lons, "longitude") if lons.size > 1 else 1
        # frozen dataclass: set derived fields through object.__setattr__.
        object.__setattr__(self, "lats", lats)
        object.__setattr__(self, "lons", lons)
        object.__setattr__(self, "_lat_dir", lat_dir)
        object.__setattr__(self, "_lon_dir", lon_dir)
        # Translate compass moves into (di, dj) index offsets for this grid.
        if self.hazard is not None:
            hazard = np.asarray(self.hazard, dtype=float)
            if hazard.shape != (lats.size, lons.size):
                raise ValueError(
                    f"hazard shape {hazard.shape} does not match grid {(lats.size, lons.size)}"
                )
            object.__setattr__(self, "hazard", hazard)
        object.__setattr__(self, "_offsets", {
            name: (dn * lat_dir, de * lon_dir) for name, (dn, de) in COMPASS_MOVES.items()
        })

    @classmethod
    def from_dataset(cls, ds: xr.Dataset, hazard_var: str = "hazard_score") -> "GeoGrid":
        """Build a grid from a Dataset's ``latitude``/``longitude`` coordinates.

        If ``hazard_var`` is present it is attached as ``grid.hazard``.
        """
        for coord in ("latitude", "longitude"):
            if coord not in ds.coords:
                raise ValueError(f"Dataset is missing the '{coord}' coordinate")
        hazard = None
        if hazard_var in ds.data_vars:
            hazard = ds[hazard_var].transpose("latitude", "longitude").values
        return cls(ds["latitude"].values, ds["longitude"].values, hazard)

    # --- size and bounds -------------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        return (self.lats.size, self.lons.size)

    @property
    def n_nodes(self) -> int:
        return self.lats.size * self.lons.size

    def n_directed_edges(self) -> int:
        """Total neighbour links summed over all nodes (each pair counted twice)."""
        r, c = self.shape
        undirected = r * (c - 1) + (r - 1) * c + 2 * (r - 1) * (c - 1)
        return 2 * undirected

    def in_bounds(self, node: Node) -> bool:
        i, j = node
        return 0 <= i < self.lats.size and 0 <= j < self.lons.size

    def validate_node(self, node: Node) -> Node:
        """Return ``node`` as a tuple of ints, or raise if it is off the grid."""
        i, j = int(node[0]), int(node[1])
        if not self.in_bounds((i, j)):
            raise IndexError(f"Node {(i, j)} outside grid of shape {self.shape}")
        return (i, j)

    # --- coordinate conversion ------------------------------------------

    def index_to_latlon(self, node: Node) -> tuple[float, float]:
        """Grid index ``(i, j)`` -> ``(lat, lon)`` in degrees."""
        i, j = self.validate_node(node)
        return float(self.lats[i]), float(self.lons[j])

    def _to_grid_lon(self, lon: float) -> float:
        """Express ``lon`` in the grid's longitude convention (0..360 or -180..180)."""
        if self.lons.max() > 180:
            return lon % 360.0
        return ((lon + 180.0) % 360.0) - 180.0

    def latlon_to_index(self, lat: float, lon: float, atol: float = 1e-6) -> Node:
        """``(lat, lon)`` of an existing grid point -> its index ``(i, j)``.

        The point must coincide with a grid point to within ``atol`` degrees;
        use ``nearest_node`` for arbitrary locations.
        """
        lon = self._to_grid_lon(lon)
        i_hits = np.flatnonzero(np.isclose(self.lats, lat, rtol=0, atol=atol))
        j_hits = np.flatnonzero(np.isclose(self.lons, lon, rtol=0, atol=atol))
        if i_hits.size == 0 or j_hits.size == 0:
            raise KeyError(f"({lat}, {lon}) is not a grid point; use nearest_node()")
        return (int(i_hits[0]), int(j_hits[0]))

    def nearest_node(self, lat: float, lon: float) -> Node:
        """Grid node geographically closest (haversine) to ``(lat, lon)``.

        Raises ``ValueError`` if the point lies more than half a grid spacing
        outside the grid's extent, since snapping it would silently move it.

        On a rectilinear grid, for any fixed row the closest column is the one
        with the smallest longitude difference, so we pick that column and then
        compare great-circle distance down it to choose the row.
        """
        lon = self._to_grid_lon(lon)
        self._check_within_extent(lat, lon)
        j = int(np.argmin(np.abs(self.lons - lon)))
        i = int(np.argmin(haversine_km(lat, lon, self.lats, self.lons[j])))
        return self.validate_node((i, j))

    def _check_within_extent(self, lat: float, lon: float) -> None:
        for name, value, coord in (("latitude", lat, self.lats), ("longitude", lon, self.lons)):
            half = np.abs(np.diff(coord)).max() / 2 if coord.size > 1 else 0.0
            lo, hi = coord.min() - half, coord.max() + half
            if not lo <= value <= hi:
                raise ValueError(f"{name} {value} outside grid extent [{lo}, {hi}]")

    # --- connectivity and distance --------------------------------------

    def neighbors(self, node: Node) -> list[Node]:
        """In-bounds 8-neighbours of ``node``, in compass order N, NE, E, ..., NW."""
        i, j = self.validate_node(node)
        out = []
        for di, dj in self._offsets.values():
            nb = (i + di, j + dj)
            if self.in_bounds(nb):
                out.append(nb)
        return out

    def neighbors_by_direction(self, node: Node) -> dict[str, Node]:
        """Like ``neighbors`` but keyed by compass direction ("N", "NE", ...)."""
        i, j = self.validate_node(node)
        return {
            name: (i + di, j + dj)
            for name, (di, dj) in self._offsets.items()
            if self.in_bounds((i + di, j + dj))
        }

    def distance_km(self, a: Node, b: Node) -> float:
        """Haversine distance in km between two grid nodes."""
        lat1, lon1 = self.index_to_latlon(a)
        lat2, lon2 = self.index_to_latlon(b)
        d = float(haversine_km(lat1, lon1, lat2, lon2))
        if a != b and d <= 0:
            raise ValueError(f"Non-positive distance {d} between distinct nodes {a}, {b}")
        return d


def load_hazard_grid(path) -> tuple[GeoGrid, xr.Dataset]:
    """Load a hazard NetCDF and build its routing grid.

    A bare filename is looked up in ``processed_dir()``. Returns the grid and
    the loaded Dataset so later layers can read per-node weather fields with
    ``ds[var].values[i, j]``, which lines up with the grid's ``(i, j)`` nodes.
    """
    path = Path(path)
    if not path.is_file() and not path.is_absolute():
        path = processed_dir() / path
    if not path.is_file():
        raise FileNotFoundError(f"Hazard file not found: {path}")
    with xr.open_dataset(path, engine="scipy") as ds:
        ds = ds.load()
    grid = GeoGrid.from_dataset(ds)
    for name, da in ds.data_vars.items():
        if da.dims == ("latitude", "longitude"):
            continue
        raise ValueError(f"{name}: expected dims ('latitude', 'longitude'), got {da.dims}")
    return grid, ds
