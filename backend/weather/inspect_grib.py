"""Temporary inspection script for the raw GFS GRIB2 file.

Usage:
    python backend/weather/inspect_grib.py [path/to/file.grib2]
"""

import sys
from collections import Counter
from pathlib import Path

import cfgrib
import eccodes
import numpy as np
import xarray as xr

DEFAULT_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "raw" / "gfs.t12z.pgrb2.0p25.f006"
)

# Don't write .idx sidecar files next to the raw data.
BACKEND_KWARGS = {"indexpath": ""}


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def inspect_messages(path):
    """List every GRIB message using eccodes directly (no hypercube assumptions)."""
    section("RAW GRIB MESSAGES (eccodes)")
    keys = [
        "shortName", "name", "units", "typeOfLevel", "level",
        "stepRange", "dataDate", "dataTime", "Ni", "Nj",
    ]
    rows = []
    with open(path, "rb") as f:
        while True:
            gid = eccodes.codes_grib_new_from_file(f)
            if gid is None:
                break
            try:
                row = {}
                for k in keys:
                    try:
                        row[k] = eccodes.codes_get_string(gid, k)
                    except eccodes.KeyValueNotFoundError:
                        row[k] = "?"
                if not rows:
                    grid = {}
                    for k in (
                        "gridType",
                        "latitudeOfFirstGridPointInDegrees",
                        "latitudeOfLastGridPointInDegrees",
                        "longitudeOfFirstGridPointInDegrees",
                        "longitudeOfLastGridPointInDegrees",
                        "iDirectionIncrementInDegrees",
                        "jDirectionIncrementInDegrees",
                    ):
                        try:
                            grid[k] = eccodes.codes_get_string(gid, k)
                        except eccodes.KeyValueNotFoundError:
                            grid[k] = "?"
                rows.append(row)
            finally:
                eccodes.codes_release(gid)

    print(f"Total messages: {len(rows)}")
    print("\nGrid (from first message):")
    for k, v in grid.items():
        print(f"  {k}: {v}")

    print(f"\n{'#':>3}  {'shortName':<10} {'typeOfLevel':<28} {'level':>6}  "
          f"{'step':>5}  {'date':<8} {'time':<4}  {'NixNj':<9} name [units]")
    for i, r in enumerate(rows, 1):
        print(f"{i:>3}  {r['shortName']:<10} {r['typeOfLevel']:<28} {r['level']:>6}  "
              f"{r['stepRange']:>5}  {r['dataDate']:<8} {r['dataTime']:<4}  "
              f"{r['Ni'] + 'x' + r['Nj']:<9} {r['name']} [{r['units']}]")

    print("\nMessages per (shortName, typeOfLevel):")
    for (sn, tl), n in sorted(Counter((r["shortName"], r["typeOfLevel"]) for r in rows).items()):
        print(f"  {sn:<10} {tl:<28} {n}")


def describe_coord(name, coord):
    vals = coord.values
    if vals.size == 0:
        return f"{name}: (empty)"
    if vals.size == 1:
        return f"{name}: {vals.item()!r}"
    if np.issubdtype(vals.dtype, np.number):
        return f"{name}: n={vals.size}, min={vals.min()}, max={vals.max()}"
    return f"{name}: n={vals.size}, first={vals.flat[0]!r}, last={vals.flat[-1]!r}"


def inspect_dataset(i, ds):
    section(f"DATASET {i}")
    print(f"Dimensions: {dict(ds.sizes)}")
    print(f"Coordinates: {list(ds.coords)}")
    print(f"Data variables: {list(ds.data_vars)}")

    print("\nCoordinate details:")
    for name, coord in ds.coords.items():
        print("  " + describe_coord(name, coord))

    if "latitude" in ds.coords:
        lat = ds["latitude"].values
        print(f"\nLatitude range:  {lat.min()} .. {lat.max()} (n={lat.size})")
    if "longitude" in ds.coords:
        lon = ds["longitude"].values
        print(f"Longitude range: {lon.min()} .. {lon.max()} (n={lon.size})")

    print("\nTime information:")
    for name in ("time", "step", "valid_time"):
        if name in ds.coords:
            print(f"  {name}: {ds[name].values}")

    level_coords = [c for c in ds.coords if c not in
                    ("time", "step", "valid_time", "latitude", "longitude")]
    for name in level_coords:
        attrs = ds[name].attrs
        print(f"\nLevel coord '{name}' ({attrs.get('long_name', '')}, "
              f"units={attrs.get('units', '?')}): {np.atleast_1d(ds[name].values).tolist()}")

    print("\nVariables:")
    for name, da in ds.data_vars.items():
        a = da.attrs
        print(f"  {name}")
        print(f"    long_name: {a.get('long_name', a.get('GRIB_name', '?'))}")
        print(f"    units:     {a.get('units', a.get('GRIB_units', '?'))}")
        print(f"    GRIB:      shortName={a.get('GRIB_shortName')}, "
              f"paramId={a.get('GRIB_paramId')}, typeOfLevel={a.get('GRIB_typeOfLevel')}")
        print(f"    dims:      {da.dims}")
        print(f"    shape:     {da.shape}")
        if np.issubdtype(da.dtype, np.number):
            vals = da.values
            n_nan = int(np.isnan(vals).sum())
            if n_nan < vals.size:
                print(f"    min/max:   {np.nanmin(vals):.6g} / {np.nanmax(vals):.6g}"
                      f"  (mean {np.nanmean(vals):.6g}, NaNs {n_nan})")
            else:
                print("    min/max:   all NaN")


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    section("FILE")
    print(f"Path:   {path}")
    if not path.is_file():
        print("ERROR: file does not exist")
        sys.exit(1)
    print(f"Size:   {path.stat().st_size} bytes")
    print(f"xarray {xr.__version__}, cfgrib {cfgrib.__version__}, "
          f"ecCodes {eccodes.codes_get_api_version()}")

    inspect_messages(path)

    # open_datasets splits the file into one Dataset per compatible hypercube,
    # which is needed when messages mix level types.
    datasets = cfgrib.open_datasets(str(path), backend_kwargs=BACKEND_KWARGS)
    section(f"CFGRIB: {len(datasets)} dataset(s) / hypercube(s)")
    for i, ds in enumerate(datasets):
        print(f"  [{i}] vars={list(ds.data_vars)} dims={dict(ds.sizes)}")

    for i, ds in enumerate(datasets):
        inspect_dataset(i, ds)


if __name__ == "__main__":
    main()
