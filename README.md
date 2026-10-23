# Turbulence & Weather-Aware Route Optimization System

Given an origin and a destination, the system picks one NOAA GFS forecast
(by its valid time) and computes two routes over that forecast's experimental
hazard grid:

1. **Distance-only** (baseline) - A* on great-circle distance
2. **Weather-aware** (recommended) - A* on distance weighted by
   `1 + 4 * hazard_score`

and compares them on distance, weather-weighted cost, hazard exposure, cells
avoided and a distance-based fuel proxy. Each forecast is routed on
independently; there is no time-aware routing across forecasts. An optional
What-if mode re-plans the weather-aware route with a different hazard
sensitivity K. Results are shown on a Leaflet map.

The hazard score is an academic routing cost, not an operational turbulence
forecast.

## Layout

| Path | Responsibility |
|------|----------------|
| `backend/weather/` | GRIB reading, preprocessing, hazard model, forecast discovery (`forecast_catalog.py`) |
| `backend/routing/` | Geographic grid, edge costs, A*, comparison metrics, route service |
| `backend/api/` | FastAPI router (`/api/...`); `/health` lives in `app.py` |
| `backend/data/` | `raw/` GRIB2 files, `processed/` generated NetCDF (git-ignored) |
| `frontend/` | React + Leaflet UI (Vite) |
| `notebooks/` | Data exploration |
| `tests/` | Backend pytest suite |

## Setup

```bash
cp .env.example .env            # optional; defaults work for local development

# Backend
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -r backend/requirements.txt
cd backend && uvicorn app:app --reload    # http://127.0.0.1:8000/health

# Frontend
cd frontend && npm install && npm run dev  # http://localhost:5173
```

## Data setup

Weather data is not in the repository: `backend/data/raw/*` and
`backend/data/processed/*` are git-ignored. A fresh clone has no forecasts,
so `GET /api/forecasts` returns an empty list and `POST /api/route` returns
`503 weather_data_unavailable` until the data is prepared locally.

1. **Raw GRIB2 files (required).** Place the NOAA GFS 0.25° GRIB2 files in
   `backend/data/raw/`. The current data set is the 2026-10-22 12Z run,
   forecast hours f006, f012, f018 and f024:

   ```
   gfs.t12z.pgrb2.0p25.f006   gfs.t12z.pgrb2.0p25.f012
   gfs.t12z.pgrb2.0p25.f018   gfs.t12z.pgrb2.0p25.f024
   ```

   Each must contain `u`, `v` and `r` on isobaric levels including 300 and
   400 hPa. The current subset covers 5–35° N, 65–95° E; the grid is taken
   from the files themselves.

2. **Generate the processed and hazard NetCDF files.** From `backend/`, with
   the virtual environment active, run this in Python:

   ```python
   import numpy as np

   from weather.grib_reader import processed_dir
   from weather.hazard_model import LOWER_QUANTILE, UPPER_QUANTILE, compute_hazard, save_hazard
   from weather.preprocessing import preprocess_file

   names = [f"gfs.t12z.pgrb2.0p25.f{h:03d}" for h in (6, 12, 18, 24)]
   processed = {n: preprocess_file(n, save=True) for n in names}  # -> data/processed/<name>.nc


   def pooled(var):
       values = np.concatenate([ds[var].values.ravel() for ds in processed.values()])
       return tuple(float(q) for q in np.nanquantile(values, [LOWER_QUANTILE, UPPER_QUANTILE]))


   bounds = {"wind": pooled("wind_speed_300"), "shear": pooled("wind_shear")}
   for n, ds in processed.items():
       save_hazard(compute_hazard(ds, bounds), processed_dir() / f"{n}.hazard.nc")
   ```

   This writes `<name>.nc` (wind, shear, humidity fields) and
   `<name>.hazard.nc` (hazard score) for each forecast. The wind and shear
   normalisation bounds are pooled over all four forecasts, so their hazard
   scores share one scale; this reproduces the current processed files.

The API serves every `*.hazard.nc` in `backend/data/processed/` (see
`WEATHER_PROCESSED_DIR` and `WEATHER_HAZARD_FILE` in `.env.example`).

## API

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Liveness check |
| `GET /api/forecasts` | Available forecasts, oldest valid time first, default marked |
| `POST /api/route` | Distance-only and weather-aware routes, their comparison, and an optional What-if route |

### Forecast selection

Each `*.hazard.nc` file is one forecast, identified by the `valid_time` stored
in the NetCDF (never by file name or modification time). `GET /api/forecasts`
returns each forecast's `valid_time`, `reference_time`, `step_hours`,
`source_file`, `grib_file` and `is_default`, plus `default_valid_time`.

`POST /api/route` takes an optional `forecast_valid_time` (ISO-8601, UTC):

```json
{"origin": {"lat": 28.6139, "lon": 77.2090},
 "destination": {"lat": 19.0760, "lon": 72.8777},
 "forecast_valid_time": "2026-10-23T06:00:00Z"}
```

Omitting it uses the default forecast: the earliest available valid time
(currently f006, 2026-10-22T18:00Z). The response's `forecast` section names
the forecast actually used. Setting `WEATHER_HAZARD_FILE` pins the server to
one file. Loaded grids are cached in memory per forecast.

### What-if mode

What-if changes the hazard sensitivity K, not the forecast time (the forecast
is chosen separately with `forecast_valid_time`). An optional
`hazard_penalty` K plans one more route with edge weight
`1 + K * hazard_score` and compares it with the normal weather-aware route:

- K ranges from 0 to 10 and must be finite.
- K = 4 is the normal weather-aware routing cost.
- Lower K makes the optimizer less sensitive to hazard (K = 0 gives the
  distance-only route).
- Higher K makes it more sensitive, avoiding hazardous cells more strongly.

```json
{"origin": {"lat": 28.6139, "lon": 77.2090},
 "destination": {"lat": 19.0760, "lon": 72.8777},
 "forecast_valid_time": "2026-10-23T12:00:00Z",
 "hazard_penalty": 8}
```

The response's `what_if` section holds `enabled`, `hazard_penalty`, the
normal route's metrics (`baseline_route`), the scenario route
(`what_if_route`) and their `comparison`; weather costs in it use the normal
K = 4 scale so the two routes compare directly. Without `hazard_penalty`,
`what_if.enabled` is `false` and the rest of the response is unchanged.

### Errors

Errors share one shape: `{"error": {"code": "...", "message": "...", "details": [...]}}`.
Codes include `invalid_request` (validation, including out-of-range
`hazard_penalty` or a malformed `forecast_valid_time`), `forecast_not_found`
(unknown valid time; `details` lists the available ones), `outside_grid`,
`same_grid_point`, `route_not_found` (all 422) and
`weather_data_unavailable` (503).

## Tests

```bash
pytest                          # from the repository root
cd frontend && npm test
cd frontend && npm run build
```

- **Backend (pytest, 281 tests):** weather pipeline, hazard model, grid,
  costs, A*, metrics, forecast discovery and caching, What-if, and the API.
  Tests that need the real processed GFS files are skipped when the data is
  absent.
- **Frontend (`node --test`, 31 tests):** helper tests for forecast selection,
  request building, formatting, What-if values and error messages, plus
  component tests that render `ControlPanel`, `ForecastSelector`,
  `WhatIfControl`, `MetricsPanel` and `MapLegend` through Vite's SSR loader
  and check the requests sent by the API client.

## Status

- [x] Weather / GRIB pipeline
- [x] Hazard model
- [x] Routing (grid, edge costs, A*)
- [x] Metrics and route comparison
- [x] FastAPI API
- [x] Multi-forecast selection
- [x] What-if mode
- [x] React + Leaflet frontend
