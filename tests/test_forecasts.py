"""Tests for forecast discovery, selection and the multi-forecast grid cache.

Uses small synthetic hazard files in a temporary processed directory, so the
tests do not depend on the real GFS data. File names and modification times
are deliberately ordered against valid time to prove neither is used.
"""

import os
import threading
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from fastapi.testclient import TestClient

from app import app
from routing import service
from weather.forecast_catalog import (
    default_forecast,
    discover_forecasts,
    find_forecast,
    read_forecast_metadata,
)

client = TestClient(app)

RUN = pd.Timestamp("2030-01-01T00:00")
UTC = timezone.utc


def utc(*args) -> datetime:
    return datetime(*args, tzinfo=UTC)


def write_forecast(directory, name, step_h, hazard_value=0.2, run=RUN, step_coord=True):
    """Write a 5x5 hazard file for ``run + step_h`` hours; returns its path."""
    lats = np.arange(20.0, 21.25, 0.25)
    lons = np.arange(75.0, 76.25, 0.25)
    hazard = np.full((lats.size, lons.size), hazard_value)
    hazard[2, 1:4] = 0.9  # a hazardous band across the middle row
    coords = {
        "latitude": lats, "longitude": lons,
        "time": run, "valid_time": run + pd.Timedelta(hours=step_h),
    }
    if step_coord:
        coords["step"] = pd.Timedelta(hours=step_h)
    ds = xr.Dataset({"hazard_score": (("latitude", "longitude"), hazard)}, coords=coords,
                    attrs={"source_file": f"gfs.t00z.pgrb2.0p25.f{step_h:03d}"})
    path = directory / name
    ds.to_netcdf(path, engine="scipy")
    return path


@pytest.fixture
def forecast_dir(tmp_path, monkeypatch):
    """Three forecasts whose names and mtimes run opposite to valid time.

    valid 06:00 -> c.hazard.nc (oldest mtime)
    valid 12:00 -> b.hazard.nc
    valid 18:00 -> a.hazard.nc (newest mtime)
    """
    for i, (name, step) in enumerate([("c.hazard.nc", 6), ("b.hazard.nc", 12), ("a.hazard.nc", 18)]):
        path = write_forecast(tmp_path, name, step, hazard_value=0.1 * (i + 1))
        stamp = 1_700_000_000 + 1000 * i
        os.utime(path, (stamp, stamp))
    monkeypatch.setenv("WEATHER_PROCESSED_DIR", str(tmp_path))
    monkeypatch.delenv(service.HAZARD_FILE_ENV, raising=False)
    service.clear_hazard_cache()
    yield tmp_path
    service.clear_hazard_cache()


# --- discovery ----------------------------------------------------------

def test_discovery_sorted_by_valid_time_not_name_or_mtime(forecast_dir):
    entries = discover_forecasts()
    assert [e.valid_time for e in entries] == [utc(2030, 1, 1, 6), utc(2030, 1, 1, 12), utc(2030, 1, 1, 18)]
    assert [e.source_file for e in entries] == ["c.hazard.nc", "b.hazard.nc", "a.hazard.nc"]


def test_metadata_read_from_netcdf(forecast_dir):
    e = read_forecast_metadata(forecast_dir / "b.hazard.nc")
    assert e.valid_time == utc(2030, 1, 1, 12)
    assert e.valid_time.tzinfo is not None
    assert e.reference_time == utc(2030, 1, 1, 0)
    assert e.step_hours == 12.0
    assert e.grib_file == "gfs.t00z.pgrb2.0p25.f012"
    assert e.to_dict() == {
        "valid_time": "2030-01-01T12:00:00+00:00",
        "reference_time": "2030-01-01T00:00:00+00:00",
        "step_hours": 12.0,
        "source_file": "b.hazard.nc",
        "grib_file": "gfs.t00z.pgrb2.0p25.f012",
    }


def test_step_derived_when_step_coordinate_missing(tmp_path):
    path = write_forecast(tmp_path, "x.hazard.nc", 9, step_coord=False)
    assert read_forecast_metadata(path).step_hours == 9.0


def test_unreadable_and_non_hazard_files_skipped(forecast_dir):
    (forecast_dir / "garbage.hazard.nc").write_bytes(b"not a netcdf file")
    xr.Dataset({"hazard_score": (("latitude", "longitude"), np.zeros((2, 2)))},
               coords={"latitude": [1.0, 2.0], "longitude": [1.0, 2.0]}).to_netcdf(
        forecast_dir / "no_time.hazard.nc", engine="scipy")
    write_forecast(forecast_dir, "processed_only.nc", 30)  # not *.hazard.nc
    assert [e.source_file for e in discover_forecasts()] == ["c.hazard.nc", "b.hazard.nc", "a.hazard.nc"]


def test_duplicate_valid_time_keeps_newest_model_run(tmp_path):
    write_forecast(tmp_path, "old_run.hazard.nc", 12, run=RUN)
    write_forecast(tmp_path, "new_run.hazard.nc", 6, run=RUN + pd.Timedelta(hours=6))
    entries = discover_forecasts(tmp_path)
    assert len(entries) == 1
    assert entries[0].source_file == "new_run.hazard.nc"
    assert entries[0].step_hours == 6.0


def test_empty_directory(tmp_path):
    assert discover_forecasts(tmp_path) == []
    assert default_forecast([]) is None


@pytest.mark.parametrize("requested", [
    "2030-01-01T12:00:00Z",
    "2030-01-01T12:00:00+00:00",
    "2030-01-01T17:30:00+05:30",
    "2030-01-01T12:00:00",  # no offset: read as UTC
    utc(2030, 1, 1, 12),
    np.datetime64("2030-01-01T12:00"),
])
def test_find_forecast_accepts_equivalent_times(forecast_dir, requested):
    assert find_forecast(discover_forecasts(), requested).source_file == "b.hazard.nc"


def test_find_forecast_unknown_time(forecast_dir):
    assert find_forecast(discover_forecasts(), "2030-01-01T13:00:00Z") is None


# --- selection ----------------------------------------------------------

def test_default_forecast_is_earliest_valid_time(forecast_dir):
    # a.hazard.nc is newest on disk and first by name; neither matters.
    assert service.resolve_forecast().valid_time == utc(2030, 1, 1, 6)
    assert service.default_hazard_path().name == "c.hazard.nc"
    assert service.get_hazard_snapshot().source_file == "c.hazard.nc"


def test_default_is_stable_when_files_touched(forecast_dir):
    before = service.resolve_forecast()
    os.utime(forecast_dir / "b.hazard.nc", None)  # make the 12:00 file newest
    service.clear_hazard_cache()
    assert service.resolve_forecast() == before


def test_select_specific_forecast_loads_that_grid(forecast_dir):
    snapshot = service.get_hazard_snapshot("2030-01-01T18:00:00Z")
    assert snapshot.valid_time == utc(2030, 1, 1, 18)
    assert snapshot.source_file == "a.hazard.nc"
    assert snapshot.grid.hazard[0, 0] == pytest.approx(0.3)


def test_unknown_forecast_raises(forecast_dir):
    with pytest.raises(service.ForecastNotFoundError) as info:
        service.resolve_forecast("2031-01-01T00:00:00Z")
    assert info.value.requested == utc(2031, 1, 1)
    assert info.value.available == [utc(2030, 1, 1, 6), utc(2030, 1, 1, 12), utc(2030, 1, 1, 18)]
    assert "2031-01-01T00:00:00Z" in str(info.value)


def test_forecast_added_after_scan_is_found(forecast_dir):
    assert len(service.available_forecasts()) == 3
    write_forecast(forecast_dir, "d.hazard.nc", 24)
    # Not requested yet, so the cached list is still used ...
    assert len(service.available_forecasts()) == 3
    # ... but asking for the new valid time rescans once.
    assert service.resolve_forecast("2030-01-02T00:00:00Z").source_file == "d.hazard.nc"
    assert len(service.available_forecasts()) == 4


def test_no_forecasts_available(tmp_path, monkeypatch):
    monkeypatch.setenv("WEATHER_PROCESSED_DIR", str(tmp_path))
    monkeypatch.delenv(service.HAZARD_FILE_ENV, raising=False)
    service.clear_hazard_cache()
    try:
        assert service.list_forecasts() == {"default_valid_time": None, "forecasts": []}
        with pytest.raises(service.WeatherDataError, match="No processed hazard file"):
            service.resolve_forecast()
    finally:
        service.clear_hazard_cache()


def test_hazard_file_override_pins_single_forecast(forecast_dir, monkeypatch):
    monkeypatch.setenv(service.HAZARD_FILE_ENV, "b.hazard.nc")
    service.clear_hazard_cache()
    assert [e.source_file for e in service.available_forecasts()] == ["b.hazard.nc"]
    assert service.resolve_forecast().valid_time == utc(2030, 1, 1, 12)
    with pytest.raises(service.ForecastNotFoundError):
        service.resolve_forecast("2030-01-01T06:00:00Z")


# --- cache --------------------------------------------------------------

@pytest.fixture
def load_calls(monkeypatch):
    calls = []
    real = service.load_snapshot
    monkeypatch.setattr(service, "load_snapshot", lambda p: calls.append(p.name) or real(p))
    return calls


def test_cache_keeps_multiple_forecasts(forecast_dir, load_calls):
    times = ["2030-01-01T06:00:00Z", "2030-01-01T12:00:00Z", "2030-01-01T18:00:00Z"]
    first = [service.get_hazard_snapshot(t) for t in times]
    second = [service.get_hazard_snapshot(t) for t in reversed(times)]
    assert sorted(load_calls) == ["a.hazard.nc", "b.hazard.nc", "c.hazard.nc"]
    assert second[::-1] == first and all(a is b for a, b in zip(first, second[::-1]))
    assert service._cache.cached_valid_times() == [utc(2030, 1, 1, h) for h in (6, 12, 18)]


def test_default_and_explicit_share_cache_entry(forecast_dir, load_calls):
    assert service.get_hazard_snapshot() is service.get_hazard_snapshot("2030-01-01T06:00:00Z")
    assert load_calls == ["c.hazard.nc"]


def test_clear_forgets_all_forecasts(forecast_dir, load_calls):
    service.get_hazard_snapshot("2030-01-01T06:00:00Z")
    service.get_hazard_snapshot("2030-01-01T12:00:00Z")
    service.clear_hazard_cache()
    assert service._cache.cached_valid_times() == []
    service.get_hazard_snapshot("2030-01-01T06:00:00Z")
    assert load_calls == ["c.hazard.nc", "b.hazard.nc", "c.hazard.nc"]


def test_refresh_rescans_and_drops_removed_forecasts(forecast_dir):
    service.get_hazard_snapshot("2030-01-01T18:00:00Z")
    (forecast_dir / "a.hazard.nc").unlink()
    write_forecast(forecast_dir, "d.hazard.nc", 24)
    refreshed = service.refresh_hazard_grid()
    assert refreshed.source_file == "c.hazard.nc"
    assert [e.source_file for e in service.available_forecasts()] == ["c.hazard.nc", "b.hazard.nc", "d.hazard.nc"]
    assert utc(2030, 1, 1, 18) not in service._cache.cached_valid_times()


def test_concurrent_requests_load_each_forecast_once(forecast_dir, load_calls):
    times = ["2030-01-01T06:00:00Z", "2030-01-01T18:00:00Z"] * 8
    results, barrier = [], threading.Barrier(len(times))

    def worker(t):
        barrier.wait()
        results.append(service.get_hazard_snapshot(t))

    threads = [threading.Thread(target=worker, args=(t,)) for t in times]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == len(times)
    assert sorted(load_calls) == ["a.hazard.nc", "c.hazard.nc"]


# --- API ----------------------------------------------------------------

CORNER_A = {"lat": 20.0, "lon": 75.0}
CORNER_B = {"lat": 21.0, "lon": 76.0}


def test_api_forecasts_lists_chronologically(forecast_dir):
    response = client.get("/api/forecasts")
    assert response.status_code == 200, response.text
    body = response.json()
    assert pd.Timestamp(body["default_valid_time"]) == pd.Timestamp("2030-01-01T06:00Z")
    items = body["forecasts"]
    assert [pd.Timestamp(f["valid_time"]) for f in items] == [
        pd.Timestamp(f"2030-01-01T{h}:00Z") for h in ("06", "12", "18")]
    assert [f["is_default"] for f in items] == [True, False, False]
    assert [f["step_hours"] for f in items] == [6.0, 12.0, 18.0]
    assert set(items[0]) == {"valid_time", "reference_time", "step_hours", "source_file",
                             "grib_file", "is_default"}
    assert str(forecast_dir) not in response.text  # no local paths leaked


def test_api_forecasts_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("WEATHER_PROCESSED_DIR", str(tmp_path))
    monkeypatch.delenv(service.HAZARD_FILE_ENV, raising=False)
    service.clear_hazard_cache()
    try:
        response = client.get("/api/forecasts")
        assert response.status_code == 200
        assert response.json() == {"default_valid_time": None, "forecasts": []}
    finally:
        service.clear_hazard_cache()


def test_api_forecasts_missing_override_is_503(forecast_dir, monkeypatch):
    monkeypatch.setenv(service.HAZARD_FILE_ENV, "missing.hazard.nc")
    service.clear_hazard_cache()
    response = client.get("/api/forecasts")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "weather_data_unavailable"


def post(payload):
    return client.post("/api/route", json={"origin": CORNER_A, "destination": CORNER_B, **payload})


def test_api_route_uses_selected_forecast(forecast_dir):
    response = post({"forecast_valid_time": "2030-01-01T12:00:00Z"})
    assert response.status_code == 200, response.text
    f = response.json()["forecast"]
    assert pd.Timestamp(f["valid_time"]) == pd.Timestamp("2030-01-01T12:00Z")
    assert f["source_file"] == "b.hazard.nc"
    assert f["step_hours"] == 12.0
    assert f["grib_file"] == "gfs.t00z.pgrb2.0p25.f012"


def test_api_route_without_forecast_uses_default(forecast_dir):
    response = post({})
    assert response.status_code == 200, response.text
    assert response.json()["forecast"]["source_file"] == "c.hazard.nc"


def test_api_route_null_forecast_uses_default(forecast_dir):
    response = post({"forecast_valid_time": None})
    assert response.status_code == 200, response.text
    assert response.json()["forecast"]["source_file"] == "c.hazard.nc"


def test_api_route_unknown_forecast(forecast_dir):
    response = post({"forecast_valid_time": "2030-01-01T13:00:00Z"})
    assert response.status_code == 422
    err = response.json()["error"]
    assert err["code"] == "forecast_not_found"
    assert "2030-01-01T13:00:00Z" in err["message"]
    assert [pd.Timestamp(d["valid_time"]) for d in err["details"]] == [
        pd.Timestamp(f"2030-01-01T{h}:00Z") for h in ("06", "12", "18")]
    assert "Traceback" not in response.text


@pytest.mark.parametrize("value", ["tomorrow", "2030-13-01T00:00:00Z", [], {"t": 1}])
def test_api_route_malformed_forecast_time(forecast_dir, value):
    response = post({"forecast_valid_time": value})
    assert response.status_code == 422
    err = response.json()["error"]
    assert err["code"] == "invalid_request"
    assert err["details"][0]["field"] == "forecast_valid_time"


def test_api_route_offset_time_matches_utc_forecast(forecast_dir):
    response = post({"forecast_valid_time": "2030-01-01T23:30:00+05:30"})
    assert response.status_code == 200, response.text
    assert response.json()["forecast"]["source_file"] == "a.hazard.nc"

