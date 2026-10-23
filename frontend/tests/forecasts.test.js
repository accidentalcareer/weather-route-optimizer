// Run with `npm test` (Node's built-in test runner; no extra dependencies).
import assert from "node:assert/strict";
import { test } from "node:test";

import { describeError } from "../src/utils/errors.js";
import { formatUtcDay, formatUtcTime } from "../src/utils/format.js";
import {
  buildRouteRequest,
  findForecast,
  forecastCode,
  formatStep,
  initialForecastTime,
  sameInstant,
  sortForecasts,
} from "../src/utils/forecasts.js";

// Shape of GET /api/forecasts for the four GFS 12Z files, deliberately unsorted.
const LIST = {
  default_valid_time: "2026-10-22T18:00:00+00:00",
  forecasts: [
    { valid_time: "2026-10-23T06:00:00+00:00", step_hours: 18, is_default: false },
    { valid_time: "2026-10-22T18:00:00+00:00", step_hours: 6, is_default: true },
    { valid_time: "2026-10-23T12:00:00+00:00", step_hours: 24, is_default: false },
    { valid_time: "2026-10-23T00:00:00+00:00", step_hours: 12, is_default: false },
  ].map((f) => ({ ...f, reference_time: "2026-10-22T12:00:00+00:00", source_file: "x.hazard.nc" })),
};

test("forecasts are sorted chronologically", () => {
  assert.deepEqual(
    sortForecasts(LIST.forecasts).map((f) => f.step_hours),
    [6, 12, 18, 24]
  );
  assert.deepEqual(sortForecasts(undefined), []);
});

test("initial selection is the backend default", () => {
  assert.equal(initialForecastTime(LIST), "2026-10-22T18:00:00+00:00");
});

test("initial selection falls back to the earliest forecast", () => {
  assert.equal(
    initialForecastTime({ ...LIST, default_valid_time: null }),
    "2026-10-22T18:00:00+00:00"
  );
  assert.equal(initialForecastTime({ default_valid_time: null, forecasts: [] }), null);
});

test("timestamps compare as instants, not strings", () => {
  assert.ok(sameInstant("2026-10-22T18:00:00Z", "2026-10-22T18:00:00+00:00"));
  assert.ok(sameInstant("2026-10-22T23:30:00+05:30", "2026-10-22T18:00:00Z"));
  assert.ok(!sameInstant("2026-10-22T18:00:00Z", "2026-10-23T00:00:00Z"));
  assert.ok(!sameInstant(null, "2026-10-22T18:00:00Z"));
  assert.equal(findForecast(LIST.forecasts, "2026-10-23T12:00:00Z").step_hours, 24);
  assert.equal(findForecast(LIST.forecasts, "2026-10-23T13:00:00Z"), null);
});

test("forecast cards show the GFS hour code and lead time", () => {
  assert.deepEqual(
    sortForecasts(LIST.forecasts).map((f) => forecastCode(f.step_hours)),
    ["f006", "f012", "f018", "f024"]
  );
  assert.equal(forecastCode(null), "");
  assert.equal(formatStep(6), "+6 h");
  assert.equal(formatUtcDay("2026-10-22T18:00:00Z"), "22 Oct");
  assert.equal(formatUtcTime("2026-10-23T00:00:00+00:00"), "00:00");
  assert.equal(formatStep(null), "");
  assert.equal(formatStep(1.5), "+1.5 h");
});

test("route request includes the selected forecast", () => {
  const o = { lat: 28.6139, lon: 77.209 };
  const d = { lat: 19.076, lon: 72.8777 };
  assert.deepEqual(buildRouteRequest(o, d, "2026-10-23T06:00:00+00:00"), {
    origin: o,
    destination: d,
    forecast_valid_time: "2026-10-23T06:00:00+00:00",
  });
});

test("route request omits the forecast when none is selected", () => {
  const body = buildRouteRequest({ lat: 1, lon: 2 }, { lat: 3, lon: 4 }, null);
  assert.ok(!("forecast_valid_time" in body));
});

test("forecast errors get a readable title and keep their code", () => {
  const err = describeError({ code: "forecast_not_found", message: "No forecast is available…" });
  assert.equal(err.title, "Forecast not available");
  assert.equal(err.code, "forecast_not_found");
  const invalid = describeError({
    code: "invalid_request",
    details: [{ field: "forecast_valid_time", message: "Input should be a valid datetime" }],
  });
  assert.equal(invalid.message, "Forecast: input should be a valid datetime");
});
