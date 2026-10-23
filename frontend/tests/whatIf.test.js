// What-if helpers. Run with `npm test`.
import assert from "node:assert/strict";
import { test } from "node:test";

import { describeError } from "../src/utils/errors.js";
import { buildRouteRequest } from "../src/utils/forecasts.js";
import {
  HAZARD_PENALTY,
  formatPenalty,
  parsePenalty,
  sensitivityLabel,
  whatIfResult,
} from "../src/utils/whatIf.js";

const O = { lat: 28.6139, lon: 77.209 };
const D = { lat: 19.076, lon: 72.8777 };

test("default K is the normal weather sensitivity, 4", () => {
  assert.deepEqual(HAZARD_PENALTY, { min: 0, max: 10, step: 0.5, normal: 4 });
  assert.equal(formatPenalty(HAZARD_PENALTY.normal), "4.0");
  assert.equal(sensitivityLabel(4), "Normal weather sensitivity (default)");
});

test("changing K: slider values parse, clamp and snap to 0.5", () => {
  assert.equal(parsePenalty("7.5"), 7.5);
  assert.equal(parsePenalty("8"), 8);
  assert.equal(parsePenalty(0), 0);
  assert.equal(parsePenalty("10"), 10);
  assert.equal(parsePenalty("12"), 10);
  assert.equal(parsePenalty("-3"), 0);
  assert.equal(parsePenalty("6.3"), 6.5);
  assert.equal(parsePenalty(""), 4);
  assert.equal(parsePenalty("abc"), 4);
  assert.equal(parsePenalty("NaN"), 4);
  assert.equal(formatPenalty(7.5), "7.5");
  assert.equal(formatPenalty(10), "10.0");
});

test("labels explain higher and lower K", () => {
  assert.match(sensitivityLabel(8), /Stronger avoidance/);
  assert.match(sensitivityLabel(2), /Less weather-sensitive/);
  assert.match(sensitivityLabel(0), /distance-only/);
});

test("what-if request carries hazard_penalty and the selected forecast", () => {
  assert.deepEqual(buildRouteRequest(O, D, "2026-10-23T12:00:00Z", 8), {
    origin: O,
    destination: D,
    forecast_valid_time: "2026-10-23T12:00:00Z",
    hazard_penalty: 8,
  });
  // K = 0 is a real value, not "missing".
  assert.equal(buildRouteRequest(O, D, null, 0).hazard_penalty, 0);
});

test("normal request leaves hazard_penalty out", () => {
  const body = buildRouteRequest(O, D, "2026-10-22T18:00:00Z");
  assert.ok(!("hazard_penalty" in body));
  assert.equal(body.forecast_valid_time, "2026-10-22T18:00:00Z");
});

test("whatIfResult only returns an enabled, complete section", () => {
  assert.equal(whatIfResult(null), null);
  assert.equal(whatIfResult({}), null);
  const disabled = { what_if: { enabled: false, hazard_penalty: 4, what_if_route: null, comparison: null } };
  assert.equal(whatIfResult(disabled), null);
  const enabled = { what_if: { enabled: true, hazard_penalty: 8, what_if_route: {}, comparison: {} } };
  assert.equal(whatIfResult(enabled), enabled.what_if);
});

test("hazard penalty errors get readable titles", () => {
  const err = describeError({
    code: "invalid_request",
    details: [{ field: "hazard_penalty", message: "Input should be less than or equal to 10" }],
  });
  assert.equal(err.message, "Hazard sensitivity: input should be less than or equal to 10");
  assert.equal(describeError({ code: "invalid_hazard_penalty", message: "x" }).title,
    "Invalid hazard sensitivity");
});
