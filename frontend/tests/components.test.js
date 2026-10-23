// Component rendering tests. JSX components are loaded through Vite's SSR
// module loader (already a dev dependency) and rendered with react-dom/server,
// so no test framework or DOM library is needed. Run with `npm test`.
import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import { fileURLToPath } from "node:url";

import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { createServer } from "vite";

const root = fileURLToPath(new URL("..", import.meta.url));
let vite;
let mod;

before(async () => {
  vite = await createServer({
    root,
    configFile: false,
    server: { middlewareMode: true, hmr: false },
    appType: "custom",
    logLevel: "error",
    esbuild: { jsx: "automatic" },
  });
  const load = (p) => vite.ssrLoadModule(p);
  mod = {
    ControlPanel: (await load("/src/components/ControlPanel.jsx")).default,
    WhatIfControl: (await load("/src/components/WhatIfControl.jsx")).default,
    MetricsPanel: (await load("/src/components/MetricsPanel.jsx")).default,
    MapLegend: (await load("/src/components/MapLegend.jsx")).default,
    ForecastSelector: (await load("/src/components/ForecastSelector.jsx")).default,
    api: await load("/src/services/api.js"),
  };
});

after(async () => {
  await vite?.close();
});

const html = (component, props) =>
  renderToStaticMarkup(React.createElement(component, props))
    .replace(/&amp;/g, "&")
    .replace(/<!-- -->/g, "");

// --- fixture: a response shaped like POST /api/route -------------------

const counts = (low, mod_, high, severe) => ({ LOW: low, MODERATE: mod_, HIGH: high, SEVERE: severe });
const route = (lengthKm, avg, max, cats, coords) => ({
  coordinates: coords,
  node_count: coords.length,
  length_km: lengthKm,
  total_cost: lengthKm * 2,
  nodes_expanded: 100,
  average_hazard: avg,
  maximum_hazard: max,
  category_counts: cats,
});
const A = { lat: 28.5, lon: 77.25 };
const B = { lat: 19.0, lon: 73.0 };

function routeData({ whatIf }) {
  const base = {
    forecast: {
      valid_time: "2026-10-23T12:00:00Z", reference_time: "2026-10-22T12:00:00Z",
      step_hours: 24, source_file: "gfs.t12z.pgrb2.0p25.f024.hazard.nc",
      grib_file: "gfs.t12z.pgrb2.0p25.f024",
      grid_bounds: { lat_min: 5, lat_max: 35, lon_min: 65, lon_max: 95 },
    },
    origin: { requested: A, snapped: A, snap_distance_km: 1 },
    destination: { requested: B, snapped: B, snap_distance_km: 1 },
    distance_route: route(1218.2, 0.5, 0.8, counts(1, 2, 3, 4), [A, B]),
    weather_route: route(1221.9, 0.43, 0.7, counts(2, 2, 3, 1), [A, { lat: 24, lon: 75 }, B]),
    comparison: {
      routes_identical: false, extra_distance_km: 3.7, extra_distance_percent: 0.3,
      distance_route_weather_cost: 3300, weather_route_weather_cost: 3150,
      weather_cost_difference: 150, weather_cost_reduction_percent: 4.66,
      average_hazard_distance_route: 0.5, average_hazard_weather_route: 0.43,
      maximum_hazard_distance_route: 0.8, maximum_hazard_weather_route: 0.7,
      severe_cells_avoided: 3, high_hazard_cells_avoided: 3, estimated_fuel_delta_percent: 0.3,
      fuel_proxy_note: "Distance-based fuel proxy only.", explanation: "NORMAL-EXPLANATION",
    },
    what_if: { enabled: false, hazard_penalty: 4, baseline_route: null, what_if_route: null, comparison: null },
  };
  if (whatIf) {
    base.what_if = {
      enabled: true,
      hazard_penalty: 8,
      baseline_route: {
        node_count: 3, length_km: 1221.9, weather_cost: 3150.5, average_hazard: 0.43,
        maximum_hazard: 0.7, category_counts: counts(2, 2, 3, 1),
      },
      what_if_route: {
        ...route(1262.3, 0.39, 0.66, counts(3, 2, 1, 0), [A, { lat: 23, lon: 74 }, B]),
        weather_cost: 3001.25,
      },
      comparison: {
        routes_identical: false, extra_distance_km: 40.4, extra_distance_percent: 3.31,
        weather_cost_difference: 149.25, weather_cost_reduction_percent: 4.74,
        average_hazard_reduction: 0.04, average_hazard_reduction_percent: 9.3,
        maximum_hazard_reduction: 0.04, severe_cells_avoided: 1, high_hazard_cells_avoided: 3,
        explanation: "WHAT-IF-EXPLANATION",
      },
    };
  }
  return base;
}

// --- what-if control ----------------------------------------------------

const noop = () => {};

test("what-if control renders the slider at the default K = 4", () => {
  const out = html(mod.WhatIfControl, { value: 4, onChange: noop, onRun: noop, disabled: false, running: false });
  assert.match(out, /Hazard sensitivity/);
  assert.match(out, /K = 4\.0/);
  assert.match(out, /type="range"[^>]*min="0"[^>]*max="10"[^>]*step="0.5"[^>]*value="4"/);
  assert.match(out, /Normal weather sensitivity \(default\)/);
  assert.match(out, /Higher K avoids hazardous cells more strongly; lower K is less\s+weather-sensitive/);
  assert.match(out, />Run What-if</);
  assert.doesNotMatch(out, /Reset to K/); // already normal
});

test("what-if control reflects a changed K", () => {
  const out = html(mod.WhatIfControl, { value: 7.5, onChange: noop, onRun: noop, disabled: false, running: false });
  assert.match(out, /K = 7\.5/);
  assert.match(out, /value="7.5"/);
  assert.match(out, /Stronger avoidance of hazardous cells/);
  assert.match(out, /Reset to K = 4\.0/);
});

test("what-if control shows loading state and disables while running", () => {
  const out = html(mod.WhatIfControl, { value: 8, onChange: noop, onRun: noop, disabled: true, running: true });
  assert.match(out, /<fieldset[^>]*disabled=""/);
  assert.match(out, /Running what-if…/);
});

const ITEMS = [
  { valid_time: "2026-10-22T18:00:00Z", reference_time: "2026-10-22T12:00:00Z", step_hours: 6, is_default: true, source_file: "a" },
  { valid_time: "2026-10-23T00:00:00Z", reference_time: "2026-10-22T12:00:00Z", step_hours: 12, is_default: false, source_file: "b" },
  { valid_time: "2026-10-23T06:00:00Z", reference_time: "2026-10-22T12:00:00Z", step_hours: 18, is_default: false, source_file: "c" },
  { valid_time: "2026-10-23T12:00:00Z", reference_time: "2026-10-22T12:00:00Z", step_hours: 24, is_default: false, source_file: "d" },
];

const panelProps = (extra = {}) => ({
  origin: { lat: "28.6139", lon: "77.2090" }, destination: { lat: "19.0760", lon: "72.8777" },
  onOriginChange: noop, onDestinationChange: noop,
  forecasts: { status: "ready", items: ITEMS, error: null },
  forecastValidTime: "2026-10-23T12:00:00Z", onForecastChange: noop, onForecastRetry: noop,
  hazardPenalty: 6, onHazardPenaltyChange: noop, hasResult: true, notice: null, loading: false,
  error: null, onRequestStart: noop, onRequestSuccess: noop, onRequestError: noop, ...extra,
});

test("control panel keeps route inputs, forecast selector and adds what-if", () => {
  const out = html(mod.ControlPanel, panelProps());
  assert.match(out, /value="28.6139"/);
  assert.match(out, /value="72.8777"/);
  assert.match(out, /<input type="radio" name="forecast" checked="" value="2026-10-23T12:00:00Z"\/>/);
  assert.equal((out.match(/checked=""/g) || []).length, 1);
  assert.match(out, />Optimize Route</);
  assert.match(out, /K = 6\.0/);
  assert.match(out, />Run What-if</);
  // Normal submit stays the form's submit button; what-if is a separate action.
  assert.match(out, /<button type="submit"[^>]*>Optimize Route/);
  assert.match(out, /<button type="button" class="button-secondary"/);
});

test("control panel follows the workflow order 1 Forecast, 2 points, 3 Optimize, 5 What-if", () => {
  const out = html(mod.ControlPanel, panelProps());
  const order = ["Forecast", "Origin & destination", "Optimize", "What-if scenario"].map((t) =>
    out.indexOf(`<span>${t}</span>`)
  );
  assert.ok(order.every((i) => i > 0), JSON.stringify(order));
  assert.deepEqual([...order].sort((a, b) => a - b), order);
  const originField = out.indexOf('id="origin-lat"');
  assert.ok(originField > order[1] && originField < order[2]);
});

test("forecast cards show code, valid time, lead time and the selected state", () => {
  const out = html(mod.ForecastSelector, {
    forecasts: { status: "ready", items: ITEMS, error: null },
    value: "2026-10-23T00:00:00Z", onChange: noop, onRetry: noop, disabled: false,
  });
  const cards = [...out.matchAll(/<label class="forecast-option( is-selected)?"[^>]*>(.*?)<\/label>/g)];
  assert.equal(cards.length, 4);
  const text = cards.map((c) => c[2].replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim());
  assert.deepEqual(text, [
    "f006 default 18:00 UTC 22 Oct · +6 h",
    "f012 00:00 UTC 23 Oct · +12 h",
    "f018 06:00 UTC 23 Oct · +18 h",
    "f024 12:00 UTC 23 Oct · +24 h",
  ]);
  assert.deepEqual(cards.map((c) => Boolean(c[1])), [false, true, false, false]);
  assert.match(out, /Selected: <strong>23 Oct 2026, 00:00 UTC<\/strong>/);
});

test("forecast loading, error and empty states", () => {
  const props = (forecasts) => ({ forecasts, value: null, onChange: noop, onRetry: noop });
  const loading = html(mod.ForecastSelector, props({ status: "loading", items: [], error: null }));
  assert.match(loading, /role="status"/);
  assert.match(loading, /Loading available forecasts/);
  const failed = html(mod.ForecastSelector, props({ status: "error", items: [], error: { message: "Cannot reach the backend" } }));
  assert.match(failed, /role="alert"/);
  assert.match(failed, /Cannot reach the backend/);
  assert.match(failed, />Retry</);
  const empty = html(mod.ForecastSelector, props({ status: "ready", items: [], error: null }));
  assert.match(empty, /No processed forecasts are available/);
});

test("what-if is locked until a route exists", () => {
  const before = html(mod.ControlPanel, panelProps({ hasResult: false }));
  assert.match(before, /Optimize a route first/);
  assert.match(before, /<fieldset class="what-if" disabled=""/);
  const after = html(mod.ControlPanel, panelProps());
  assert.doesNotMatch(after, /Optimize a route first/);
  assert.doesNotMatch(after, /<fieldset class="what-if" disabled=""/);
});

test("route error is shown next to the Optimize button", () => {
  const out = html(mod.ControlPanel, panelProps({
    error: { title: "Outside the forecast area", message: "Origin is outside" },
  }));
  assert.match(out, /role="alert"><strong>Outside the forecast area<\/strong><span>Origin is outside<\/span>/);
  assert.ok(out.indexOf("Outside the forecast area") < out.indexOf("What-if scenario"));
});

// --- requests -----------------------------------------------------------

async function captureRequest(call) {
  const realFetch = globalThis.fetch;
  let captured;
  globalThis.fetch = async (url, opts) => {
    captured = { url, body: JSON.parse(opts.body) };
    return new Response(JSON.stringify(routeData({ whatIf: true })), { status: 200 });
  };
  try {
    await call();
  } finally {
    globalThis.fetch = realFetch;
  }
  return captured;
}

test("Run What-if request contains hazard_penalty and keeps the selected forecast", async () => {
  const sent = await captureRequest(() =>
    mod.api.planRoute(
      { lat: 28.6139, lon: 77.209 }, { lat: 19.076, lon: 72.8777 }, "2026-10-23T12:00:00Z", 8
    )
  );
  assert.match(sent.url, /\/api\/route$/);
  assert.deepEqual(sent.body, {
    origin: { lat: 28.6139, lon: 77.209 },
    destination: { lat: 19.076, lon: 72.8777 },
    forecast_valid_time: "2026-10-23T12:00:00Z",
    hazard_penalty: 8,
  });
});

test("Optimize Route request has no hazard_penalty", async () => {
  const sent = await captureRequest(() =>
    mod.api.planRoute({ lat: 1, lon: 2 }, { lat: 3, lon: 4 }, "2026-10-22T18:00:00Z")
  );
  assert.ok(!("hazard_penalty" in sent.body));
  assert.equal(sent.body.forecast_valid_time, "2026-10-22T18:00:00Z");
});

// --- results ------------------------------------------------------------

test("what-if results render from the response", () => {
  const out = html(mod.MetricsPanel, { routeData: routeData({ whatIf: true }), loading: false });
  assert.match(out, /What-if scenario · K = 8\.0/);
  assert.match(out, /WHAT-IF-EXPLANATION/);
  const table = out.match(/data-testid="what-if-table">(.*?)<\/table>/)[1];
  const row = (label) => [...table.matchAll(new RegExp(`<th scope="row">${label}</th><td>([^<]*)</td><td>([^<]*)</td>`, "g"))][0].slice(1);
  assert.deepEqual(row("Length \\(km\\)"), ["1,221.9", "1,262.3"]);
  assert.deepEqual(row("Weather cost\\*"), ["3,150.5", "3,001.3"]);
  assert.deepEqual(row("Average hazard"), ["0.430", "0.390"]);
  assert.deepEqual(row("Maximum hazard"), ["0.700", "0.660"]);
  assert.deepEqual(row("Severe cells"), ["1", "0"]);
  assert.deepEqual(row("High-hazard cells"), ["3", "1"]);
  assert.match(out, /Extra vs weather-aware.*?\+40\.4.*?km \(\+3\.3%\)/);
  assert.match(out, /Avg hazard reduction.*?\+9\.3/);
});

test("headline results show extra distance, cost reduction, severe cells and average hazard", () => {
  const out = html(mod.MetricsPanel, { routeData: routeData({ whatIf: false }), loading: false });
  const tiles = out.match(/data-testid="tradeoffs">(.*?)<\/dl>/)[1]
    .replace(/<[^>]+>/g, " ").replace(/\s+/g, " ");
  assert.match(tiles, /Extra distance \+3\.7 km \+0\.30% vs distance-only/);
  assert.match(tiles, /Weather-cost reduction 4\.7 %/);
  assert.match(tiles, /Severe cells avoided 3 4 → 1 on route/);
  assert.match(tiles, /Average hazard 0\.430 vs 0\.500 distance-only/);
  assert.match(out, /<strong>f024<\/strong> 23 Oct 2026, 12:00 UTC/);
  // The recommendation comes before, and is distinct from, the what-if scenario.
  const withWhatIf = html(mod.MetricsPanel, { routeData: routeData({ whatIf: true }), loading: false });
  assert.ok(withWhatIf.indexOf("NORMAL-EXPLANATION") < withWhatIf.indexOf("WHAT-IF-EXPLANATION"));
  assert.match(withWhatIf, /not the recommendation/);
});

test("results empty and loading states", () => {
  const empty = html(mod.MetricsPanel, { routeData: null, loading: false });
  assert.match(empty, /No route yet/);
  assert.match(empty, /Select <strong>Optimize Route<\/strong>/);
  const loading = html(mod.MetricsPanel, { routeData: null, loading: true });
  assert.match(loading, /role="status"/);
  assert.match(loading, /Calculating weather-aware routes/);
  const updating = html(mod.MetricsPanel, { routeData: routeData({ whatIf: false }), loading: true });
  assert.match(updating, /class="panel metrics-panel is-stale"/);
  assert.match(updating, /Updating…/);
});

test("existing route display is intact with and without what-if", () => {
  for (const whatIf of [false, true]) {
    const out = html(mod.MetricsPanel, { routeData: routeData({ whatIf }), loading: false });
    assert.match(out, /NORMAL-EXPLANATION/);
    assert.match(out, /data-testid="tradeoffs"/);
    assert.match(out, /data-testid="route-table"/);
    assert.match(out, /Forecast used/);
    assert.match(out, /23 Oct 2026, 12:00 UTC/);
    assert.match(out, /gfs\.t12z\.pgrb2\.0p25\.f024\.hazard\.nc/);
    assert.equal(/data-testid="what-if-result"/.test(out), whatIf);
  }
});

test("map legend lists all three routes only when what-if ran", () => {
  const normal = html(mod.MapLegend, { whatIf: null });
  assert.match(normal, /Distance-only <em>— baseline<\/em>/);
  assert.match(normal, /Weather-aware <em>— recommended<\/em>/);
  assert.doesNotMatch(normal, /What-if/);
  const scenario = html(mod.MapLegend, { whatIf: routeData({ whatIf: true }).what_if });
  assert.match(scenario, /Distance-only/);
  assert.match(scenario, /Weather-aware/);
  assert.match(scenario, /What-if <em>— scenario, K = 8\.0<\/em>/);
  assert.match(scenario, /map-legend__line--what-if/);
});
