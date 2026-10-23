import {
  formatInteger,
  formatNumber,
  formatSigned,
  formatUtc,
} from "../utils/format.js";
import { forecastCode, formatStep } from "../utils/forecasts.js";
import { formatPenalty, whatIfResult } from "../utils/whatIf.js";
import StepTitle from "./StepTitle.jsx";

// Every value shown here is read directly from the POST /api/route response.

function Stat({ label, value, unit, emphasis }) {
  return (
    <div className={`stat${emphasis ? " stat--emphasis" : ""}`}>
      <dt>{label}</dt>
      <dd>
        <span className="stat__value">{value}</span>
        {unit && <span className="stat__unit">{unit}</span>}
      </dd>
    </div>
  );
}

/** Headline figure: big value, unit, and one line of context. */
function KeyStat({ label, value, unit, note, tone }) {
  return (
    <div className={`key-stat${tone ? ` key-stat--${tone}` : ""}`}>
      <dt>{label}</dt>
      <dd>
        <span className="key-stat__value">{value}</span>
        {unit && <span className="key-stat__unit">{unit}</span>}
      </dd>
      {note && <dd className="key-stat__note">{note}</dd>}
    </div>
  );
}

/** Normal weather-aware route vs the what-if route, from the response's what_if section. */
function WhatIfSection({ whatIf }) {
  const { baseline_route: b, what_if_route: x, comparison: wc } = whatIf;
  const k = formatPenalty(whatIf.hazard_penalty);
  const rows = [
    ["Length (km)", formatNumber(b.length_km, 1), formatNumber(x.length_km, 1)],
    ["Weather cost*", formatNumber(b.weather_cost, 1), formatNumber(x.weather_cost, 1)],
    ["Average hazard", formatNumber(b.average_hazard, 3), formatNumber(x.average_hazard, 3)],
    ["Maximum hazard", formatNumber(b.maximum_hazard, 3), formatNumber(x.maximum_hazard, 3)],
    ["Severe cells", formatInteger(b.category_counts.SEVERE), formatInteger(x.category_counts.SEVERE)],
    ["High-hazard cells", formatInteger(b.category_counts.HIGH), formatInteger(x.category_counts.HIGH)],
  ];
  return (
    <section className="what-if-result" data-testid="what-if-result">
      <h3 className="what-if-result__title">
        <span className="swatch swatch--what-if" />
        What-if scenario · K = {k}
        <span className="what-if-result__badge">not the recommendation</span>
      </h3>
      <p className="what-if-result__text" data-testid="what-if-explanation">
        {wc.explanation}
      </p>
      <dl className="stats">
        <Stat label="What-if route length" value={formatNumber(x.length_km, 1)} unit="km" />
        <Stat
          label="Extra vs weather-aware"
          value={formatSigned(wc.extra_distance_km, 1)}
          unit={`km (${formatSigned(wc.extra_distance_percent, 1)}%)`}
        />
        <Stat
          label="Avg hazard reduction"
          value={formatSigned(wc.average_hazard_reduction_percent, 1)}
          unit="%"
        />
        <Stat label="High+severe avoided" value={formatSigned(wc.high_hazard_cells_avoided, 0)} />
      </dl>
      <table className="compare-table" data-testid="what-if-table">
        <thead>
          <tr>
            <th scope="col" />
            <th scope="col">
              <span className="swatch swatch--weather" />
              Weather-aware
            </th>
            <th scope="col">
              <span className="swatch swatch--what-if" />
              What-if
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([label, normal, scenario]) => (
            <tr key={label}>
              <th scope="row">{label}</th>
              <td>{normal}</td>
              <td>{scenario}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="fineprint">
        * Weather cost uses the normal K = 4 scale for both routes, so they compare directly.
        {wc.routes_identical && " The what-if route is the same as the weather-aware route."}
      </p>
    </section>
  );
}

function EmptyState({ loading }) {
  return (
    <aside className="panel metrics-panel" aria-label="Route comparison" aria-busy={loading}>
      <StepTitle n={4} id="results-title">
        Results
      </StepTitle>
      {loading ? (
        <p className="state-note" role="status">
          <span className="spinner" aria-hidden="true" /> Calculating weather-aware routes…
        </p>
      ) : (
        <div className="empty-steps">
          <p>No route yet. To compare routes:</p>
          <ol>
            <li>Pick a forecast.</li>
            <li>Check the origin and destination.</li>
            <li>
              Select <strong>Optimize Route</strong>; results appear here.
            </li>
            <li value={5}>Optionally run a what-if scenario.</li>
          </ol>
        </div>
      )}
    </aside>
  );
}

export default function MetricsPanel({ routeData, loading }) {
  if (!routeData) return <EmptyState loading={loading} />;

  const { comparison: c, distance_route: d, weather_route: w, forecast: f } = routeData;
  const whatIf = whatIfResult(routeData);
  const code = forecastCode(f.step_hours);

  return (
    <aside
      className={`panel metrics-panel${loading ? " is-stale" : ""}`}
      aria-label="Route comparison"
      aria-busy={loading}
    >
      <div className="results-head">
        <StepTitle n={4} id="results-title">
          Results
        </StepTitle>
        <span className="forecast-chip" title={f.source_file}>
          {code && <strong>{code}</strong>} {formatUtc(f.valid_time)}
        </span>
      </div>
      {loading && (
        <p className="state-note" role="status">
          <span className="spinner" aria-hidden="true" /> Updating…
        </p>
      )}

      <section className="recommendation">
        <div className="recommendation__label">
          <span className="swatch swatch--weather" />
          {c.routes_identical ? "Routes are identical" : "Recommended: weather-aware route"}
        </div>
        <p className="recommendation__text" data-testid="explanation">
          {c.explanation}
        </p>
      </section>

      <dl className="key-stats" data-testid="tradeoffs">
        <KeyStat
          label="Extra distance"
          value={formatSigned(c.extra_distance_km, 1)}
          unit="km"
          note={`${formatSigned(c.extra_distance_percent, 2)}% vs distance-only`}
        />
        <KeyStat
          label="Weather-cost reduction"
          value={formatNumber(c.weather_cost_reduction_percent, 1)}
          unit="%"
          note="weather-weighted cost"
          tone="good"
        />
        <KeyStat
          label="Severe cells avoided"
          value={formatInteger(c.severe_cells_avoided)}
          note={`${formatInteger(d.category_counts.SEVERE)} → ${formatInteger(w.category_counts.SEVERE)} on route`}
          tone="good"
        />
        <KeyStat
          label="Average hazard"
          value={formatNumber(c.average_hazard_weather_route, 3)}
          note={`vs ${formatNumber(c.average_hazard_distance_route, 3)} distance-only`}
        />
      </dl>
      <dl className="minor-stats">
        <div>
          <dt>High-hazard cells avoided</dt>
          <dd>{formatInteger(c.high_hazard_cells_avoided)}</dd>
        </div>
        <div>
          <dt>Est. fuel delta*</dt>
          <dd>{formatSigned(c.estimated_fuel_delta_percent, 2)} %</dd>
        </div>
      </dl>
      <p className="fineprint">* {c.fuel_proxy_note}</p>

      <h3 className="section-title">Routes</h3>
      <table className="compare-table" data-testid="route-table">
        <thead>
          <tr>
            <th scope="col" />
            <th scope="col">
              <span className="swatch swatch--distance" />
              Distance-only
            </th>
            <th scope="col">
              <span className="swatch swatch--weather" />
              Weather-aware
            </th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <th scope="row">Length (km)</th>
            <td>{formatNumber(d.length_km, 1)}</td>
            <td>{formatNumber(w.length_km, 1)}</td>
          </tr>
          <tr>
            <th scope="row">Weather cost</th>
            <td>{formatNumber(c.distance_route_weather_cost, 1)}</td>
            <td>{formatNumber(c.weather_route_weather_cost, 1)}</td>
          </tr>
          <tr>
            <th scope="row">Average hazard</th>
            <td>{formatNumber(c.average_hazard_distance_route, 3)}</td>
            <td>{formatNumber(c.average_hazard_weather_route, 3)}</td>
          </tr>
          <tr>
            <th scope="row">Maximum hazard</th>
            <td>{formatNumber(c.maximum_hazard_distance_route, 3)}</td>
            <td>{formatNumber(c.maximum_hazard_weather_route, 3)}</td>
          </tr>
        </tbody>
      </table>
      <p className="fineprint">
        Hazard is an experimental 0–1 routing score, not an operational turbulence forecast.
      </p>

      {whatIf && <WhatIfSection whatIf={whatIf} />}

      <h3 className="section-title">Forecast used</h3>
      <dl className="forecast" data-testid="forecast">
        <div>
          <dt>Valid</dt>
          <dd>
            <strong>{formatUtc(f.valid_time)}</strong>
            {code && ` (${code})`}
          </dd>
        </div>
        <div>
          <dt>Model run</dt>
          <dd>
            {formatUtc(f.reference_time)}
            {formatStep(f.step_hours) && ` (${formatStep(f.step_hours)})`}
          </dd>
        </div>
        <div>
          <dt>Source</dt>
          <dd className="forecast__file">{f.source_file}</dd>
        </div>
      </dl>
    </aside>
  );
}
