import { useState } from "react";

import { planRoute } from "../services/api.js";
import ForecastSelector from "./ForecastSelector.jsx";
import StepTitle from "./StepTitle.jsx";
import WhatIfControl from "./WhatIfControl.jsx";
import { describeError } from "../utils/errors.js";

const RANGES = { lat: [-90, 90], lon: [-180, 180] };

function CoordinateFields({ id, label, value, onChange, disabled }) {
  return (
    <fieldset className="coord-fields" disabled={disabled}>
      <legend>{label}</legend>
      <div className="coord-fields__row">
        {[
          ["lat", "Latitude"],
          ["lon", "Longitude"],
        ].map(([key, text]) => (
          <label htmlFor={`${id}-${key}`} key={key}>
            {text}
            <input
              id={`${id}-${key}`}
              type="text"
              inputMode="decimal"
              autoComplete="off"
              value={value[key]}
              onChange={(e) => onChange({ ...value, [key]: e.target.value })}
            />
          </label>
        ))}
      </div>
    </fieldset>
  );
}

/**
 * Parse the four text fields into {origin, destination} numbers, or return
 * {problems} describing what is wrong. The backend validates again.
 */
function parseInputs(origin, destination) {
  const problems = [];
  const parse = (name, point) => {
    const out = {};
    for (const key of ["lat", "lon"]) {
      const label = `${name} ${key === "lat" ? "latitude" : "longitude"}`;
      const raw = String(point[key]).trim();
      const num = Number(raw);
      if (raw === "" || !Number.isFinite(num)) {
        problems.push(`${label} must be a number.`);
      } else if (num < RANGES[key][0] || num > RANGES[key][1]) {
        problems.push(`${label} must be between ${RANGES[key][0]} and ${RANGES[key][1]}.`);
      } else {
        out[key] = num;
      }
    }
    return out;
  };
  const o = parse("Origin", origin);
  const d = parse("Destination", destination);
  return problems.length ? { problems } : { origin: o, destination: d };
}

export default function ControlPanel({
  origin,
  destination,
  onOriginChange,
  onDestinationChange,
  forecasts,
  forecastValidTime,
  onForecastChange,
  onForecastRetry,
  hazardPenalty,
  onHazardPenaltyChange,
  hasResult,
  notice,
  loading,
  error,
  onRequestStart,
  onRequestSuccess,
  onRequestError,
}) {
  // Which request is in flight: "route" (normal) or "what-if".
  const [running, setRunning] = useState(null);

  /** Plan routes for the current inputs; `whatIfPenalty` adds a what-if route. */
  async function run(whatIfPenalty = null) {
    if (loading) return;

    const parsed = parseInputs(origin, destination);
    if (parsed.problems) {
      onRequestError({ title: "Check the coordinates", message: parsed.problems.join(" ") });
      return;
    }

    setRunning(whatIfPenalty === null ? "route" : "what-if");
    onRequestStart();
    try {
      onRequestSuccess(
        await planRoute(parsed.origin, parsed.destination, forecastValidTime, whatIfPenalty)
      );
    } catch (err) {
      onRequestError(describeError(err));
    } finally {
      setRunning(null);
    }
  }

  function handleSubmit(event) {
    event.preventDefault();
    run();
  }

  return (
    <aside className="panel control-panel" aria-label="Route inputs">
      <form onSubmit={handleSubmit} noValidate>
        <section className="step" aria-labelledby="step-forecast">
          <StepTitle n={1} id="step-forecast">
            Forecast
          </StepTitle>
          <ForecastSelector
            forecasts={forecasts}
            value={forecastValidTime}
            onChange={onForecastChange}
            onRetry={onForecastRetry}
            disabled={loading}
          />
        </section>

        <section className="step" aria-labelledby="step-points">
          <StepTitle n={2} id="step-points">
            Origin &amp; destination
          </StepTitle>
          <CoordinateFields
            id="origin"
            label="Origin"
            value={origin}
            onChange={onOriginChange}
            disabled={loading}
          />
          <CoordinateFields
            id="destination"
            label="Destination"
            value={destination}
            onChange={onDestinationChange}
            disabled={loading}
          />
          <p className="panel__hint">Decimal degrees. Forecast area: 5–35° N, 65–95° E.</p>
        </section>

        <section className="step step--action" aria-labelledby="step-run">
          <StepTitle n={3} id="step-run">
            Optimize
          </StepTitle>
          <button type="submit" className="button-primary" disabled={loading} aria-busy={loading}>
            {loading && running !== "what-if" ? (
              <>
                <span className="spinner spinner--light" aria-hidden="true" /> Calculating…
              </>
            ) : (
              "Optimize Route"
            )}
          </button>
          {loading && (
            <p className="panel__status" role="status">
              {running === "what-if"
                ? "Calculating weather-aware and what-if routes…"
                : "Calculating distance-only and weather-aware routes…"}
            </p>
          )}
          {notice && (
            <p className="panel__note" role="status">
              {notice}
            </p>
          )}
          {error && !loading && (
            <div className="panel__error" role="alert">
              <strong>{error.title}</strong>
              <span>{error.message}</span>
            </div>
          )}
        </section>

        <WhatIfControl
          value={hazardPenalty}
          onChange={onHazardPenaltyChange}
          onRun={() => run(hazardPenalty)}
          disabled={loading || !hasResult}
          locked={!hasResult}
          running={loading && running === "what-if"}
        />
      </form>
    </aside>
  );
}
