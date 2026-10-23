import { findForecast, forecastCode, formatStep } from "../utils/forecasts.js";
import { formatUtc, formatUtcDay, formatUtcTime } from "../utils/format.js";

/**
 * Choose which forecast POST /api/route uses: one radio card per forecast,
 * showing its GFS forecast hour (f006...), valid time and lead time.
 *
 * `forecasts` is {status: "loading" | "ready" | "error", items, error} from
 * GET /api/forecasts; `value` is the selected valid_time.
 */
export default function ForecastSelector({ forecasts, value, onChange, onRetry, disabled }) {
  const { status, items, error } = forecasts;
  const selected = findForecast(items, value);

  if (status === "loading") {
    return (
      <p className="state-note" role="status">
        <span className="spinner" aria-hidden="true" /> Loading available forecasts…
      </p>
    );
  }

  if (status === "error") {
    return (
      <div className="state-note state-note--error" role="alert">
        <span>{error?.message || "Could not load the forecast list."}</span>
        <span>Routes will use the server default forecast.</span>
        <button type="button" className="button-link" onClick={onRetry}>
          Retry
        </button>
      </div>
    );
  }

  if (items.length === 0) {
    return (
      <p className="state-note state-note--error" role="alert">
        No processed forecasts are available on the server.
      </p>
    );
  }

  return (
    <fieldset className="forecast-picker" disabled={disabled}>
      <legend className="visually-hidden">Forecast</legend>
      <div className="forecast-picker__grid">
        {items.map((f) => {
          const checked = f === selected;
          const code = forecastCode(f.step_hours);
          return (
            <label
              key={f.valid_time}
              className={`forecast-option${checked ? " is-selected" : ""}`}
              title={`Valid ${formatUtc(f.valid_time)} · ${f.source_file}`}
            >
              <input
                type="radio"
                name="forecast"
                value={f.valid_time}
                checked={checked}
                onChange={() => onChange(f.valid_time)}
              />
              <span className="forecast-option__head">
                <span className="forecast-option__code">{code || "forecast"}</span>
                {f.is_default && <span className="forecast-option__tag">default</span>}
              </span>
              <span className="forecast-option__time">{formatUtcTime(f.valid_time)} UTC</span>
              <span className="forecast-option__step">
                {formatUtcDay(f.valid_time)}
                {formatStep(f.step_hours) && ` · ${formatStep(f.step_hours)}`}
              </span>
            </label>
          );
        })}
      </div>
      {selected && (
        <p className="forecast-picker__summary" data-testid="selected-forecast">
          Selected: <strong>{formatUtc(selected.valid_time)}</strong>
          <span>
            {" "}
            · GFS run {formatUtc(selected.reference_time)}
          </span>
        </p>
      )}
    </fieldset>
  );
}
