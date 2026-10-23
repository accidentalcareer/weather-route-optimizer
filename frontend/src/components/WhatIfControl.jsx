import StepTitle from "./StepTitle.jsx";
import {
  HAZARD_PENALTY,
  formatPenalty,
  parsePenalty,
  sensitivityLabel,
} from "../utils/whatIf.js";

const { min, max, step, normal } = HAZARD_PENALTY;

/**
 * Hazard sensitivity K for a what-if run. Moving the slider only changes the
 * value; a request is sent only when "Run What-if" is pressed. `locked` means
 * there is no route yet, so the scenario has nothing to compare against.
 */
export default function WhatIfControl({ value, onChange, onRun, disabled, locked, running }) {
  return (
    <section className="step step--secondary" aria-labelledby="step-what-if">
      <StepTitle n={5} id="step-what-if" optional>
        What-if scenario
      </StepTitle>
      <fieldset className="what-if" disabled={disabled} data-testid="what-if-control">
        <legend className="visually-hidden">What-if</legend>
        {locked && (
          <p className="what-if__locked">Optimize a route first, then compare a scenario with it.</p>
        )}
        <div className="what-if__header">
          <label htmlFor="hazard-penalty">Hazard sensitivity</label>
          <output htmlFor="hazard-penalty" className="what-if__value" aria-live="polite">
            K = {formatPenalty(value)}
          </output>
        </div>
        <input
          id="hazard-penalty"
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(parsePenalty(e.target.value))}
          aria-describedby="hazard-penalty-help"
          aria-valuetext={`K = ${formatPenalty(value)}`}
        />
        <div className="what-if__scale" aria-hidden="true">
          <span>{min} · less</span>
          <span className="what-if__normal" style={{ left: `${(normal / max) * 100}%` }}>
            {normal} · normal
          </span>
          <span>{max} · more</span>
        </div>
        <p id="hazard-penalty-help" className="what-if__help">
          <strong>{sensitivityLabel(value)}.</strong> K = {normal} is the normal weather
          sensitivity. Higher K avoids hazardous cells more strongly; lower K is less
          weather-sensitive.
        </p>
        <div className="what-if__actions">
          <button type="button" className="button-secondary" onClick={onRun} aria-busy={running}>
            {running ? "Running what-if…" : "Run What-if"}
          </button>
          {value !== normal && (
            <button type="button" className="button-link" onClick={() => onChange(normal)}>
              Reset to K = {formatPenalty(normal)}
            </button>
          )}
        </div>
      </fieldset>
    </section>
  );
}
