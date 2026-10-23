// Forecast list helpers for GET /api/forecasts. Display and selection only:
// which forecasts exist and which is the default are decided by the backend.

function toMillis(iso) {
  const t = new Date(iso).getTime();
  return Number.isNaN(t) ? null : t;
}

/** True if two ISO timestamps are the same instant ("...Z" equals "...+00:00"). */
export function sameInstant(a, b) {
  if (!a || !b) return false;
  const ta = toMillis(a);
  return ta !== null && ta === toMillis(b);
}

/** Forecasts oldest valid time first (the backend already sorts; this keeps it so). */
export function sortForecasts(forecasts) {
  return [...(forecasts || [])].sort((a, b) => toMillis(a.valid_time) - toMillis(b.valid_time));
}

/** The valid time to preselect: the backend default, else the earliest forecast. */
export function initialForecastTime(list) {
  const forecasts = sortForecasts(list?.forecasts);
  const fallback = forecasts[0]?.valid_time ?? null;
  const wanted = forecasts.find((f) => sameInstant(f.valid_time, list?.default_valid_time));
  return wanted?.valid_time ?? fallback;
}

export function findForecast(forecasts, validTime) {
  return (forecasts || []).find((f) => sameInstant(f.valid_time, validTime)) ?? null;
}

/** "+6 h" lead time, or "" when the step is unknown. */
export function formatStep(stepHours) {
  if (typeof stepHours !== "number" || !Number.isFinite(stepHours)) return "";
  const hours = Number.isInteger(stepHours) ? stepHours : stepHours.toFixed(1);
  return `+${hours} h`;
}

/** GFS forecast-hour code from the lead time: 6 -> "f006", or "" when unknown. */
export function forecastCode(stepHours) {
  if (typeof stepHours !== "number" || !Number.isFinite(stepHours)) return "";
  return `f${String(Math.round(stepHours)).padStart(3, "0")}`;
}

/**
 * POST /api/route body. forecast_valid_time is left out when none is chosen;
 * hazard_penalty (what-if K) only when a what-if run asks for it.
 */
export function buildRouteRequest(origin, destination, forecastValidTime, hazardPenalty = null) {
  const body = { origin, destination };
  if (forecastValidTime) body.forecast_valid_time = forecastValidTime;
  if (typeof hazardPenalty === "number") body.hazard_penalty = hazardPenalty;
  return body;
}
