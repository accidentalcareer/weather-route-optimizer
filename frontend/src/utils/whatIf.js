// What-if hazard sensitivity K (weather multiplier = 1 + K * hazard). The
// backend validates K and computes every route and metric; these helpers only
// handle the control's value and labels.

export const HAZARD_PENALTY = { min: 0, max: 10, step: 0.5, normal: 4 };

/** Slider/input value -> K clamped to [0, 10] on the 0.5 grid (normal if unusable). */
export function parsePenalty(value) {
  const num = typeof value === "number" ? value : Number(String(value).trim());
  if (value === "" || value === null || !Number.isFinite(num)) return HAZARD_PENALTY.normal;
  const clamped = Math.min(HAZARD_PENALTY.max, Math.max(HAZARD_PENALTY.min, num));
  return Math.round(clamped / HAZARD_PENALTY.step) * HAZARD_PENALTY.step;
}

/** 4 -> "4.0", 7.5 -> "7.5". */
export function formatPenalty(k) {
  return typeof k === "number" && Number.isFinite(k) ? k.toFixed(1) : "—";
}

/** What K means relative to normal routing. */
export function sensitivityLabel(k) {
  if (k === HAZARD_PENALTY.normal) return "Normal weather sensitivity (default)";
  if (k === 0) return "Weather ignored: same as the distance-only route";
  return k > HAZARD_PENALTY.normal
    ? "Stronger avoidance of hazardous cells than normal"
    : "Less weather-sensitive than normal";
}

/** The response's what-if section if what-if was run, else null. */
export function whatIfResult(routeData) {
  const w = routeData?.what_if;
  return w && w.enabled && w.what_if_route && w.comparison ? w : null;
}
