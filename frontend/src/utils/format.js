// Display formatting only. Values come straight from the API; nothing here
// recalculates a metric.

const DASH = "—";

function isNum(value) {
  return typeof value === "number" && Number.isFinite(value);
}

export function formatNumber(value, digits = 1) {
  if (!isNum(value)) return DASH;
  return value.toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

/** Like formatNumber but with an explicit "+" for positive values. */
export function formatSigned(value, digits = 1) {
  if (!isNum(value)) return DASH;
  const text = formatNumber(value, digits);
  return value > 0 ? `+${text}` : text;
}

export function formatInteger(value) {
  return isNum(value) ? Math.round(value).toLocaleString("en-US") : DASH;
}

export function formatCoord(point) {
  return point ? `${point.lat.toFixed(4)}, ${point.lon.toFixed(4)}` : DASH;
}

const UTC_FORMAT = new Intl.DateTimeFormat("en-GB", {
  timeZone: "UTC",
  day: "numeric",
  month: "short",
  year: "numeric",
  hour: "2-digit",
  minute: "2-digit",
  hour12: false,
});

const UTC_DAY_FORMAT = new Intl.DateTimeFormat("en-GB", { timeZone: "UTC", day: "numeric", month: "short" });
const UTC_TIME_FORMAT = new Intl.DateTimeFormat("en-GB", {
  timeZone: "UTC",
  hour: "2-digit",
  minute: "2-digit",
  hour12: false,
});

function formatWith(fmt, iso) {
  if (!iso) return DASH;
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? iso : fmt.format(date);
}

/** "2026-10-22T18:00:00Z" -> "22 Oct" (UTC). */
export const formatUtcDay = (iso) => formatWith(UTC_DAY_FORMAT, iso);

/** "2026-10-22T18:00:00Z" -> "18:00" (UTC). */
export const formatUtcTime = (iso) => formatWith(UTC_TIME_FORMAT, iso);

/** "2026-10-22T18:00:00Z" -> "22 Oct 2026, 18:00 UTC". */
export function formatUtc(iso) {
  if (!iso) return DASH;
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? iso : `${UTC_FORMAT.format(date)} UTC`;
}
