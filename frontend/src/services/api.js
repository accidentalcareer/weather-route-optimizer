// Thin client for the FastAPI backend. All routing, hazard and comparison
// logic lives in the backend; this module only sends requests and normalises
// errors.

import { API_BASE_URL } from "../config.js";
import { buildRouteRequest } from "../utils/forecasts.js";

export class ApiError extends Error {
  constructor(message, { status = 0, code = "network_error", details = null } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      headers: { "Content-Type": "application/json", ...options.headers },
      ...options,
    });
  } catch {
    throw new ApiError(`Cannot reach the backend at ${API_BASE_URL}. Is it running?`);
  }

  const body = await response.json().catch(() => null);
  if (!response.ok) {
    // Backend errors use {"error": {"code", "message", "details"?}}.
    const err = body?.error;
    throw new ApiError(err?.message || `Request failed with HTTP ${response.status}`, {
      status: response.status,
      code: err?.code || "http_error",
      details: err?.details || null,
    });
  }
  return body;
}

/**
 * Available forecasts, oldest valid time first.
 *
 * @returns {Promise<{default_valid_time: string|null, forecasts: object[]}>}
 *   ForecastListResponse from GET /api/forecasts
 */
export function listForecasts() {
  return request("/api/forecasts");
}

/**
 * Plan distance-only and weather-aware routes.
 *
 * @param {{lat: number, lon: number}} origin
 * @param {{lat: number, lon: number}} destination
 * @param {string|null} [forecastValidTime] valid_time from listForecasts();
 *   omitted to let the backend use its default forecast
 * @param {number|null} [hazardPenalty] what-if hazard sensitivity K (0-10);
 *   omitted for normal routing
 * @returns {Promise<object>} RouteResponse from POST /api/route
 */
export function planRoute(origin, destination, forecastValidTime = null, hazardPenalty = null) {
  return request("/api/route", {
    method: "POST",
    body: JSON.stringify(
      buildRouteRequest(origin, destination, forecastValidTime, hazardPenalty)
    ),
  });
}
