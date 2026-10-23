// Turn an ApiError (services/api.js) into a short title and message for users.
// Messages come from the backend, which already writes them for people; only
// request-validation details are reworded here.

const TITLES = {
  invalid_request: "Invalid input",
  invalid_coordinate: "Invalid coordinates",
  outside_grid: "Outside the forecast area",
  same_grid_point: "Points too close together",
  forecast_not_found: "Forecast not available",
  invalid_hazard_penalty: "Invalid hazard sensitivity",
  route_not_found: "No route found",
  weather_data_unavailable: "Weather data unavailable",
  internal_error: "Server error",
  network_error: "Backend unreachable",
  http_error: "Request failed",
};

const FIELD_LABELS = {
  "origin.lat": "Origin latitude",
  "origin.lon": "Origin longitude",
  "destination.lat": "Destination latitude",
  "destination.lon": "Destination longitude",
  forecast_valid_time: "Forecast",
  hazard_penalty: "Hazard sensitivity",
};

export function describeError(error) {
  const code = error?.code || "http_error";
  const title = TITLES[code] || TITLES.http_error;

  if (code === "invalid_request" && Array.isArray(error.details) && error.details.length) {
    const message = error.details
      .map((d) => `${FIELD_LABELS[d.field] || d.field}: ${d.message.toLowerCase()}`)
      .join("; ");
    return { code, title, message };
  }
  return { code, title, message: error?.message || "Something went wrong. Please try again." };
}
