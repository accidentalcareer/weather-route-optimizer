import { useCallback, useEffect, useState } from "react";

import ControlPanel from "./components/ControlPanel.jsx";
import Header from "./components/Header.jsx";
import MapView from "./components/MapView.jsx";
import MetricsPanel from "./components/MetricsPanel.jsx";
import { listForecasts } from "./services/api.js";
import { findForecast, initialForecastTime, sameInstant, sortForecasts } from "./utils/forecasts.js";
import { formatUtc } from "./utils/format.js";
import { HAZARD_PENALTY } from "./utils/whatIf.js";

// Demo pair: New Delhi -> Mumbai. Kept as strings because they back text inputs.
const INITIAL_ORIGIN = { lat: "28.6139", lon: "77.2090" };
const INITIAL_DESTINATION = { lat: "19.0760", lon: "72.8777" };
// Must match the single-column breakpoint in global.css.
const MOBILE_QUERY = "(max-width: 760px)";

export default function App() {
  const [origin, setOrigin] = useState(INITIAL_ORIGIN);
  const [destination, setDestination] = useState(INITIAL_DESTINATION);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  // Last successful POST /api/route response; the only source of route results.
  const [routeData, setRouteData] = useState(null);
  // GET /api/forecasts result and the selected forecast's valid_time.
  const [forecasts, setForecasts] = useState({ status: "loading", items: [], error: null });
  const [forecastValidTime, setForecastValidTime] = useState(null);
  // What-if hazard sensitivity K; only sent when "Run What-if" is pressed.
  const [hazardPenalty, setHazardPenalty] = useState(HAZARD_PENALTY.normal);

  const loadForecasts = useCallback(async (isCurrent = () => true) => {
    setForecasts((prev) => ({ ...prev, status: "loading", error: null }));
    try {
      const list = await listForecasts();
      if (!isCurrent()) return;
      const items = sortForecasts(list.forecasts);
      setForecasts({ status: "ready", items, error: null });
      // Keep the user's choice if it still exists; otherwise use the backend default.
      setForecastValidTime((current) =>
        findForecast(items, current) ? current : initialForecastTime(list)
      );
    } catch (err) {
      if (!isCurrent()) return;
      setForecasts({ status: "error", items: [], error: err });
      setForecastValidTime(null);
    }
  }, []);

  useEffect(() => {
    let current = true;
    loadForecasts(() => current);
    return () => {
      current = false;
    };
  }, [loadForecasts]);

  const shownForecast = routeData?.forecast?.valid_time;
  const forecastChanged =
    shownForecast && forecastValidTime && !sameInstant(shownForecast, forecastValidTime);

  return (
    <div className="app">
      <Header />
      <main className="app-body">
        <ControlPanel
          origin={origin}
          destination={destination}
          onOriginChange={setOrigin}
          onDestinationChange={setDestination}
          forecasts={forecasts}
          forecastValidTime={forecastValidTime}
          onForecastChange={setForecastValidTime}
          onForecastRetry={() => loadForecasts()}
          hazardPenalty={hazardPenalty}
          onHazardPenaltyChange={setHazardPenalty}
          hasResult={routeData !== null}
          notice={
            forecastChanged && !loading
              ? `Results shown are for the ${formatUtc(shownForecast)} forecast. ` +
                "Run the route again to use the selected forecast."
              : null
          }
          loading={loading}
          error={error}
          onRequestStart={() => {
            setLoading(true);
            setError(null);
          }}
          onRequestSuccess={(data) => {
            setRouteData(data);
            setLoading(false);
            // Stacked mobile layout: the map and results are below the inputs.
            if (window.matchMedia?.(MOBILE_QUERY).matches) {
              document.querySelector(".map-area")?.scrollIntoView({ behavior: "smooth" });
            }
          }}
          onRequestError={(err) => {
            // Keep the previous successful route on screen.
            setError(err);
            setLoading(false);
            // The server's forecast list changed under us: fetch it again.
            if (err.code === "forecast_not_found") loadForecasts();
          }}
        />
        <MapView routeData={routeData} loading={loading} />
        <MetricsPanel routeData={routeData} loading={loading} />
      </main>
    </div>
  );
}
