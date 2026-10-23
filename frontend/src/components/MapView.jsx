import { useEffect } from "react";
import {
  CircleMarker,
  MapContainer,
  Polyline,
  Popup,
  TileLayer,
  Tooltip,
  useMap,
} from "react-leaflet";

import { INITIAL_MAP_VIEW } from "../config.js";
import { formatCoord, formatNumber } from "../utils/format.js";
import { formatPenalty, whatIfResult } from "../utils/whatIf.js";
import MapLegend from "./MapLegend.jsx";

// smoothFactor 0 turns off Leaflet's on-screen line simplification, so every
// route coordinate from the API is drawn as a vertex.
const ROUTE_STYLES = {
  weather: { color: "#1d5fb8", weight: 5, opacity: 1, smoothFactor: 0 },
  weatherCasing: { color: "#ffffff", weight: 9, opacity: 0.9, smoothFactor: 0 },
  distance: { color: "#4a5462", weight: 3, opacity: 0.9, dashArray: "8 7", smoothFactor: 0 },
  // What-if scenario: orange dots, drawn above the weather route so both stay visible.
  whatIf: {
    color: "#c2410c", weight: 4, opacity: 1, dashArray: "1 8", lineCap: "round", smoothFactor: 0,
  },
};
const ENDPOINT_COLORS = { origin: "#12805c", destination: "#b3261e" };

/** API coordinates [{lat, lon}] -> Leaflet [lat, lng] pairs, unchanged otherwise. */
const toLatLngs = (coords) => coords.map((c) => [c.lat, c.lon]);

// Extra bottom padding keeps routes clear of the legend in the bottom-left corner.
const FIT_OPTIONS = { paddingTopLeft: [48, 48], paddingBottomRight: [48, 112] };

/** All routes plus the requested points, as Leaflet bounds. */
function routeBounds(routeData) {
  return [
    ...toLatLngs(routeData.distance_route.coordinates),
    ...toLatLngs(routeData.weather_route.coordinates),
    ...toLatLngs(whatIfResult(routeData)?.what_if_route.coordinates ?? []),
    [routeData.origin.requested.lat, routeData.origin.requested.lon],
    [routeData.destination.requested.lat, routeData.destination.requested.lon],
  ];
}

/** Fit the view to the routes whenever a new result arrives. */
function FitToRoutes({ routeData }) {
  const map = useMap();
  useEffect(() => {
    if (routeData) map.fitBounds(routeBounds(routeData), FIT_OPTIONS);
  }, [map, routeData]);
  return null;
}

/**
 * Keep Leaflet's size in sync when the layout changes (breakpoints, window
 * resize) and re-fit so the whole route stays visible in the new size.
 */
function ResizeWatcher({ routeData }) {
  const map = useMap();
  useEffect(() => {
    const observer = new ResizeObserver(() => {
      map.invalidateSize();
      if (routeData) map.fitBounds(routeBounds(routeData), FIT_OPTIONS);
    });
    observer.observe(map.getContainer());
    return () => observer.disconnect();
  }, [map, routeData]);
  return null;
}

function EndpointPopup({ label, location }) {
  return (
    <div className="endpoint-popup">
      <strong>{label}</strong>
      <div>Requested: {formatCoord(location.requested)}</div>
      <div>Snapped: {formatCoord(location.snapped)}</div>
      <div>Snap distance: {formatNumber(location.snap_distance_km, 1)} km</div>
    </div>
  );
}

/**
 * Snapped grid point (filled, labelled) where the route starts/ends, plus the
 * requested point (small ring) joined to it by a faint dotted line.
 */
function Endpoint({ label, location, color }) {
  const requested = [location.requested.lat, location.requested.lon];
  const snapped = [location.snapped.lat, location.snapped.lon];
  const popup = (
    <Popup>
      <EndpointPopup label={label} location={location} />
    </Popup>
  );
  return (
    <>
      <Polyline
        positions={[requested, snapped]}
        pathOptions={{ color, weight: 1.5, opacity: 0.8, dashArray: "2 4" }}
        interactive={false}
      />
      <CircleMarker
        center={requested}
        radius={4}
        pathOptions={{ color, weight: 2, fillColor: "#ffffff", fillOpacity: 1 }}
      >
        {popup}
      </CircleMarker>
      <CircleMarker
        center={snapped}
        radius={7}
        pathOptions={{ color: "#ffffff", weight: 2, fillColor: color, fillOpacity: 1 }}
      >
        <Tooltip permanent direction="top" offset={[0, -8]} className="endpoint-label">
          {label}
        </Tooltip>
        {popup}
      </CircleMarker>
    </>
  );
}

export default function MapView({ routeData, loading }) {
  const whatIf = whatIfResult(routeData);
  return (
    <section className="map-area" aria-label="Map" aria-busy={loading}>
      <MapContainer
        center={INITIAL_MAP_VIEW.center}
        zoom={INITIAL_MAP_VIEW.zoom}
        className="map-area__map"
        worldCopyJump
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        <ResizeWatcher routeData={routeData} />

        {routeData && (
          <>
            {/* Baseline first so the recommended route draws on top. */}
            <Polyline
              positions={toLatLngs(routeData.distance_route.coordinates)}
              pathOptions={{ ...ROUTE_STYLES.distance, className: "route route--distance" }}
            >
              <Tooltip sticky>
                Distance-only · {formatNumber(routeData.distance_route.length_km, 1)} km
              </Tooltip>
            </Polyline>
            <Polyline
              positions={toLatLngs(routeData.weather_route.coordinates)}
              pathOptions={{ ...ROUTE_STYLES.weatherCasing, className: "route route--casing" }}
              interactive={false}
            />
            <Polyline
              positions={toLatLngs(routeData.weather_route.coordinates)}
              pathOptions={{ ...ROUTE_STYLES.weather, className: "route route--weather" }}
            >
              <Tooltip sticky>
                Weather-aware (recommended) ·{" "}
                {formatNumber(routeData.weather_route.length_km, 1)} km
              </Tooltip>
            </Polyline>
            {whatIf && (
              <Polyline
                positions={toLatLngs(whatIf.what_if_route.coordinates)}
                pathOptions={{ ...ROUTE_STYLES.whatIf, className: "route route--what-if" }}
              >
                <Tooltip sticky>
                  What-if (K = {formatPenalty(whatIf.hazard_penalty)}) ·{" "}
                  {formatNumber(whatIf.what_if_route.length_km, 1)} km
                </Tooltip>
              </Polyline>
            )}
            <Endpoint label="Origin" location={routeData.origin} color={ENDPOINT_COLORS.origin} />
            <Endpoint
              label="Destination"
              location={routeData.destination}
              color={ENDPOINT_COLORS.destination}
            />
            <FitToRoutes routeData={routeData} />
          </>
        )}
      </MapContainer>

      {routeData && <MapLegend whatIf={whatIf} />}
      {loading && (
        <div className="map-status" role="status">
          Calculating weather-aware routes…
        </div>
      )}
    </section>
  );
}
