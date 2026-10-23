import { formatPenalty } from "../utils/whatIf.js";

/** Map legend. `whatIf` is the response's enabled what-if section, or null. */
export default function MapLegend({ whatIf }) {
  return (
    <div className="map-legend" aria-label="Map legend" data-testid="map-legend">
      <div className="map-legend__item">
        <span className="map-legend__line map-legend__line--distance" />
        Distance-only <em>— baseline</em>
      </div>
      <div className="map-legend__item">
        <span className="map-legend__line map-legend__line--weather" />
        Weather-aware <em>— recommended</em>
      </div>
      {whatIf && (
        <div className="map-legend__item">
          <span className="map-legend__line map-legend__line--what-if" />
          What-if <em>— scenario, K = {formatPenalty(whatIf.hazard_penalty)}</em>
        </div>
      )}
      <div className="map-legend__item">
        <span className="map-legend__ring" />
        Requested point
      </div>
    </div>
  );
}
