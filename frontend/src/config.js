// Runtime configuration. Vite exposes VITE_* variables from the repo-root .env
// (see envDir in vite.config.js).

const DEFAULT_API_BASE_URL = "http://localhost:8000";

export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || DEFAULT_API_BASE_URL).replace(
  /\/+$/,
  ""
);

// Initial map view: the current GFS subset covers 5-35 N, 65-95 E.
export const INITIAL_MAP_VIEW = { center: [20, 80], zoom: 5 };
