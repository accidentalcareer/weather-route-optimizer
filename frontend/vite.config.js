import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// VITE_* variables are read from ../.env so config stays in one place.
export default defineConfig({
  plugins: [react()],
  envDir: "..",
});
