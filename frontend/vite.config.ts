import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8008",
    },
  },
  build: {
    // Build straight into the Python package: the wheel picks it up as an
    // artifact, and `doctest-dashboard serve` finds it in dev and installs
    // alike. No files exist here on a fresh clone — that must stay a valid
    // state (uv sync works without Node).
    outDir: "../doctest_dashboard/static",
    emptyOutDir: true,
  },
});
