/// <reference types="vitest" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The dashboard only talks to the backend API, under /api (proxied in dev and by nginx).
const backend = process.env.FEDRAD_API_URL ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": { target: backend, rewrite: (path) => path.replace(/^\/api/, "") } },
  },
  preview: {
    port: 4173,
    proxy: { "/api": { target: backend, rewrite: (path) => path.replace(/^\/api/, "") } },
  },
  build: {
    // Plotly alone is ~4.7 MB minified; it is split out and loaded only with the charts.
    chunkSizeWarningLimit: 5000,
    rollupOptions: {
      output: {
        manualChunks: {
          plotly: ["plotly.js-dist-min", "react-plotly.js"],
          mui: ["@mui/material", "@emotion/react", "@emotion/styled"],
        },
      },
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["src/test/setup.ts"],
    css: false,
  },
});
