import { defineConfig } from "vite";

// The Python server is the only referee and the only source of game state, so
// everything under /api and /art is proxied to it rather than mocked here.
const BACKEND = "http://127.0.0.1:8000";

export default defineConfig({
  server: {
    port: 5173,
    proxy: {
      "/api": { target: BACKEND, changeOrigin: true },
      "/art": { target: BACKEND, changeOrigin: true },
    },
  },
});
