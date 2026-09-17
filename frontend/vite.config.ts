import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Inside compose the API is reachable as http://api:8000; on the host it is localhost.
const apiTarget = process.env.VITE_API_PROXY ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    // Bind-mounted source on macOS does not deliver inotify events reliably.
    watch: { usePolling: true, interval: 300 },
    proxy: {
      "/api": {
        target: apiTarget,
        changeOrigin: true,
        // SSE must not be buffered by the dev proxy or the run timeline arrives
        // in one lump at the end, which is worse than no timeline at all.
        configure: (proxy) => {
          proxy.on("proxyRes", (proxyRes) => {
            if (proxyRes.headers["content-type"]?.includes("text/event-stream")) {
              proxyRes.headers["cache-control"] = "no-cache, no-transform";
            }
          });
        },
      },
    },
  },
});
