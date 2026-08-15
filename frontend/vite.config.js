import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/upload": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/query": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/methods": { target: "http://127.0.0.1:8000", changeOrigin: true },
    },
  },
});
