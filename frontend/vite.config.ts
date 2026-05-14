import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const backendUrl = "http://localhost:8110";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5174,
    proxy: {
      "/health": backendUrl,
      "/model": backendUrl,
      "/infer": backendUrl,
      "/results": backendUrl,
    },
  },
});
