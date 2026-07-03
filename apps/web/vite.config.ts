import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig(() => {
  const env = process.env;
  const agentBaseUrl = env.VITE_AGENT_BASE_URL || "http://127.0.0.1:46181";

  return {
    plugins: [react()],
    server: {
      host: "127.0.0.1",
      port: 46174,
      proxy: {
        "/api": {
          target: agentBaseUrl,
          changeOrigin: true,
        },
      },
    },
  };
});
