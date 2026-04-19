import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"
import path from 'node:path'


export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: "0.0.0.0",
    port: 5173,
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  // Load .env files from env_config/synced/ instead of project root.
  // dev script uses dotenv-cli to load .env.dev explicitly.
  // build script passes --mode production, so Vite loads .env.production from here.
  envDir: path.resolve(__dirname, "env_config/synced"),
})
