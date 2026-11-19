import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// In development the API runs on :8000; requests to /api and /health are proxied.
// In production set VITE_API_BASE_URL to the backend origin.
export default defineConfig({
  base: '/',
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8000',
      '/health': 'http://127.0.0.1:8000',
    },
  },
  build: {
    outDir: 'dist',
    chunkSizeWarningLimit: 900,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('node_modules/leaflet')) return 'vendor-leaflet'
          if (id.includes('node_modules/recharts') || id.includes('node_modules/d3')) return 'vendor-charts'
          if (id.includes('node_modules/react')) return 'vendor-react'
        },
      },
    },
  },
})
