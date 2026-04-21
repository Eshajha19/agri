import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => ({
  plugins: [
    react()
  ],
  server: {
    host: true,
    hmr: {
      overlay: false
    },
    proxy: {
      '/predict': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      }
    }
  },
  build: {
    outDir: 'dist'
  }
}))