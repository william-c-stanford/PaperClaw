import { resolve } from 'path'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  resolve: { alias: { '@': resolve(__dirname, 'src') } },
  server: {
    proxy: {
      '/api': { target: 'http://127.0.0.1:8230', changeOrigin: true }
    }
  },
  build: { outDir: 'dist/web' }
})
