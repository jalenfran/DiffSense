import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
    plugins: [react()],    server: {
        port: 5173,
        proxy: {
            '/api': {
                target: 'http://76.125.217.28:8080',
                changeOrigin: true,
                rewrite: (path) => path.replace(/^\/api/, '/api')
            },
            '/auth': {
                target: 'http://localhost:3000',
                changeOrigin: true
            },
            '/repos': {
                target: 'http://localhost:3000',
                changeOrigin: true
            },
            '/repositories': {
                target: 'http://localhost:3000',
                changeOrigin: true
            }
        }
    },
    build: {
        outDir: '../oauth-server/dist'
    }
})
