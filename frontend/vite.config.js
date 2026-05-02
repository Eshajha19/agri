import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { VitePWA } from 'vite-plugin-pwa'

const spaFallbackPlugin = () => ({
  name: 'spa-fallback',
  configureServer(server) {
    server.middlewares.use((req, res, next) => {
      // List of routes that conflict with .jsx filenames
      const conflictRoutes = ['/advisor', '/community', '/dashboard', '/home', '/auth', '/resources'];
      const urlPath = req.url.split('?')[0].toLowerCase();
      
      if (conflictRoutes.includes(urlPath)) {
        req.url = '/index.html';
      }
      next();
    });
  }
});

const codespaceDevPlugin = () => ({
  name: 'codespace-dev',
  configureServer(server) {
    server.middlewares.use((req, res, next) => {
      res.setHeader('Access-Control-Allow-Origin', '*');
      res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, PUT, PATCH, DELETE');
      res.setHeader('Access-Control-Allow-Headers', 'X-Requested-With,content-type');
      res.setHeader('Access-Control-Allow-Credentials', 'true');
      next();
    });
  }
});

export default defineConfig(() => ({
  plugins: [
    spaFallbackPlugin(),
    codespaceDevPlugin(),
    react(),
    // Legacy browser support removed: React Router 7 requires modern syntax.
    // Minimum supported: Chrome 90+, Android 5+, Safari 14+, Edge 90+
    
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['favicon.ico', 'tractor.png'],
      devOptions: {
        enabled: true
      },
      manifest: {
        name: 'Fasal Saathi - AI-Powered Farming Assistant',
        short_name: 'FasalSaathi',
        description: 'Agriculture App for Farmers with Offline First architecture - Get AI crop predictions, weather insights, market prices, and government schemes',
        theme_color: '#4caf50',
        background_color: '#ffffff',
        display: 'standalone',
        start_url: '/',
        orientation: 'portrait',
        scope: '/',
        categories: ['productivity', 'utilities', 'education'],
        icons: [
          {
            src: '/tractor.png',
            sizes: '192x192',
            type: 'image/png'
          },
          {
            src: '/tractor.png',
            sizes: '512x512',
            type: 'image/png'
          }
        ]
      },
      workbox: {
         globPatterns: ['**/*.{js,css,html,ico,png,svg,woff2,json,webmanifest}'],
          runtimeCaching: [
            // API endpoints for offline data and search
            {
              urlPattern: /^https:\/\/api\.data\.gov\.in\/.*/i,
              handler: 'StaleWhileRevalidate',
              options: {
                cacheName: 'market-prices-api',
                expiration: {
                  maxEntries: 10,
                  maxAgeSeconds: 3600 // 1 hour - market prices refresh
                },
                cacheableResponse: {
                  statuses: [0, 200]
                }
              }
            },
            {
              urlPattern: /^https:\/\/api\.open-meteo\.com\/.*/i,
              handler: 'CacheFirst',
              options: {
                cacheName: 'weather-api',
                expiration: {
                  maxEntries: 20,
                  maxAgeSeconds: 1800 // 30 minutes - weather data gets stale
                },
                cacheableResponse: {
                  statuses: [0, 200]
                }
              }
            },
            {
              urlPattern: /^https:\/\/geocoding-api\.open-meteo\.com\/.*/i,
              handler: 'CacheFirst',
              options: {
                cacheName: 'geocoding-api',
                expiration: {
                  maxEntries: 50,
                  maxAgeSeconds: 7 * 24 * 3600 // 7 days - locations are static
                },
                cacheableResponse: {
                  statuses: [0, 200]
                }
              }
            },
             {
               urlPattern: /^https:\/\/get\.geojs\.io\/.*/i,
               handler: 'NetworkFirst',
               options: {
                 cacheName: 'ip-geo-api',
                 networkTimeoutSeconds: 5
               }
             },
            // Offline fallback for critical pages
             {
               urlPattern: /\.(?:js|css|json)$/,
               handler: 'StaleWhileRevalidate',
               options: {
                 cacheName: 'static-resources',
                 expiration: {
                   maxEntries: 100,
                   maxAgeSeconds: 30 * 24 * 60 * 60
                 },
                 cacheableResponse: {
                   statuses: [0, 200]
                 }
               }
             },
             {
               urlPattern: /\.(?:png|jpg|jpeg|svg|webp)$/,
               handler: 'CacheFirst',
               options: {
                 cacheName: 'images',
                 expiration: {
                   maxEntries: 60,
                   maxAgeSeconds: 30 * 24 * 60 * 60
                 },
                 cacheableResponse: {
                   statuses: [0, 200]
                 }
               }
             },
            // Static assets
             {
               urlPattern: /^https:\/\/images\.unsplash\.com\/.*/i,
               handler: 'CacheFirst',
               options: {
                 cacheName: 'unsplash-images',
                 expiration: {
                   maxEntries: 10,
                   maxAgeSeconds: 30 * 24 * 60 * 60 // 30 days
                 },
                 cacheableResponse: {
                   statuses: [0, 200]
                 }
               }
             }
           ]
        }
      })
    ],
    server: {
      port: 5173,
      host: true,
      cors: true,
      allowedHosts: 'all',
      hmr: {
        overlay: true
      },
      proxy: {
        '/predict': {
          target: 'http://localhost:8000',
          changeOrigin: true
        }
      }
    },
    build: {
      outDir: 'build',
      rollupOptions: {
        external: [],
        output: {
          manualChunks: {
            vendor: ['react', 'react-dom', 'react-router-dom'],
            firebase: ['firebase/app', 'firebase/auth', 'firebase/firestore']
          }
        }
      }
    }
  }))