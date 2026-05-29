/**
 * Service Worker - Fixed Version
 * Cleaned duplicate block
 * Fixed syntax errors
 * Improved caching strategies
 */

if (!self.define) {
  let registry = {};
  let nextDefineUri;

  const singleRequire = (uri, parentUri) => {
    uri = new URL(uri + ".js", parentUri).href;

    return (
      registry[uri] ||
      new Promise((resolve) => {
        if ("document" in self) {
          const script = document.createElement("script");
          script.src = uri;
          script.onload = resolve;
          document.head.appendChild(script);
        } else {
          nextDefineUri = uri;
          importScripts(uri);
          resolve();
        }
      }).then(() => {
        const promise = registry[uri];

        if (!promise) {
          throw new Error(`Module ${uri} didn’t register its module`);
        }

        return promise;
      })
    );
  };

  self.define = (depsNames, factory) => {
    const uri =
      nextDefineUri ||
      ("document" in self ? document.currentScript.src : "") ||
      location.href;

    if (registry[uri]) return;

    let exports = {};

    const require = (depUri) => singleRequire(depUri, uri);

    const specialDeps = {
      module: { uri },
      exports,
      require,
    };

    registry[uri] = Promise.all(
      depsNames.map((depName) => specialDeps[depName] || require(depName))
    ).then((deps) => {
      factory(...deps);
      return exports;
    });
  };
}

define(["./workbox-d20fdc50"], function (workbox) {
  "use strict";

  // Activate updated SW immediately
  self.skipWaiting();

  // Take control of open tabs
  workbox.clientsClaim();

  /**
   * Precache critical app shell
   */
  workbox.precacheAndRoute(
    [
      {
        url: "index.html",
        revision: "0.1qsoophb1t4",
      },
    ],
    {}
  );

  /**
   * Remove old caches
   */
  workbox.cleanupOutdatedCaches();

  /**
   * SPA Routing
   * Handles refreshes on all frontend routes
   */
  workbox.registerRoute(
    new workbox.NavigationRoute(
      workbox.createHandlerBoundToURL("index.html"),
      {
        allowlist: [/.*/],
      }
    )
  );

  /**
   * Market API
   * Fast + background refresh
   */
  workbox.registerRoute(
    /^https:\/\/api\.data\.gov\.in\/.*/i,
    new workbox.StaleWhileRevalidate({
      cacheName: "market-prices-api",
      plugins: [
        new workbox.ExpirationPlugin({
          maxEntries: 10,
          maxAgeSeconds: 3600,
        }),

        new workbox.CacheableResponsePlugin({
          statuses: [0, 200],
        }),
      ],
    }),
    "GET"
  );

  /**
   * Weather API
   * Changed from CacheFirst -> StaleWhileRevalidate
   * Prevents stale weather data
   */
  workbox.registerRoute(
    /^https:\/\/api\.open-meteo\.com\/.*/i,
    new workbox.StaleWhileRevalidate({
      cacheName: "weather-api",

      plugins: [
        new workbox.ExpirationPlugin({
          maxEntries: 20,
          maxAgeSeconds: 1800,
        }),

        new workbox.CacheableResponsePlugin({
          statuses: [0, 200],
        }),
      ],
    }),
    "GET"
  );

  /**
   * Geocoding API
   */
  workbox.registerRoute(
    /^https:\/\/geocoding-api\.open-meteo\.com\/.*/i,
    new workbox.CacheFirst({
      cacheName: "geocoding-api",

      plugins: [
        new workbox.ExpirationPlugin({
          maxEntries: 50,
          maxAgeSeconds: 604800,
        }),

        new workbox.CacheableResponsePlugin({
          statuses: [0, 200],
        }),
      ],
    }),
    "GET"
  );

  /**
   * IP Geo API
   */
  workbox.registerRoute(
    /^https:\/\/get\.geojs\.io\/.*/i,
    new workbox.NetworkFirst({
      cacheName: "ip-geo-api",
      networkTimeoutSeconds: 5,
    }),
    "GET"
  );

  /**
   * Static Assets
   */
  workbox.registerRoute(
    /\.(?:js|css|json)$/,
    new workbox.StaleWhileRevalidate({
      cacheName: "static-resources",

      plugins: [
        new workbox.ExpirationPlugin({
          maxEntries: 100,

          // Reduced from 30 days -> 7 days
          maxAgeSeconds: 604800,
        }),

        new workbox.CacheableResponsePlugin({
          statuses: [0, 200],
        }),
      ],
    }),
    "GET"
  );

  /**
   * Local Images
   */
  workbox.registerRoute(
    /\.(?:png|jpg|jpeg|svg|webp)$/,
    new workbox.CacheFirst({
      cacheName: "images",

      plugins: [
        new workbox.ExpirationPlugin({
          maxEntries: 60,
          maxAgeSeconds: 2592000,
        }),

        new workbox.CacheableResponsePlugin({
          statuses: [0, 200],
        }),
      ],
    }),
    "GET"
  );

  /**
   * Unsplash Images
   */
  workbox.registerRoute(
    /^https:\/\/images\.unsplash\.com\/.*/i,
    new workbox.CacheFirst({
      cacheName: "unsplash-images",

      plugins: [
        new workbox.ExpirationPlugin({
          maxEntries: 10,
          maxAgeSeconds: 2592000,
        }),

        new workbox.CacheableResponsePlugin({
          statuses: [0, 200],
        }),
      ],
    }),
    "GET"
  );

  /**
   * Global error logging
   */
  self.addEventListener("error", (event) => {
    console.error("Service Worker Error:", event.message);
  });

  self.addEventListener("unhandledrejection", (event) => {
    console.error("Unhandled Promise Rejection:", event.reason);
  });
});
