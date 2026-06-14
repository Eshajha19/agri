import { useState, useCallback, useRef } from "react";

/**
 * useWeatherLocation Hook
 *
 * Provides functionality to:
 * - Geocode location names to coordinates
 * - Manage weather location state
 * - Handle location selection errors
 * - Cache recently resolved locations
 * - Track lookup performance metrics
 */
export const useWeatherLocation = () => {
  const [location, setLocation] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const CACHE_TTL_MS = 5 * 60 * 1000;

  const cacheRef = useRef(new Map());

  const metricsRef = useRef({
    cacheHits: 0,
    cacheMisses: 0,
    cacheExpirations: 0,
    totalLookups: 0,
  });

  const cleanupExpiredCache = useCallback(() => {
    const now = Date.now();

    for (const [key, value] of cacheRef.current.entries()) {
      if (now - value.timestamp > CACHE_TTL_MS) {
        cacheRef.current.delete(key);
        metricsRef.current.cacheExpirations += 1;
      }
    }
  }, []);

  const getCachedLocation = useCallback(
    (locationName) => {
      cleanupExpiredCache();

      const cacheKey = String(locationName)
        .trim()
        .toLowerCase();

      const cached = cacheRef.current.get(cacheKey);

      if (!cached) {
        return null;
      }

      metricsRef.current.cacheHits += 1;

      return cached.data;
    },
    [cleanupExpiredCache]
  );

  const cacheLocationResult = useCallback(
    (locationName, data) => {
      const cacheKey = String(locationName)
        .trim()
        .toLowerCase();

      cacheRef.current.set(cacheKey, {
        data,
        timestamp: Date.now(),
      });
    },
    []
  );

  const geocodeLocation = useCallback(
    async (locationName) => {
      setLoading(true);
      setError(null);

      metricsRef.current.totalLookups += 1;

      const cached = getCachedLocation(locationName);

      if (cached) {
        setLocation(cached);
        setLoading(false);
        return cached;
      }

      metricsRef.current.cacheMisses += 1;

      try {
        const response = await fetch(
          "/api/weather/geocode",
          {
            method: "POST",
            headers: {
              "Content-Type":
                "application/json",
            },
            body: JSON.stringify({
              location: locationName,
            }),
          }
        );

        if (!response.ok) {
          throw new Error(
            "Location not found"
          );
        }

        const data = await response.json();

        if (data.success) {
          const newLocation = {
            name: data.location,
            latitude: data.latitude,
            longitude: data.longitude,
          };

          cacheLocationResult(
            locationName,
            newLocation
          );

          setLocation(newLocation);

          return newLocation;
        }

        throw new Error(
          "Failed to resolve location"
        );
      } catch (err) {
        const errorMsg =
          err.message ||
          "Failed to geocode location";

        setError(errorMsg);
        throw err;
      } finally {
        setLoading(false);
      }
    },
    [
      getCachedLocation,
      cacheLocationResult,
    ]
  );

  const setLocationManually = useCallback(
    (lat, lon, name) => {
      const newLocation = {
        name: name || "Unknown",
        latitude: lat,
        longitude: lon,
      };

      cacheLocationResult(
        name || "Unknown",
        newLocation
      );

      setLocation(newLocation);
      setError(null);

      return newLocation;
    },
    [cacheLocationResult]
  );

  const clearLocation = useCallback(() => {
    setLocation(null);
    setError(null);

    cacheRef.current.clear();
  }, []);

  return {
    location,
    loading,
    error,
    geocodeLocation,
    setLocationManually,
    clearLocation,
    locationMetrics: metricsRef.current,
    cacheSize: cacheRef.current.size,
  };
};

export default useWeatherLocation;