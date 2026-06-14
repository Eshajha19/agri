import { useState, useCallback } from "react";

/**
 * useWeatherLocation Hook
 * 
 * Provides functionality to:
 * - Geocode location names to coordinates
 * - Manage weather location state
 * - Handle location selection errors
 */
export const useWeatherLocation = () => {
  const [location, setLocation] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const geocodeLocation = useCallback(async (locationName) => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch("/api/weather/geocode", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ location: locationName }),
      });

      if (!response.ok) {
        throw new Error("Location not found");
      }

      const data = await response.json();

      if (data.success) {
        const newLocation = {
          name: data.location,
          latitude: data.latitude,
          longitude: data.longitude,
        };
        setLocation(newLocation);
        return newLocation;
      }
    } catch (err) {
      const errorMsg = err.message || "Failed to geocode location";
      setError(errorMsg);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const setLocationManually = useCallback((lat, lon, name) => {
    const newLocation = {
      name: name || "Unknown",
      latitude: lat,
      longitude: lon,
    };
    setLocation(newLocation);
    setError(null);
    return newLocation;
  }, []);

  const clearLocation = useCallback(() => {
    setLocation(null);
    setError(null);
  }, []);

  return {
    location,
    loading,
    error,
    geocodeLocation,
    setLocationManually,
    clearLocation,
  };
};

export default useWeatherLocation;
