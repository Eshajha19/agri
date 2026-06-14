import { useState, useCallback } from "react";
import { searchLocationByName } from "../services/weatherService";

/**
 * useWeatherLocation Hook
 * 
 * Provides functionality to:
 * - Geocode location names to coordinates (via public Open-Meteo API)
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
      const result = await searchLocationByName(locationName);
      if (result) {
        const newLocation = {
          name: result.name,
          latitude: result.latitude,
          longitude: result.longitude,
        };
        setLocation(newLocation);
        return newLocation;
      }
      throw new Error("Location not found");
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
