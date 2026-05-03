import { useState, useCallback } from 'react';
import { getCropRecommendations, getSoilTypeSuggestions, getSeasonOptions } from '../services/cropRecommendationApi';
import { useErrorHandler } from './useErrorHandler';

export const useCropRecommendation = () => {
  const { handleWarning } = useErrorHandler();

  // Form state
  const [location, setLocation] = useState('');
  const [soilType, setSoilType] = useState('Loamy Soil');
  const [latitude, setLatitude] = useState(null);
  const [longitude, setLongitude] = useState(null);
  const [season, setSeason] = useState('kharif');
  const [areaSize, setAreaSize] = useState(null);

  // Results state
  const [recommendations, setRecommendations] = useState([]);
  const [weatherData, setWeatherData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [hasSearched, setHasSearched] = useState(false);

  // UI state
  const [selectedCrop, setSelectedCrop] = useState(null);
  const [showDetails, setShowDetails] = useState(false);

  // Suggestions
  const soilTypeSuggestions = getSoilTypeSuggestions();
  const seasonOptions = getSeasonOptions();

  /**
   * Fetch recommendations from API
   */
  const fetchRecommendations = useCallback(async () => {
    if (!location.trim() || !soilType) {
      setError('Please enter location and select soil type');
      handleWarning('Location and soil type are required', 'crop-recommendation');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const result = await getCropRecommendations(
        location,
        soilType,
        latitude,
        longitude,
        season,
        areaSize
      );

      if (result.success) {
        setRecommendations(result.recommendations);
        setWeatherData(result.weather);
        setHasSearched(true);
        setSelectedCrop(result.recommendations[0] || null);
      } else {
        setError(result.error);
        handleWarning(result.error, 'crop-recommendation');
      }
    } catch (err) {
      const errorMsg = 'Failed to fetch recommendations';
      setError(errorMsg);
      handleWarning(errorMsg, 'crop-recommendation');
    } finally {
      setLoading(false);
    }
  }, [location, soilType, latitude, longitude, season, areaSize, handleWarning]);

  /**
   * Reset all form and results
   */
  const resetRecommendations = useCallback(() => {
    setLocation('');
    setSoilType('Loamy Soil');
    setLatitude(null);
    setLongitude(null);
    setSeason('kharif');
    setAreaSize(null);
    setRecommendations([]);
    setWeatherData(null);
    setError(null);
    setHasSearched(false);
    setSelectedCrop(null);
    setShowDetails(false);
  }, []);

  /**
   * Use browser geolocation
   */
  const useGeolocation = useCallback(() => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          setLatitude(position.coords.latitude);
          setLongitude(position.coords.longitude);
          handleWarning('Location captured successfully', 'crop-recommendation');
        },
        (err) => {
          handleWarning('Could not access location. Please enter manually.', 'crop-recommendation');
        }
      );
    } else {
      handleWarning('Geolocation not supported by your browser', 'crop-recommendation');
    }
  }, [handleWarning]);

  return {
    // Form state
    location,
    setLocation,
    soilType,
    setSoilType,
    latitude,
    setLatitude,
    longitude,
    setLongitude,
    season,
    setSeason,
    areaSize,
    setAreaSize,

    // Results state
    recommendations,
    weatherData,
    loading,
    error,
    hasSearched,

    // UI state
    selectedCrop,
    setSelectedCrop,
    showDetails,
    setShowDetails,

    // Suggestions
    soilTypeSuggestions,
    seasonOptions,

    // Actions
    fetchRecommendations,
    resetRecommendations,
    useGeolocation,
  };
};

export default useCropRecommendation;
