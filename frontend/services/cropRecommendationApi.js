import apiClient from './api';

/**
 * Fetch smart crop recommendations based on location and soil type
 * @param {string} location - Location name (city/region)
 * @param {string} soilType - Type of soil
 * @param {number} latitude - Optional latitude for weather API
 * @param {number} longitude - Optional longitude for weather API
 * @param {string} season - Optional season (kharif, rabi, summer)
 * @param {number} areaSize - Optional farm area size in hectares
 * @returns {Promise} API response with crop recommendations
 */
export const getCropRecommendations = async (
  location,
  soilType,
  latitude = null,
  longitude = null,
  season = null,
  areaSize = null
) => {
  try {
    const payload = {
      location: location || 'Default Location',
      soil_type: soilType,
      latitude,
      longitude,
      season,
      area_size: areaSize,
    };

    const response = await apiClient.post('/api/crop-recommendations', payload);

    if (response.data.success) {
      return {
        success: true,
        location: response.data.location,
        soilType: response.data.soil_type,
        weather: response.data.weather,
        recommendations: response.data.recommendations,
        totalRecommendations: response.data.total_recommendations,
      };
    }

    return {
      success: false,
      error: 'Failed to fetch recommendations',
    };
  } catch (error) {
    console.error('Crop recommendation API error:', error);
    return {
      success: false,
      error: error.response?.data?.detail || 'Unable to fetch crop recommendations',
    };
  }
};

/**
 * Get soil type suggestions for autocomplete
 */
export const getSoilTypeSuggestions = () => {
  return [
    'Black Soil',
    'Red Soil',
    'Loamy Soil',
    'Clayey Soil',
    'Sandy Soil',
    'Sandy-Loam',
    'Alluvial Soil',
    'Silty Soil',
    'Laterite Soil',
  ];
};

/**
 * Get season options
 */
export const getSeasonOptions = () => {
  return [
    { value: 'kharif', label: '🌾 Kharif (Jun-Oct)' },
    { value: 'rabi', label: '🌾 Rabi (Oct-Mar)' },
    { value: 'summer', label: '☀️ Summer (Mar-Jun)' },
  ];
};

/**
 * Calculate compatibility score between crop and conditions
 */
export const calculateCompatibilityPercentage = (recommendations) => {
  return recommendations.map(rec => ({
    ...rec,
    compatibilityPercentage: Math.min(100, rec.score),
  }));
};

export default {
  getCropRecommendations,
  getSoilTypeSuggestions,
  getSeasonOptions,
  calculateCompatibilityPercentage,
};