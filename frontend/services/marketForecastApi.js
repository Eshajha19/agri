/**
 * marketForecastApi.js — client for the /api/market/forecast backend endpoint.
 *
 * Replaces the Math.random()-based fetchPriceTrends() with real LSTM-backed
 * forecasts from the backend.
 */

import apiClient from './api';

/**
 * Fetch a 14-day LSTM price forecast for a commodity from the backend.
 *
 * @param {string} commodity  - Commodity name (e.g. "Wheat", "Cotton")
 * @param {number} [days=14]  - Forecast horizon (1-30)
 * @returns {Promise<Object>} - Forecast response from backend
 */
export const fetchPriceForecast = async (commodity, days = 14) => {
  try {
    const response = await apiClient.post(
      '/api/market/forecast',
      { commodity, days },
      {
        timeout: 120000, // 120 seconds - LSTM training on first request can be slow
        retries: 0,      // Don't retry slow ML tasks
        errorContext: 'market-price-forecast',
        errorMessage: 'Unable to fetch price forecast. Please try again.',
      }
    );
    return response.data;
  } catch (error) {
    console.error('Price forecast API error:', error);
    // Return a structured fallback so the UI degrades gracefully
    return null;
  }
};

/**
 * List of commodities supported by the forecasting engine.
 * Kept in sync with ml/price_forecaster.py HISTORICAL_PRICES keys.
 */
export const FORECASTABLE_COMMODITIES = [
  'Wheat',
  'Paddy (Dhan)',
  'Cotton',
  'Onion',
  'Soybean',
  'Maize',
];
