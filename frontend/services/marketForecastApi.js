/**
 * marketForecastApi.js — client for the /api/market/forecast backend endpoint.
 *
 * Replaces the Math.random()-based fetchPriceTrends() with real LSTM-backed
 * forecasts from the backend.
 */

import apiClient from './api';

/**
 * Normalize and validate forecast horizon values.
 *
 * Rules:
 * - Convert incoming values to numbers.
 * - Default invalid values to 14 days.
 * - Clamp values between 1 and 30.
 * - Convert decimal values to integers.
 */
const normalizeForecastDays = (days) => {
  const parsedDays = Number(days);

  // Fallback for NaN, Infinity, undefined, null, etc.
  if (!Number.isFinite(parsedDays)) {
    return 14;
  }

  // Ensure integer day values
  const normalizedDays = Math.floor(parsedDays);

  // Clamp to supported backend range (1–30 days)
  return Math.min(Math.max(normalizedDays, 1), 30);
};

/**
 * Fetch a price forecast for a commodity from the backend.
 *
 * Validation hardening:
 * - Prevents negative values
 * - Prevents zero-day forecasts
 * - Prevents values above backend limits
 * - Handles invalid numeric inputs gracefully
 * - Normalizes decimal inputs
 *
 * @param {string} commodity - Commodity name (e.g. "Wheat", "Cotton")
 * @param {number} [days=14] - Forecast horizon (1-30)
 * @returns {Promise<Object|null>}
 */
export const fetchPriceForecast = async (commodity, days = 14) => {
  // Normalize edge-case numeric inputs before sending to backend
  const normalizedDays = normalizeForecastDays(days);

  try {
    const response = await apiClient.post(
      '/api/market/forecast',
      {
        commodity,
        days: normalizedDays,
      },
      {
        timeout: 120000, // LSTM training may take time on first request
        retries: 0, // Avoid retrying expensive ML jobs
        errorContext: 'market-price-forecast',
        errorMessage:
          'Unable to fetch price forecast. Please try again.',
      }
    );

    return response.data;
  } catch (error) {
    // Enhanced diagnostics for validation and forecasting failures
    console.error(
      'Price forecast API error:',
      error,
      {
        commodity,
        requestedDays: days,
        normalizedDays,
      }
    );

    // Graceful degradation
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