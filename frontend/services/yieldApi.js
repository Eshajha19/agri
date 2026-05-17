import apiClient from './api';

export const predictYield = async (payload) => {
  const response = await apiClient.post('/predict', payload, {
    retries: 2,
    errorContext: 'yield-prediction',
    errorMessage: 'Failed to get yield prediction. Please try again.',
  });

  return response.data;
};

export const fetchYieldAnalytics = async () => {
  const response = await apiClient.get('/api/yield/analytics', {
    errorContext: 'yield-analytics',
    errorMessage: 'Failed to load yield history.',
  });
  return response.data;
};

export const recordActualYield = async (payload) => {
  const response = await apiClient.post('/api/yield/record-actual', payload, {
    retries: 0,
    retryNonIdempotent: false,
    errorContext: 'yield-record-actual',
    errorMessage: 'Failed to save actual yield.',
  });
  return response.data;
};
