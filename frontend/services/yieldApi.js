import apiClient from './api';

export const predictYield = async (payload) => {
  const response = await apiClient.post('/predict', payload, {
    retries: 2,
    errorContext: 'yield-prediction',
    errorMessage: 'Failed to get yield prediction. Please try again.',
  });

  return response.data;
};
