import { useCallback } from 'react';

import { predictYield } from '../services/yieldApi';

export const useYieldApi = () => {
  const requestYieldPrediction = useCallback(async (payload) => {
    return predictYield(payload);
  }, []);

  return {
    requestYieldPrediction,
  };
};
