import { useCallback } from 'react';
import { useYieldStore } from '../stores/yieldStore';
import { useYieldApi } from './useYieldApi';

export const useYieldPrediction = () => {
  const { requestYieldPrediction } = useYieldApi();
  const {
    yieldForm,
    updateYieldFormField,
    setYieldForm,
    yieldPrediction,
    setYieldPrediction,
    yieldError,
    setYieldError,
    yieldLoading,
    setYieldLoading,
    showYieldPopup,
    setShowYieldPopup,
    resetYieldStore,
  } = useYieldStore();

  const fetchYield = useCallback(
    async (e) => {
      if (e) e.preventDefault();
      setYieldLoading(true);
      setYieldError(null);
      try {
        const data = await requestYieldPrediction(yieldForm);
        setYieldPrediction(data.predicted_ExpYield);
        setShowYieldPopup(true);
      } catch (error) {
        const errorMessage =
          error?.response?.data?.detail ||
          error.message ||
          'Failed to get prediction';
        setYieldError(errorMessage);
      } finally {
        setYieldLoading(false);
      }
    },
    [
      requestYieldPrediction,
      yieldForm,
      setYieldLoading,
      setYieldError,
      setYieldPrediction,
      setShowYieldPopup,
    ]
  );

  const handleFormChange = useCallback(
    (field, value) => {
      updateYieldFormField(field, value);
    },
    [updateYieldFormField]
  );

  const closeYieldPopup = useCallback(() => {
    setShowYieldPopup(false);
    setYieldPrediction(null);
    setYieldError(null);
  }, [setShowYieldPopup, setYieldPrediction, setYieldError]);

  return {
    yieldForm,
    updateYieldFormField: handleFormChange,
    setYieldForm,
    yieldPrediction,
    yieldError,
    yieldLoading,
    showYieldPopup,
    fetchYield,
    closeYieldPopup,
    resetYieldStore,
  };
};
