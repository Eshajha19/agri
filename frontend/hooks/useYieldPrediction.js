import { useCallback } from 'react';
import { useYieldStore } from '../stores/yieldStore';
import { useErrorHandler } from './useErrorHandler';

export const useYieldPrediction = () => {
  const { handleError } = useErrorHandler();
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
        const response = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(yieldForm),
        });
        if (!response.ok) {
          throw new Error(`Server error: ${response.status}`);
        }
        const data = await response.json();
        setYieldPrediction(data.predicted_ExpYield);
        setShowYieldPopup(true);
      } catch (error) {
        const errorMessage = error.message || 'Failed to get prediction';
        handleError(error, 'yield-prediction', 'Failed to get yield prediction. Please try again.');
        setYieldError(errorMessage);
      } finally {
        setYieldLoading(false);
      }
    },
    [yieldForm, setYieldLoading, setYieldError, setYieldPrediction, setShowYieldPopup]
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
