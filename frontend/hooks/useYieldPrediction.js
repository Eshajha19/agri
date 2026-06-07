import { useCallback, useEffect, useRef } from 'react';
import { useYieldStore } from '../stores/yieldStore';
import { predictYield } from '../services/yieldApi';

export const useYieldPrediction = () => {
  const {
    yieldForm,
    updateYieldFormField,
    setYieldForm,
    yieldPrediction,
    yieldLastUpdated,
    setYieldPrediction,
    yieldError,
    setYieldError,
    yieldLoading,
    setYieldLoading,
    showYieldPopup,
    setShowYieldPopup,
    resetYieldStore,
  } = useYieldStore();

  const mountedRef = useRef(true);
  const predictionRequestIdRef = useRef(0);
  const predictionInProgressRef = useRef(false);

  useEffect(() => {
    mountedRef.current = true;

    return () => {
      mountedRef.current = false;
      predictionRequestIdRef.current++;
    };
  }, []);

  const fetchYield = useCallback(
    async (e) => {
      if (e) e.preventDefault();

      if (predictionInProgressRef.current) {
        return;
      }

      predictionInProgressRef.current = true;
      const requestId = ++predictionRequestIdRef.current;

      if (
        mountedRef.current &&
        requestId === predictionRequestIdRef.current
      ) {
        setYieldLoading(true);
        setYieldError(null);
      }

      try {
        const data = await predictYield(yieldForm);

        if (
          !mountedRef.current ||
          requestId !== predictionRequestIdRef.current
        ) {
          return;
        }

        setYieldPrediction(data.predicted_ExpYield);
        setShowYieldPopup(true);
      } catch (error) {
        const errorMessage =
          error?.response?.data?.detail ||
          error?.message ||
          'Failed to get prediction';

        if (
          mountedRef.current &&
          requestId === predictionRequestIdRef.current
        ) {
          setYieldError(errorMessage);
        }
      } finally {
        predictionInProgressRef.current = false;

        if (
          mountedRef.current &&
          requestId === predictionRequestIdRef.current
        ) {
          setYieldLoading(false);
        }
      }
    },
    [
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
    predictionRequestIdRef.current++;

    setShowYieldPopup(false);
    setYieldPrediction(null);
    setYieldError(null);
  }, [
    setShowYieldPopup,
    setYieldPrediction,
    setYieldError,
  ]);

  return {
    yieldForm,
    updateYieldFormField: handleFormChange,
    setYieldForm,
    yieldPrediction,
    yieldLastUpdated,
    yieldError,
    yieldLoading,
    showYieldPopup,
    setShowYieldPopup,
    fetchYield,
    closeYieldPopup,
    resetYieldStore,
  };
};