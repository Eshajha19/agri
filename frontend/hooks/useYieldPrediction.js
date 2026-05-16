/**
 * useYieldPrediction — single authoritative hook for the yield prediction workflow.
 *
 * Consolidation notes
 * -------------------
 * Previously the prediction logic was split across three layers:
 *
 *   1. yieldStore.fetchYield   — a store action that called predictYield directly,
 *                                managed its own loading/error/popup state, and was
 *                                never actually called by any component.
 *   2. useYieldApi             — a thin hook that wrapped predictYield() in a
 *                                useCallback with no additional logic.
 *   3. useYieldPrediction      — the hook actually used by components, which called
 *                                useYieldApi and then manually managed the same
 *                                loading/error/popup state as the store action.
 *
 * This created two divergent fetchYield implementations with different error
 * formats, popup synchronization, and loading state ownership.
 *
 * Fix: all prediction logic now lives exclusively here.
 *   - predictYield() is imported directly (no useYieldApi indirection).
 *   - yieldStore.fetchYield has been removed; the store is now a pure state
 *     container with no async logic.
 *   - useYieldApi.js has been deleted as it is no longer needed.
 */

import { useCallback } from 'react';
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

  const fetchYield = useCallback(
    async (e) => {
      if (e) e.preventDefault();
      setYieldLoading(true);
      setYieldError(null);
      try {
        const data = await predictYield(yieldForm);
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
