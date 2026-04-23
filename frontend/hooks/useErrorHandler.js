import { useCallback } from 'react';
import toast from 'react-hot-toast';
import { reportErrorToBackend } from '../utils/errorReporting';

/**
 * Custom hook for centralized error handling
 * Provides methods to handle errors with user-friendly notifications
 * and automatic backend logging
 */
export const useErrorHandler = () => {
  /**
   * Handle a critical error - shows toast to user and logs to backend
   * @param {Error} error - The error object
   * @param {string} context - Context where error occurred (e.g., 'yield-prediction')
   * @param {string} userMessage - Custom message to show user (optional)
   */
  const handleError = useCallback(
    (error, context, userMessage = 'An error occurred. Please try again.') => {
      // Log to backend in production
      if (process.env.NODE_ENV === 'production') {
        reportErrorToBackend({
          error,
          context,
          timestamp: new Date().toISOString(),
          userAgent: navigator.userAgent,
        });
      }

      // Show user-friendly toast notification
      toast.error(userMessage, {
        duration: 4000,
        position: 'top-right',
      });

      // Log for debugging in development
      if (process.env.NODE_ENV === 'development') {
        console.warn(`[${context}]`, error);
      }
    },
    []
  );

  /**
   * Handle a non-critical warning - shows subtle notification
   * @param {string} message - Message to display
   * @param {string} context - Context for logging
   */
  const handleWarning = useCallback((message, context) => {
    if (process.env.NODE_ENV === 'development') {
      console.warn(`[${context}]`, message);
    }

    toast(message, {
      duration: 3000,
      position: 'bottom-right',
      icon: '⚠️',
    });
  }, []);

  /**
   * Handle a recoverable error - silent backend logging only
   * @param {Error} error - The error object
   * @param {string} context - Context where error occurred
   */
  const handleSilentError = useCallback((error, context) => {
    if (process.env.NODE_ENV === 'production') {
      reportErrorToBackend({
        error,
        context,
        timestamp: new Date().toISOString(),
        severity: 'low',
      });
    }

    if (process.env.NODE_ENV === 'development') {
      console.warn(`[${context}] (silent)`, error);
    }
  }, []);

  /**
   * Handle a success action - show brief success toast
   * @param {string} message - Message to display
   */
  const handleSuccess = useCallback((message) => {
    toast.success(message, {
      duration: 2000,
      position: 'top-right',
    });
  }, []);

  return {
    handleError,
    handleWarning,
    handleSilentError,
    handleSuccess,
  };
};
