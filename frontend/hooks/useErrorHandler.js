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
   * Categorize errors for better diagnostics
   * @param {Error} error
   * @returns {string}
   */
  const getErrorCategory = (error) => {
    if (!error) return 'unknown';

    switch (error.name) {
      case 'TypeError':
        return 'runtime';
      case 'NetworkError':
        return 'network';
      case 'ValidationError':
        return 'validation';
      case 'SyntaxError':
        return 'syntax';
      default:
        return 'unknown';
    }
  };

  /**
   * Provide lightweight recovery suggestions
   * @param {string} category
   * @returns {string}
   */
  const getRecoverySuggestion = (category) => {
    switch (category) {
      case 'network':
        return 'Please check your internet connection and try again.';
      case 'validation':
        return 'Please review the provided input and try again.';
      case 'runtime':
        return 'Refresh the page and retry the action.';
      case 'syntax':
        return 'Reload the application and try again.';
      default:
        return 'Please try again later or contact support if the issue persists.';
    }
  };

  /**
   * Handle a critical error - shows toast to user and logs to backend
   * @param {Error} error - The error object
   * @param {string} context - Context where error occurred (e.g., 'yield-prediction')
   * @param {string} userMessage - Custom message to show user (optional)
   */
  const handleError = useCallback(
    (error, context, userMessage = 'An error occurred. Please try again.') => {
      const timestamp = new Date().toISOString();
      const category = getErrorCategory(error);
      const recoverySuggestion = getRecoverySuggestion(category);

      // Log to backend in production
      if (import.meta.env.MODE === 'production') {
        reportErrorToBackend({
          error,
          context,
          timestamp,
          userAgent: navigator.userAgent,
          category,
          recoverySuggestion,
        });
      }

      // Show user-friendly toast notification
      toast.error(userMessage, {
        duration: 4000,
        position: 'top-right',
      });

      // Log detailed diagnostics in development
      if (import.meta.env.MODE === 'development') {
        console.warn(`[${context}]`, {
          error,
          category,
          timestamp,
          recoverySuggestion,
        });
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
    if (import.meta.env.MODE === 'development') {
      console.warn(`[${context}]`, {
        message,
        timestamp: new Date().toISOString(),
      });
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
    const timestamp = new Date().toISOString();
    const category = getErrorCategory(error);
    const recoverySuggestion = getRecoverySuggestion(category);

    if (import.meta.env.MODE === 'production') {
      reportErrorToBackend({
        error,
        context,
        timestamp,
        severity: 'low',
        category,
        recoverySuggestion,
      });
    }

    if (import.meta.env.MODE === 'development') {
      console.warn(`[${context}] (silent)`, {
        error,
        category,
        timestamp,
        recoverySuggestion,
      });
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