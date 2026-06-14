import { useCallback } from 'react';
import toast from 'react-hot-toast';
import { reportErrorToBackend } from '../utils/errorReporting';

const ERROR_TAXONOMY = {
  network: {
    transient: ['NetworkError'],
    timeout: ['AbortError'],
  },
  validation: {
    input: ['ValidationError'],
  },
  runtime: {
    application: ['TypeError', 'ReferenceError'],
  },
};

const ERROR_TRENDS = {
  total: 0,
  byCategory: {},
  byContext: {},
};

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

    for (const [category, groups] of Object.entries(ERROR_TAXONOMY)) {
      for (const [subtype, errors] of Object.entries(groups)) {
        if (errors.includes(error.name)) {
          return `${category}.${subtype}`;
        }
      }
    }

    return 'unknown';
  };

  const buildDiagnosticContext = (error, context) => ({
    context,
    errorName: error?.name || 'UnknownError',
    errorMessage: error?.message || 'No message',
    timestamp: new Date().toISOString(),
    userAgent: navigator.userAgent,
  });

  const trackErrorTrend = (category, context) => {
    ERROR_TRENDS.total += 1;

    ERROR_TRENDS.byCategory[category] =
      (ERROR_TRENDS.byCategory[category] || 0) + 1;

    ERROR_TRENDS.byContext[context] =
      (ERROR_TRENDS.byContext[context] || 0) + 1;
  };

  /**
   * Provide lightweight recovery suggestions
   * @param {string} category
   * @returns {string}
   */
  const getRecoverySuggestion = (category) => {
    const suggestions = {
      'network.transient':
        'Check connectivity and retry the request.',
      'network.timeout':
        'Request timed out. Retry after a short delay.',
      'validation.input':
        'Review submitted information and try again.',
      'runtime.application':
        'Refresh the page and retry the action.',
      unknown:
        'Contact support if the issue persists.',
    };

    return suggestions[category] || suggestions.unknown;
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

      trackErrorTrend(category, context);

      const diagnostics = buildDiagnosticContext(
        error,
        context
      );

      // Log to backend in production
      if (import.meta.env.MODE === 'production') {
        reportErrorToBackend({
          error,
          context,
          timestamp,
          userAgent: navigator.userAgent,
          category,
          recoverySuggestion,
          diagnostics,
          errorTrendSnapshot: ERROR_TRENDS,
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
          diagnostics,
          recoverySuggestion,
          trendAnalysis: ERROR_TRENDS,
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

    trackErrorTrend(category, context);

    const diagnostics = buildDiagnosticContext(
      error,
      context
    );

    if (import.meta.env.MODE === 'production') {
      reportErrorToBackend({
        error,
        context,
        timestamp,
        severity: 'low',
        category,
        recoverySuggestion,
        diagnostics,
        errorTrendSnapshot: ERROR_TRENDS,
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

  const getErrorAnalytics = () => ({
    taxonomy: ERROR_TAXONOMY,
    trends: ERROR_TRENDS,
  });

  return {
    handleError,
    handleWarning,
    handleSilentError,
    handleSuccess,
    getErrorAnalytics,
  };
};