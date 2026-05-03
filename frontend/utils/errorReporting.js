/**
 * Error reporting utility for backend logging
 * Sends error details to the backend for monitoring and debugging
 */

const ERROR_LOG_ENDPOINT = '/api/log-error';

/**
 * Report an error to the backend
 * @param {Object} errorData - Error data object
 * @param {Error} errorData.error - The error object
 * @param {string} errorData.context - Context where error occurred
 * @param {string} errorData.timestamp - ISO timestamp
 * @param {string} errorData.userAgent - Browser user agent
 * @param {string} errorData.severity - Error severity ('high', 'medium', 'low')
 */
export const reportErrorToBackend = async (errorData) => {
  try {
    const payload = {
      message: errorData.error?.message || 'Unknown error',
      stack: errorData.error?.stack || '',
      context: errorData.context,
      timestamp: errorData.timestamp,
      userAgent: errorData.userAgent || navigator.userAgent,
      severity: errorData.severity || 'high',
      url: window.location.href,
    };

    // Only attempt to report in browser environments
    if (typeof fetch !== 'undefined') {
      await fetch(ERROR_LOG_ENDPOINT, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      }).catch(() => {
        // Silently fail if backend logging endpoint is unavailable
        // This prevents errors in error reporting from breaking the app
      });
    }
  } catch (e) {
    // Prevent error reporting from causing cascading failures
    if (import.meta.env.MODE === 'development') {
      console.warn('[errorReporting] Failed to report error:', e);
    }
  }
};

/**
 * Format error message for display
 * Removes sensitive details and creates user-friendly message
 * @param {Error} error - Error object
 * @returns {string} Formatted error message
 */
export const formatErrorMessage = (error) => {
  if (typeof error === 'string') {
    return error;
  }

  if (error?.response?.status === 404) {
    return 'Resource not found. Please try again.';
  }

  if (error?.response?.status === 500) {
    return 'Server error. Please try again later.';
  }

  if (error?.message) {
    // Hide technical details in production
    if (import.meta.env.MODE === 'production') {
      // Check for common error patterns and provide friendly messages
      if (error.message.includes('NetworkError') || error.message.includes('fetch')) {
        return 'Network error. Please check your connection.';
      }
      if (error.message.includes('timeout')) {
        return 'Request timed out. Please try again.';
      }
      return 'An error occurred. Please try again.';
    }
    return error.message;
  }

  return 'An unexpected error occurred.';
};
