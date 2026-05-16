/**
 * Feedback Service
 * Provides secure API calls for feedback operations with error handling and validation.
 * Uses the unified apiClient (lib/apiClient.js) so all requests get auth injection,
 * retries, global loader tracking, and offline queue support automatically.
 */

import apiClient from '../lib/apiClient';

/**
 * Submit feedback to the secure backend API
 * @param {Object} feedbackData - Feedback data to submit
 * @returns {Promise<Object>} - Response from API
 */
export const submitFeedback = async (feedbackData) => {
  try {
    const response = await apiClient.post('/api/feedback', feedbackData, {
      errorContext: 'feedback-submit',
      errorMessage: 'Failed to submit feedback. Please try again.',
    });

    const result = response.data;

    if (!result.success) {
      throw new Error(result.message || 'Feedback submission failed');
    }

    return {
      success: true,
      data: result,
      feedbackId: result.feedback_id,
    };
  } catch (error) {
    console.error('Feedback submission error:', error);

    // Provide user-friendly error messages
    let userMessage = 'Failed to submit feedback. Please try again.';
    const msg = error.response?.data?.detail || error.message || '';

    if (msg.includes('Message is required')) {
      userMessage = 'Please enter a valid feedback message.';
    } else if (msg.includes('Invalid data format')) {
      userMessage = 'Your feedback contains invalid characters. Please remove any special symbols and try again.';
    } else if (msg.includes('rating')) {
      userMessage = 'Please select a valid rating between 1 and 5 stars.';
    } else if (msg.includes('network') || msg.includes('fetch') || msg.includes('offline')) {
      userMessage = 'Network error. Please check your connection and try again.';
    }

    return {
      success: false,
      error: userMessage,
      originalError: msg,
    };
  }
};

/**
 * Get feedback statistics (admin only)
 * @returns {Promise<Object>} - Statistics data
 */
export const getFeedbackStats = async () => {
  try {
    const response = await apiClient.get('/api/feedback/stats', {
      errorContext: 'feedback-stats',
      errorMessage: 'Failed to load feedback statistics.',
    });

    return {
      success: true,
      data: response.data,
    };
  } catch (error) {
    console.error('Feedback stats error:', error);

    return {
      success: false,
      error: 'Failed to load feedback statistics. Please try again later.',
      originalError: error.response?.data?.detail || error.message,
    };
  }
};

/**
 * Validate feedback data client-side before submission
 * @param {Object} data - Feedback data to validate
 * @returns {Object} - Validation result
 */
export const validateFeedbackData = (data) => {
  const errors = {};
  
  // Message validation
  if (!data.message || data.message.trim().length < 3) {
    errors.message = 'Feedback message must be at least 3 characters long.';
  } else if (data.message.length > 2000) {
    errors.message = 'Feedback message is too long (maximum 2000 characters).';
  }
  
  // Rating validation
  if (!data.rating || data.rating < 1 || data.rating > 5) {
    errors.rating = 'Please select a rating between 1 and 5 stars.';
  }
  
  // Name validation (optional)
  if (data.name && data.name.length > 100) {
    errors.name = 'Name is too long (maximum 100 characters).';
  }
  
  // Location validation (optional)
  if (data.location && data.location.length > 200) {
    errors.location = 'Location is too long (maximum 200 characters).';
  }
  
  // Category validation
  const validCategories = ['general', 'feature', 'bug', 'ui', 'accuracy', 'other'];
  if (!validCategories.includes(data.category)) {
    errors.category = 'Invalid feedback category selected.';
  }
  
  // Crop type validation (optional)
  const validCrops = [
    'Rice', 'Wheat', 'Cotton', 'Sugarcane', 'Maize',
    'Soybean', 'Potato', 'Onion', 'Tomato', 'Vegetables',
    'Fruits', 'Other'
  ];
  if (data.cropType && !validCrops.includes(data.cropType)) {
    errors.cropType = 'Invalid crop type selected.';
  }
  
  // Check for potentially dangerous patterns
  const dangerousPatterns = [
    /\$[a-zA-Z_][a-zA-Z0-9_]*\s*:/, // MongoDB operators
    /\{.*\}\s*:\s*\{/, // Nested object injection
    /\.\.\//, // Path traversal
    /<script.*?>.*?<\/script>/i, // XSS
    /on\w+\s*=/i, // Event handlers
    /javascript:/i, // JavaScript protocol
    /data:/i, // Data URLs
  ];
  
  const allFields = [data.message, data.name, data.location].filter(Boolean);
  for (const field of allFields) {
    for (const pattern of dangerousPatterns) {
      if (pattern.test(field)) {
        errors.security = 'Your input contains potentially unsafe characters. Please remove any special symbols and try again.';
        break;
      }
    }
    if (errors.security) break;
  }
  
  return {
    isValid: Object.keys(errors).length === 0,
    errors,
  };
};

/**
 * Sanitize feedback data before submission
 * @param {Object} data - Raw feedback data
 * @returns {Object} - Sanitized feedback data
 */
export const sanitizeFeedbackData = (data) => {
  const sanitized = { ...data };
  
  // Helper function to sanitize strings
  const sanitizeString = (str, maxLength) => {
    if (!str) return str;
    
    // Remove null bytes and control characters
    let sanitizedStr = str.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, '');
    
    // Remove extra whitespace
    sanitizedStr = sanitizedStr.replace(/\s+/g, ' ').trim();
    
    // Truncate to max length
    if (sanitizedStr.length > maxLength) {
      sanitizedStr = sanitizedStr.substring(0, maxLength);
    }
    
    return sanitizedStr;
  };
  
  // Sanitize each field
  if (sanitized.name) {
    sanitized.name = sanitizeString(sanitized.name, 100);
  }
  
  if (sanitized.location) {
    sanitized.location = sanitizeString(sanitized.location, 200);
  }
  
  if (sanitized.message) {
    sanitized.message = sanitizeString(sanitized.message, 2000);
  }
  
  // Ensure rating is integer
  if (sanitized.rating) {
    sanitized.rating = parseInt(sanitized.rating, 10);
    if (isNaN(sanitized.rating) || sanitized.rating < 1) {
      sanitized.rating = 1;
    } else if (sanitized.rating > 5) {
      sanitized.rating = 5;
    }
  }
  
  return sanitized;
};

/**
 * Test the validation API endpoint
 * @returns {Promise<Object>} - Test results
 */
export const testValidationEndpoint = async () => {
  try {
    const response = await apiClient.get('/api/feedback/validate-test', {
      errorContext: 'feedback-validate-test',
      errorMessage: 'Validation test failed.',
    });

    return {
      success: true,
      data: response.data,
    };
  } catch (error) {
    console.error('Validation test error:', error);

    return {
      success: false,
      error: 'Validation test failed. API may not be available.',
      originalError: error.response?.data?.detail || error.message,
    };
  }
};

export default {
  submitFeedback,
  getFeedbackStats,
  validateFeedbackData,
  sanitizeFeedbackData,
  testValidationEndpoint,
};