import axios from 'axios';
import { reportErrorToBackend } from '../utils/errorReporting';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

const FALLBACK_NEWS_RESPONSE = {
  success: true,
  source: 'fallback',
  articles: [],
  page: 1,
  total: 0,
};

const SOURCE_HEALTH = {
  failures: 0,
  lastFailure: null,
  lastSuccess: null,
};

const MAX_CONSECUTIVE_FAILURES = 3;

/**
 * Fetch farming news articles with optional filtering, sorting, and pagination
 * @param {number} page - Page number (starts at 1)
 * @param {number} pageSize - Number of articles per page
 * @param {string} category - Optional category filter
 * @param {string} search - Optional search text
 * @param {string} sortBy - Sort order: 'latest' | 'relevant'
 * @returns {Promise} News list response with articles and pagination info
 */
export async function fetchFarmingNews(
  page = 1,
  pageSize = 10,
  category = null,
  search = null,
  sortBy = 'latest'
) {
  try {
    const params = new URLSearchParams();
    params.append('page', page);
    params.append('page_size', pageSize);

    if (category) {
      params.append('category', category);
    }

    if (search) {
      params.append('search', search);
    }

    if (sortBy) {
      params.append('sort_by', sortBy);
    }

    const response = await performNewsRequest(
      `${API_BASE}/api/farming-news?${params.toString()}`
    );

    if (!response?.data) {
      console.warn('[NEWS_FALLBACK] Empty API response received');

      return {
        ...FALLBACK_NEWS_RESPONSE,
        page,
        degraded: true,
        sourceFailures: SOURCE_HEALTH.failures,
      };
    }

    return {
      ...response.data,
      source: 'api',
    };
  } catch (error) {
    console.error('Error fetching farming news:', error);

    reportErrorToBackend({
      message: 'Failed to fetch farming news',
      source: 'newsApi.js',
      stack: error.stack,
      level: 'error',
    });

    console.warn('[NEWS_FALLBACK] Serving fallback content');

    return {
      ...FALLBACK_NEWS_RESPONSE,
      page,
      degraded: true,
      sourceFailures: SOURCE_HEALTH.failures,
      lastFailure: SOURCE_HEALTH.lastFailure,
    };
  }
}

async function performNewsRequest(url) {
  try {
    const response = await axios.get(url, {
      timeout: 15000,
    });

    SOURCE_HEALTH.failures = 0;
    SOURCE_HEALTH.lastSuccess = Date.now();

    return response;
  } catch (error) {
    SOURCE_HEALTH.failures += 1;
    SOURCE_HEALTH.lastFailure = Date.now();

    throw error;
  }
}

/**
 * Fetch featured/important news articles
 * @param {number} limit - Number of featured articles to return
 * @returns {Promise} Featured articles list
 */
export async function fetchFeaturedNews(limit = 3) {
  try {
    const response = await performNewsRequest(
      `${API_BASE}/api/farming-news/featured?limit=${limit}`
    );

    if (!response?.data) {
      console.warn('[NEWS_FALLBACK] Empty featured response received');

      return {
        success: true,
        source: 'fallback',
        articles: [],
        degraded: true,
        sourceFailures: SOURCE_HEALTH.failures,
        lastFailure: SOURCE_HEALTH.lastFailure,
      };
    }

    return {
      ...response.data,
      source: 'api',
    };
  } catch (error) {
    console.error('Error fetching featured news:', error);

    console.warn('[NEWS_FALLBACK] Serving fallback featured content');

    return {
      success: true,
      source: 'fallback',
      articles: [],
    };
  }
}

/**
 * Get list of available news categories
 * @returns {Array} Array of category strings
 */
export function getNewsCategories() {
  return [
    'All',
    'Weather',
    'Government Schemes',
    'Crop Management',
    'Technology',
    'Insurance',
    'Organic Farming',
    'Market Prices',
    'Soil Management'
  ];
}

/**
 * Format date string to readable relative format
 * @param {string} dateStr - ISO date string
 * @returns {string} Formatted date
 */
export function formatNewsDate(dateStr) {
  try {
    const date = new Date(dateStr);
    const now = new Date();
    const diffTime = now - date;
    const diffMinutes = Math.floor(diffTime / (1000 * 60));
    const diffHours = Math.floor(diffTime / (1000 * 60 * 60));
    const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24));

    if (diffMinutes < 1) {
      return 'Just now';
    } else if (diffMinutes < 60) {
      return `${diffMinutes}m ago`;
    } else if (diffHours < 24) {
      return `${diffHours}h ago`;
    } else if (diffDays === 1) {
      return 'Yesterday';
    } else if (diffDays < 7) {
      return `${diffDays} days ago`;
    } else if (diffDays < 30) {
      const weeks = Math.floor(diffDays / 7);
      return `${weeks} week${weeks > 1 ? 's' : ''} ago`;
    }

    return date.toLocaleDateString('en-IN', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  } catch (error) {
    return dateStr;
  }
}

export function getNewsSourceHealth() {
  return {
    failures: SOURCE_HEALTH.failures,
    lastFailure: SOURCE_HEALTH.lastFailure,
    lastSuccess: SOURCE_HEALTH.lastSuccess,
    degraded:
      SOURCE_HEALTH.failures >=
      MAX_CONSECUTIVE_FAILURES,
  };
}