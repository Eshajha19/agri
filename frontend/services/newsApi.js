import axios from 'axios';
import { reportErrorToBackend } from '../utils/errorReporting';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

/**
 * Fetch farming news articles with optional filtering and pagination
 * @param {number} page - Page number (starts at 1)
 * @param {number} pageSize - Number of articles per page
 * @param {string} category - Optional category filter
 * @param {string} search - Optional search text
 * @returns {Promise} News list response with articles and pagination info
 */
export async function fetchFarmingNews(page = 1, pageSize = 10, category = null, search = null) {
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

    const response = await axios.get(`${API_BASE}/api/farming-news?${params.toString()}`, {
      timeout: 15000,
    });

    return response.data;
  } catch (error) {
    console.error('Error fetching farming news:', error);
    reportErrorToBackend({
      message: 'Failed to fetch farming news',
      source: 'newsApi.js',
      stack: error.stack,
      level: 'error'
    });
    throw error;
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
 * Format date string to readable format
 * @param {string} dateStr - ISO date string
 * @returns {string} Formatted date
 */
export function formatNewsDate(dateStr) {
  try {
    const date = new Date(dateStr);
    const now = new Date();
    const diffTime = now - date;
    const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24));

    if (diffDays === 0) {
      return 'Today';
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
