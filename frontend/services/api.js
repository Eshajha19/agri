import axios from 'axios';
import toast from 'react-hot-toast';

import { useUiStore } from '../stores/uiStore';
import { formatErrorMessage, reportErrorToBackend } from '../utils/errorReporting';

const toNumberOr = (value, fallback) => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
};

const API_TIMEOUT_MS = toNumberOr(import.meta.env.VITE_API_TIMEOUT_MS, 15000);
const DEFAULT_RETRIES = toNumberOr(import.meta.env.VITE_API_RETRIES, 2);
const RETRY_BASE_DELAY_MS = toNumberOr(import.meta.env.VITE_API_RETRY_DELAY_MS, 400);

const isErrorLoggingEndpoint = (url) => String(url || '').includes('/api/log-error');

const canRetryRequest = (error, config) => {
  const retries =
    typeof config.retries === 'number' ? config.retries : DEFAULT_RETRIES;
  const retryCount = config.__retryCount || 0;

  if (retryCount >= retries) {
    return false;
  }

  const status = error?.response?.status;

  // Retry transient failures only.
  return !status || status === 408 || status === 429 || status >= 500;
};

const getRetryDelayMs = (retryCount, retryDelayMs) => {
  const baseDelay =
    typeof retryDelayMs === 'number' ? retryDelayMs : RETRY_BASE_DELAY_MS;
  return baseDelay * Math.pow(2, retryCount);
};

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '/',
  timeout: API_TIMEOUT_MS,
  headers: {
    'Content-Type': 'application/json',
  },
});

apiClient.interceptors.request.use(
  (config) => {
    const nextConfig = { ...config };
    const token =
      localStorage.getItem('authToken') || localStorage.getItem('token');

    if (token && !nextConfig.headers?.Authorization) {
      nextConfig.headers = {
        ...nextConfig.headers,
        Authorization: `Bearer ${token}`,
      };
    }

    if (!nextConfig.skipGlobalLoader) {
      useUiStore.getState().incrementApiPendingRequests();
    }

    return nextConfig;
  },
  (requestError) => Promise.reject(requestError)
);

apiClient.interceptors.response.use(
  (response) => {
    if (!response.config.skipGlobalLoader) {
      useUiStore.getState().decrementApiPendingRequests();
    }

    return response;
  },
  async (error) => {
    const config = error.config || {};

    if (!config.skipGlobalLoader) {
      useUiStore.getState().decrementApiPendingRequests();
    }

    if (canRetryRequest(error, config)) {
      const retryCount = config.__retryCount || 0;
      config.__retryCount = retryCount + 1;
      const retryDelay = getRetryDelayMs(retryCount, config.retryDelayMs);
      await wait(retryDelay);
      return apiClient(config);
    }

    if (!config.suppressToast) {
      const message = config.errorMessage || formatErrorMessage(error);
      toast.error(message, {
        duration: 4000,
        position: 'top-right',
      });
    }

    if (config.logError !== false && !isErrorLoggingEndpoint(config.url)) {
      reportErrorToBackend({
        error,
        context: config.errorContext || 'api-client',
        timestamp: new Date().toISOString(),
      });
    }

    return Promise.reject(error);
  }
);

export default apiClient;
