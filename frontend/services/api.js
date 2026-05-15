import axios from 'axios';

import { useUiStore } from '../stores/uiStore';
import { reportErrorToBackend } from '../utils/errorReporting';
import { auth } from '../lib/firebase';

const toNumberOr = (value, fallback) => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
};

const API_TIMEOUT_MS = toNumberOr(import.meta.env.VITE_API_TIMEOUT_MS, 15000);
const DEFAULT_RETRIES = toNumberOr(import.meta.env.VITE_API_RETRIES, 2);
const RETRY_BASE_DELAY_MS = toNumberOr(import.meta.env.VITE_API_RETRY_DELAY_MS, 400);

const isErrorLoggingEndpoint = (url) => String(url || '').includes('/api/log-error');

const canRetryRequest = (error, config) => {
  // Prevent automatic retries on non-idempotent HTTP methods (like POST)
  // to avoid side-effects such as creating duplicate database entries
  // (e.g. submitting the same feedback multiple times on a timeout).
  const method = (config.method || 'get').toLowerCase();
  const isIdempotent = ['get', 'head', 'options', 'put', 'delete'].includes(method);
  
  if (!isIdempotent && !config.retryNonIdempotent) {
    return false;
  }

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

/**
 * Retrieve the current Firebase ID token for the signed-in user.
 *
 * Why not localStorage?
 * ---------------------
 * Firebase Authentication manages ID tokens internally through the SDK.
 * It does NOT persist them to localStorage under any predictable key, so
 * reading localStorage.getItem('agri:authToken') always returns null for
 * Firebase-authenticated users, causing every protected request to be sent
 * without an Authorization header.
 *
 * auth.currentUser.getIdToken(false) returns the cached token when it is
 * still valid (< 1 hour old) and automatically fetches a fresh one when it
 * has expired — giving us transparent token refresh at zero extra cost.
 *
 * Returns null when:
 *   - No user is signed in (anonymous or unauthenticated)
 *   - Firebase is not configured (missing env vars)
 *   - The token fetch fails for any reason (network error, revoked token)
 *
 * In all null cases the request proceeds without an Authorization header,
 * which is the correct behaviour for public endpoints.
 */
async function getFirebaseIdToken() {
  try {
    if (auth?.currentUser) {
      // Pass false to use the cached token; Firebase refreshes automatically
      // when the token is within 5 minutes of expiry.
      return await auth.currentUser.getIdToken(false);
    }
  } catch (err) {
    // Log but do not throw — unauthenticated requests are valid for public routes.
    console.warn('[api] Could not retrieve Firebase ID token:', err?.message);
  }
  return null;
}

const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '/',
  timeout: API_TIMEOUT_MS,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Axios request interceptors support returning a Promise, so the async
// token fetch integrates cleanly without any additional wrapper.
apiClient.interceptors.request.use(
  async (config) => {
    const nextConfig = { ...config };

    // Only inject the token when the caller has not already set one.
    if (!nextConfig.headers?.Authorization) {
      const token = await getFirebaseIdToken();
      if (token) {
        nextConfig.headers = {
          ...nextConfig.headers,
          Authorization: `Bearer ${token}`,
        };
      }
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
      // Increment the retry count to track attempts
      const retryCount = config.__retryCount || 0;
      config.__retryCount = retryCount + 1;

      // Enforce a strict timeout on retries to prevent indefinite waiting
      // if a server connection drops without a proper response
      config.timeout = 10000;

      // Calculate the exponential backoff delay based on the retry count
      const retryDelay = getRetryDelayMs(retryCount, config.retryDelayMs);

      // Pause execution for the calculated delay duration
      await wait(retryDelay);

      // Re-issue the request with the updated configuration
      return apiClient(config);
    }

    if (config.logError !== false && !isErrorLoggingEndpoint(config.url)) {
      reportErrorToBackend({
        error,
        context: config.errorContext || 'api-client',
        timestamp: new Date().toISOString(),
      });
    }

    // NOTE: UI feedback (like toast.error) is intentionally omitted here.
    // Errors are propagated so that the specific component or hook making
    // the request can handle them and provide context-aware feedback to the user.
    return Promise.reject(error);
  }
);

export default apiClient;
