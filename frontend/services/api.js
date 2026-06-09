import axios from 'axios';

import { useUiStore } from '../stores/uiStore';
import { reportErrorToBackend } from '../utils/errorReporting';
import { auth } from '../lib/firebase';

// ============================================
// Request & Response Validation
// ============================================

/**
 * Schema validator for request payloads
 */
class SchemaValidator {
  constructor() {
    this.schemas = new Map();
  }

  registerSchema(endpoint, schema) {
    this.schemas.set(endpoint, schema);
  }

  validate(endpoint, data) {
    const schema = this.schemas.get(endpoint);
    if (!schema) return { valid: true };

    const errors = [];

    // Validate required fields
    if (schema.required) {
      for (const field of schema.required) {
        if (!(field in data)) {
          errors.push(`Missing required field: ${field}`);
        }
      }
    }

    // Validate field types
    if (schema.fields) {
      for (const [field, fieldSchema] of Object.entries(schema.fields)) {
        if (field in data) {
          const value = data[field];

          // Type validation
          if (fieldSchema.type && typeof value !== fieldSchema.type) {
            errors.push(`Field ${field} must be ${fieldSchema.type}, got ${typeof value}`);
          }

          // Range validation
          if (fieldSchema.min !== undefined && value < fieldSchema.min) {
            errors.push(`Field ${field} must be >= ${fieldSchema.min}`);
          }
          if (fieldSchema.max !== undefined && value > fieldSchema.max) {
            errors.push(`Field ${field} must be <= ${fieldSchema.max}`);
          }

          // Pattern validation
          if (fieldSchema.pattern && !fieldSchema.pattern.test(value)) {
            errors.push(`Field ${field} invalid format`);
          }

          // Enum validation
          if (fieldSchema.enum && !fieldSchema.enum.includes(value)) {
            errors.push(`Field ${field} must be one of: ${fieldSchema.enum.join(', ')}`);
          }
        }
      }
    }

    return {
      valid: errors.length === 0,
      errors
    };
  }
}

/**
 * Response schema validator
 */
class ResponseValidator {
  constructor() {
    this.expectedSchemas = new Map();
  }

  registerExpected(endpoint, schema) {
    this.expectedSchemas.set(endpoint, schema);
  }

  validate(endpoint, response) {
    const schema = this.expectedSchemas.get(endpoint);
    if (!schema) return { valid: true };

    const errors = [];

    // Check required fields in response
    if (schema.required) {
      for (const field of schema.required) {
        if (!(field in response)) {
          errors.push(`Missing field in response: ${field}`);
        }
      }
    }

    // Check field types
    if (schema.fields) {
      for (const [field, fieldSchema] of Object.entries(schema.fields)) {
        if (field in response) {
          const value = response[field];
          if (fieldSchema.type && typeof value !== fieldSchema.type) {
            errors.push(`Response field ${field} has wrong type`);
          }
        }
      }
    }

    return {
      valid: errors.length === 0,
      errors
    };
  }
}

/**
 * Retry logic with exponential backoff
 */
class RetryStrategy {
  constructor(maxRetries = 3, baseDelay = 100, maxDelay = 5000) {
    this.maxRetries = maxRetries;
    this.baseDelay = baseDelay;
    this.maxDelay = maxDelay;
  }

  shouldRetry(error, attempt) {
    // Don't retry 4xx errors (except 429)
    if (error.response?.status >= 400 && error.response?.status < 500 && error.response?.status !== 429) {
      return false;
    }

    // Retry on 5xx and network errors
    return attempt < this.maxRetries;
  }

  getDelay(attempt) {
    const delay = this.baseDelay * Math.pow(2, attempt);
    return Math.min(delay, this.maxDelay);
  }
}

/**
 * Rate limit handler
 */
class RateLimitHandler {
  constructor() {
    this.rateLimits = new Map();
  }

  handleRateLimit(endpoint, retryAfter) {
    const delayMs = (retryAfter || 60) * 1000;
    const resetTime = Date.now() + delayMs;

    this.rateLimits.set(endpoint, {
      blockedUntil: resetTime,
      retryAfter: delayMs
    });
  }

  isRateLimited(endpoint) {
    const limit = this.rateLimits.get(endpoint);
    if (!limit) return false;

    const now = Date.now();
    if (now >= limit.blockedUntil) {
      this.rateLimits.delete(endpoint);
      return false;
    }

    return true;
  }

  getRemainingWait(endpoint) {
    const limit = this.rateLimits.get(endpoint);
    if (!limit) return 0;

    return Math.max(0, limit.blockedUntil - Date.now());
  }
}

const schemaValidator = new SchemaValidator();
const responseValidator = new ResponseValidator();
const retryStrategy = new RetryStrategy();
const rateLimitHandler = new RateLimitHandler();

// ============================================
// Original API Client Code
// ============================================

const toNumberOr = (value, fallback) => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
};

const API_TIMEOUT_MS = toNumberOr(import.meta.env.VITE_API_TIMEOUT_MS, 15000);
const DEFAULT_RETRIES = toNumberOr(import.meta.env.VITE_API_RETRIES, 2);
const RETRY_BASE_DELAY_MS = toNumberOr(import.meta.env.VITE_API_RETRY_DELAY_MS, 400);
const RETRY_BACKOFF_MULTIPLIER = toNumberOr(import.meta.env.VITE_API_RETRY_MULTIPLIER, 2);
const DEFAULT_CIRCUIT_BREAKER_FAILURES = toNumberOr(import.meta.env.VITE_API_CIRCUIT_BREAKER_FAILURES, 3);
const DEFAULT_CIRCUIT_BREAKER_RESET_MS = toNumberOr(import.meta.env.VITE_API_CIRCUIT_BREAKER_RESET_MS, 15000);

const isErrorLoggingEndpoint = (url) => String(url || '').includes('/api/log-error');
const inFlightRequests = new Map();
const circuitBreakers = new Map();

const isIdempotentMethod = (method) => ['get', 'head', 'options', 'put', 'delete'].includes(method);

const createRequestId = () => {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `req-${Date.now()}-${Math.random().toString(36).slice(2)}`;
};

const sortObjectKeys = (value) => {
  if (Array.isArray(value)) {
    return value.map(sortObjectKeys);
  }

  if (value && typeof value === 'object') {
    return Object.keys(value)
      .sort()
      .reduce((accumulator, key) => {
        accumulator[key] = sortObjectKeys(value[key]);
        return accumulator;
      }, {});
  }

  return value;
};

const stableStringify = (value) => {
  if (value === undefined) {
    return '';
  }

  try {
    return JSON.stringify(sortObjectKeys(value));
  } catch {
    return String(value);
  }
};

const getRequestDedupKey = (method, url, config) => {
  const dedupeKey = config.dedupeKey || config.requestKey;
  if (dedupeKey) {
    return String(dedupeKey);
  }

  return [
    method,
    String(url || ''),
    stableStringify(config.params),
    stableStringify(config.data),
  ].join('|');
};

const shouldDeduplicateRequest = (method, config) => {
  if (config.dedupe === false || config.skipRequestDeduplication) {
    return false;
  }

  if (isIdempotentMethod(method)) {
    return true;
  }

  return Boolean(config.dedupeNonIdempotent || config.headers?.['X-Idempotency-Key']);
};

const getCircuitBreakerKey = (method, url, config) => {
  if (config.circuitBreakerKey) {
    return String(config.circuitBreakerKey);
  }

  return `${method}:${String(url || '')}`;
};

const getCircuitBreakerState = (key) => {
  if (!circuitBreakers.has(key)) {
    circuitBreakers.set(key, {
      failures: 0,
      state: 'closed',
      openedAt: 0,
      halfOpenProbeActive: false,
    });
  }

  return circuitBreakers.get(key);
};

const createCircuitBreakerError = (key, resetMs) => {
  const error = new Error(`Circuit breaker open for ${key}`);
  error.code = 'ERR_CIRCUIT_BREAKER_OPEN';
  error.status = 503;
  error.response = {
    status: 503,
    statusText: 'Service Unavailable',
    data: {
      message: 'Service temporarily unavailable. Please try again shortly.',
      circuitBreakerKey: key,
      retryAfterMs: resetMs,
    },
  };
  return error;
};

const updateCircuitBreakerOnSuccess = (state) => {
  state.failures = 0;
  state.state = 'closed';
  state.openedAt = 0;
  state.halfOpenProbeActive = false;
};

const updateCircuitBreakerOnFailure = (state, key, failureThreshold) => {
  state.failures += 1;

  if (state.state === 'half-open') {
    state.state = 'open';
    state.openedAt = Date.now();
    state.halfOpenProbeActive = false;
    return;
  }

  if (state.failures >= failureThreshold) {
    state.state = 'open';
    state.openedAt = Date.now();
    state.halfOpenProbeActive = false;
  }
};

const resolveCircuitBreakerGate = (state, key, resetMs) => {
  if (state.state !== 'open') {
    return { allowed: true };
  }

  const elapsed = Date.now() - state.openedAt;
  if (elapsed < resetMs) {
    return { allowed: false, error: createCircuitBreakerError(key, resetMs - elapsed) };
  }

  if (state.halfOpenProbeActive) {
    return { allowed: false, error: createCircuitBreakerError(key, 0) };
  }

  state.state = 'half-open';
  state.halfOpenProbeActive = true;
  return { allowed: true };
};

const canRetryRequest = (error, config) => {
  // Prevent automatic retries on non-idempotent HTTP methods (like POST)
  // unless they have an idempotency key to prevent duplicate records.
  const method = (config.method || 'get').toLowerCase();
  const isIdempotent = ['get', 'head', 'options', 'put', 'delete'].includes(method);
  const hasIdempotencyKey = !!config.headers?.['X-Idempotency-Key'];
  
  if (!isIdempotent && !config.retryNonIdempotent && !hasIdempotencyKey) {
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

const getRetryDelayMs = (retryCount, retryDelayMs, retryMultiplier) => {
  const baseDelay =
    typeof retryDelayMs === 'number' ? retryDelayMs : RETRY_BASE_DELAY_MS;
  const multiplier =
    typeof retryMultiplier === 'number' && retryMultiplier > 0
      ? retryMultiplier
      : RETRY_BACKOFF_MULTIPLIER;
  return baseDelay * Math.pow(multiplier, retryCount);
};

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const normalizeBaseUrl = (value) => {
  if (!value) {
    return '';
  }

  return String(value).replace(/\/$/, '');
};

const resolveApiBaseUrl = () => {
  const configuredBaseUrl = normalizeBaseUrl(
    import.meta.env.VITE_API_BASE_URL || import.meta.env.VITE_BACKEND_URL
  );

  if (configuredBaseUrl) {
    return configuredBaseUrl;
  }

  // Local development relies on the Vite proxy; production deployments need
  // an explicit backend URL so requests do not stay on the static frontend host.
  if (typeof window !== 'undefined') {
    const hostname = window.location.hostname;
    const isLocalhost =
      hostname === 'localhost' ||
      hostname === '127.0.0.1' ||
      hostname.endsWith('.localhost');

    if (!isLocalhost) {
      console.error(
        '[api.js] VITE_API_BASE_URL is not configured in production. ' +
        'API requests will fail. Set VITE_API_BASE_URL to your backend origin.'
      );
    }
    return '';
  }

  return '';
};

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

const axiosClient = axios.create({
  baseURL: resolveApiBaseUrl(),
  timeout: API_TIMEOUT_MS,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Axios request interceptors support returning a Promise, so the async
// token fetch integrates cleanly without any additional wrapper.
axiosClient.interceptors.request.use(
  async (config) => {
    const nextConfig = { ...config };
    const method = (nextConfig.method || 'get').toLowerCase();
    const requestId = nextConfig.requestId || nextConfig.headers?.['X-Request-ID'] || createRequestId();
    const currentUserId = auth?.currentUser?.uid || 'anonymous';

    nextConfig.requestId = requestId;
    nextConfig.userId = nextConfig.userId || currentUserId;
    nextConfig.errorContext = {
      ...(typeof nextConfig.errorContext === 'object' ? nextConfig.errorContext : { label: nextConfig.errorContext }),
      method,
      url: nextConfig.url,
      requestId,
      userId: nextConfig.userId,
    };

    nextConfig.headers = {
      ...nextConfig.headers,
      'X-Request-ID': requestId,
    };

    // Automatically attach idempotency keys to non-idempotent requests (POST)
    // to allow safe retries and prevent duplicate records on the backend.
    if (method === 'post' && !nextConfig.headers?.['X-Idempotency-Key']) {
      // Generate a unique ID for this request session
      const idempotencyKey = typeof crypto !== 'undefined' && crypto.randomUUID 
        ? crypto.randomUUID() 
        : `idemp-${Date.now()}-${Math.random().toString(36).slice(2)}`;
        
      nextConfig.headers = {
        ...nextConfig.headers,
        'X-Idempotency-Key': idempotencyKey,
      };
    }

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

axiosClient.interceptors.response.use(
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
      const retryDelay = getRetryDelayMs(retryCount, config.retryDelayMs, config.retryMultiplier);

      // Pause execution for the calculated delay duration
      await wait(retryDelay);

      // Re-issue the request with the updated configuration
      return axiosClient(config);
    }

    if (config.logError !== false && !isErrorLoggingEndpoint(config.url)) {
      reportErrorToBackend({
        error,
        context: config.errorContext || 'api-client',
        requestId: config.requestId,
        userId: config.userId,
        timestamp: new Date().toISOString(),
      });
    }

    // NOTE: UI feedback (like toast.error) is intentionally omitted here.
    // Errors are propagated so that the specific component or hook making
    // the request can handle them and provide context-aware feedback to the user.
    return Promise.reject(error);
  }
);

const request = (method, url, data = undefined, config = {}) => {
  const normalizedMethod = method.toLowerCase();
  const dedupeEnabled = shouldDeduplicateRequest(normalizedMethod, config);
  const dedupeKey = dedupeEnabled ? getRequestDedupKey(normalizedMethod, url, config) : null;
  const circuitBreakerKey = getCircuitBreakerKey(normalizedMethod, url, config);
  const breakerState = getCircuitBreakerState(circuitBreakerKey);
  const circuitBreakerThreshold = toNumberOr(config.circuitBreakerThreshold, DEFAULT_CIRCUIT_BREAKER_FAILURES);
  const circuitBreakerResetMs = toNumberOr(config.circuitBreakerResetMs, DEFAULT_CIRCUIT_BREAKER_RESET_MS);

  if (dedupeKey && inFlightRequests.has(dedupeKey)) {
    return inFlightRequests.get(dedupeKey);
  }

  const executeRequest = async () => {
    const gate = resolveCircuitBreakerGate(breakerState, circuitBreakerKey, circuitBreakerResetMs);
    if (!gate.allowed) {
      return Promise.reject(gate.error);
    }

    try {
      const requestConfig = {
        ...config,
        method: normalizedMethod,
        url,
      };

      if (data !== undefined) {
        requestConfig.data = data;
      }

      const response = await axiosClient.request(requestConfig);
      updateCircuitBreakerOnSuccess(breakerState);
      return response;
    } catch (error) {
      updateCircuitBreakerOnFailure(breakerState, circuitBreakerKey, circuitBreakerThreshold);
      throw error;
    } finally {
      if (dedupeKey) {
        inFlightRequests.delete(dedupeKey);
      }
    }
  };

  const pendingRequest = executeRequest();

  if (dedupeKey) {
    inFlightRequests.set(dedupeKey, pendingRequest);
  }

  return pendingRequest;
};

const apiClient = {
  request: (config = {}) => request(config.method || 'get', config.url, config.data, config),
  get: (url, config = {}) => request('get', url, undefined, config),
  post: (url, data = {}, config = {}) => request('post', url, data, config),
  put: (url, data = {}, config = {}) => request('put', url, data, config),
  patch: (url, data = {}, config = {}) => request('patch', url, data, config),
  delete: (url, config = {}) => request('delete', url, config.data, config),
};

export default apiClient;
// Enhanced API validation
