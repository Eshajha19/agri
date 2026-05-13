/**
 * Unified HTTP client — single entry point for all API calls.
 *
 * Architecture
 * ------------
 * Previously the codebase had two separate HTTP clients:
 *
 *   • services/api.js  — Axios-based, with auth injection, retries,
 *                        global loader, and error reporting.
 *   • lib/apiClient.js — Native-fetch wrapper with offline queue support
 *                        but no auth, retries, or interceptors.
 *
 * This module unifies both by:
 *   1. Delegating all **online** requests to the Axios client in
 *      services/api.js so every request automatically gets auth
 *      injection, exponential-backoff retries, global loader tracking,
 *      and backend error reporting.
 *   2. Preserving the **offline queue** behaviour for mutating requests
 *      (POST / PUT / PATCH / DELETE) when the browser is offline, so
 *      those requests are persisted to IndexedDB and replayed later.
 *
 * Usage
 * -----
 * Import this module wherever you need to make an HTTP request:
 *
 *   import apiClient from '../lib/apiClient';
 *
 *   // GET — always goes through Axios (or throws if offline)
 *   const res = await apiClient.get('/api/data');
 *
 *   // POST — queued to IndexedDB when offline, sent via Axios when online
 *   const res = await apiClient.post('/api/feedback', payload);
 *
 * The returned value is the Axios response object (online) or a
 * synthetic offline-queue response (offline mutation):
 *   { data: { success: true, message: 'Request queued offline.' }, offlineQueue: true }
 */

import axiosClient from '../services/api';
import { addOfflineRequest } from './db';

const MUTATING_METHODS = new Set(['post', 'put', 'patch', 'delete']);

/**
 * Core dispatcher.
 *
 * @param {string} method  - HTTP method (lowercase)
 * @param {string} url     - Request URL or path
 * @param {*}      data    - Request body (for mutating methods)
 * @param {object} config  - Axios request config overrides
 * @returns {Promise}
 */
async function request(method, url, data = undefined, config = {}) {
  const isMutation = MUTATING_METHODS.has(method.toLowerCase());

  // ── Offline path: queue mutations, reject reads ──────────────────────────
  if (!navigator.onLine) {
    if (isMutation) {
      console.log('[apiClient] Offline — queuing request to IndexedDB:', method.toUpperCase(), url);
      await addOfflineRequest({
        url,
        method: method.toUpperCase(),
        body: data !== undefined ? JSON.stringify(data) : undefined,
        headers: { 'Content-Type': 'application/json', ...config.headers },
        queuedAt: new Date().toISOString(),
      });
      // Return a synthetic response so callers do not crash.
      return {
        data: { success: true, message: 'Request queued offline.' },
        offlineQueue: true,
        status: 202,
        statusText: 'Queued',
      };
    }

    // GET requests while offline: let the service worker cache handle it,
    // or propagate the error so the caller can show an appropriate message.
    throw new Error('You are currently offline. Please connect to the internet to fetch this data.');
  }

  // ── Online path: delegate to Axios client ────────────────────────────────
  // Axios method signatures:
  //   axiosClient.get(url, config)
  //   axiosClient.post(url, data, config)
  //   axiosClient.put(url, data, config)
  //   axiosClient.patch(url, data, config)
  //   axiosClient.delete(url, config)
  if (isMutation && method.toLowerCase() !== 'delete') {
    return axiosClient[method.toLowerCase()](url, data, config);
  }
  // GET, HEAD, OPTIONS, DELETE — data goes into config, not as body arg
  const mergedConfig = data !== undefined ? { ...config, data } : config;
  return axiosClient[method.toLowerCase()](url, mergedConfig);
}

const apiClient = {
  get:    (url, config = {})             => request('get',    url, undefined, config),
  post:   (url, data = {}, config = {})  => request('post',   url, data,      config),
  put:    (url, data = {}, config = {})  => request('put',    url, data,      config),
  patch:  (url, data = {}, config = {})  => request('patch',  url, data,      config),
  delete: (url, config = {})             => request('delete', url, undefined, config),
};

export default apiClient;
