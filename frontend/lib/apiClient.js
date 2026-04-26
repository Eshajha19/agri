import { addOfflineRequest } from './db';

/**
 * A wrapper around native fetch that supports offline queueing.
 * Only use this for requests that mutate data or should be sent
 * eventually (e.g. POST, PUT, DELETE). GET requests will just fail
 * or be served from the service worker cache.
 */
export const offlineFetch = async (url, options = {}) => {
  if (!navigator.onLine) {
    if (options.method && options.method !== 'GET') {
      console.log('User offline. Queueing request to IDB for later sync:', url);
      await addOfflineRequest({ url, ...options });
      return { 
        ok: true, 
        offlineQueue: true, 
        json: async () => ({ success: true, message: 'Request queued offline.'}) 
      };
    } else {
      throw new Error('You are currently offline. Please connect to internet to fetch this data.');
    }
  }

  return fetch(url, options);
};
