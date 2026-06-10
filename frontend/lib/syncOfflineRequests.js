import { getOfflineRequests, deleteOfflineRequest } from './db.js';

const normalizeRequestBody = (body) => {
  if (body == null) return undefined;
  if (typeof body === 'string') return body;
  if (body instanceof Blob || body instanceof FormData || body instanceof URLSearchParams) {
    return body;
  }
  return JSON.stringify(body);
};

export const syncOfflineRequests = async () => {
  if (!navigator.onLine) {
    return { attempted: 0, synced: 0, failed: 0, remaining: 0 };
  }

  try {
    const requests = await getOfflineRequests();
    if (requests.length === 0) {
      return { attempted: 0, synced: 0, failed: 0, remaining: 0 };
    }

    console.log(`Syncing ${requests.length} offline requests...`);
    const queuedRequests = [...requests].sort((left, right) => {
      const leftQueuedAt = left.queuedAt ? new Date(left.queuedAt).getTime() : 0;
      const rightQueuedAt = right.queuedAt ? new Date(right.queuedAt).getTime() : 0;
      return leftQueuedAt - rightQueuedAt;
    });

    let syncedCount = 0;
    let failedCount = 0;

    for (const req of queuedRequests) {
      try {
        const response = await fetch(req.url, {
          method: req.method || 'POST',
          headers: req.headers || {
            'Content-Type': 'application/json'
          },
          body: normalizeRequestBody(req.body),
        });

        if (response.ok) {
          await deleteOfflineRequest(req.id);
          syncedCount += 1;
          console.log(`Successfully synced request ID: ${req.id}`);
        } else {
          failedCount += 1;
          console.warn(`Failed to sync request ID: ${req.id}, status: ${response.status}`);
        }
      } catch (error) {
        failedCount += 1;
        console.error(`Error syncing request ID: ${req.id}`, error);
        // We do not delete from IndexedDB if the network request fails, we'll try again later
      }
    }

    return {
      attempted: queuedRequests.length,
      synced: syncedCount,
      failed: failedCount,
      remaining: queuedRequests.length - syncedCount,
    };
  } catch (error) {
    console.error('Error fetching offline requests from IDB:', error);
    return { attempted: 0, synced: 0, failed: 0, remaining: 0 };
  }
};
