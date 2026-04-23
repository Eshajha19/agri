import { getOfflineRequests, deleteOfflineRequest } from './db.js';

export const syncOfflineRequests = async () => {
  if (!navigator.onLine) return; // Only sync if online

  try {
    const requests = await getOfflineRequests();
    if (requests.length === 0) return;

    console.log(`Syncing ${requests.length} offline requests...`);

    for (const req of requests) {
      try {
        const response = await fetch(req.url, {
          method: req.method || 'POST',
          headers: req.headers || {
            'Content-Type': 'application/json'
          },
          body: req.body ? JSON.stringify(req.body) : undefined,
        });

        if (response.ok) {
          await deleteOfflineRequest(req.id);
          console.log(`Successfully synced request ID: ${req.id}`);
        } else {
          console.warn(`Failed to sync request ID: ${req.id}, status: ${response.status}`);
        }
      } catch (error) {
        console.error(`Error syncing request ID: ${req.id}`, error);
        // We do not delete from IndexedDB if the network request fails, we'll try again later
      }
    }
  } catch (error) {
    console.error('Error fetching offline requests from IDB:', error);
  }
};
