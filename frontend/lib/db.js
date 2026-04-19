import { openDB } from 'idb';

const dbPromise = openDB('fasal-sathi-db', 1, {
  upgrade(db) {
    if (!db.objectStoreNames.contains('offline-data')) {
      db.createObjectStore('offline-data');
    }
  },
});

export const getOfflineData = async (key) => {
  return (await dbPromise).get('offline-data', key);
};

export const setOfflineData = async (key, val) => {
  return (await dbPromise).put('offline-data', val, key);
};

export const deleteOfflineData = async (key) => {
  return (await dbPromise).delete('offline-data', key);
};

export const clearOfflineData = async () => {
  return (await dbPromise).clear('offline-data');
};
