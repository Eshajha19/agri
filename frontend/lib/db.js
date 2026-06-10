import { openDB } from 'idb';

const dbPromise = openDB('fasal-sathi-db', 2, {
  upgrade(db, oldVersion) {
    if (oldVersion < 1 && !db.objectStoreNames.contains('offline-data')) {
      db.createObjectStore('offline-data');
    }
    if (oldVersion < 2 && !db.objectStoreNames.contains('offline-requests')) {
      db.createObjectStore('offline-requests', { keyPath: 'id', autoIncrement: true });
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

export const clearOfflineRequests = async () => {
  return (await dbPromise).clear('offline-requests');
};

export const saveOfflineSnapshot = async (key, value) => {
  return setOfflineData(key, {
    value,
    updatedAt: new Date().toISOString(),
  });
};

export const getOfflineSnapshot = async (key) => {
  const record = await getOfflineData(key);
  return record && typeof record === 'object' && 'value' in record ? record.value : record;
};

export const deleteOfflineSnapshot = async (key) => {
  return deleteOfflineData(key);
};

export const addOfflineRequest = async (requestConfig) => {
  return (await dbPromise).add('offline-requests', requestConfig);
};

export const getOfflineRequests = async () => {
  return (await dbPromise).getAll('offline-requests');
};

export const deleteOfflineRequest = async (id) => {
  return (await dbPromise).delete('offline-requests', id);
};
