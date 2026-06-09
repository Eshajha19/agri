import { create } from 'zustand';
import { openDB } from 'idb';

const DB_NAME = 'fasal-saathi-offline';
const DB_VERSION = 1;
const STORE_NAME = 'farmIntelligenceQueue';

const getDeviceId = () => {
  let id = localStorage.getItem('fs_device_id');
  if (!id) {
    id = crypto.randomUUID?.() || `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    localStorage.setItem('fs_device_id', id);
  }
  return id;
};

const initDB = async () => {
  return openDB(DB_NAME, DB_VERSION, {
    upgrade(db) {
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        const store = db.createObjectStore(STORE_NAME, { keyPath: 'localId', autoIncrement: true });
        store.createIndex('by_synced', 'synced', { unique: false });
        store.createIndex('by_uid', 'uid', { unique: false });
      }
    },
  });
};

export const useOfflineStore = create((set, get) => ({
  pendingCount: 0,
  isOnline: navigator.onLine,
  lastSyncAt: null,
  isSyncing: false,

  init: async () => {
    const db = await initDB();
    const tx = db.transaction(STORE_NAME, 'readonly');
    const store = tx.objectStore(STORE_NAME);
    const all = await store.getAll();
    set({ 
      pendingCount: all.filter(i => !i.synced).length,
      isOnline: navigator.onLine,
    });
    
    const handleOnline = () => {
      set({ isOnline: true });
      get().flushQueue();
    };
    const handleOffline = () => set({ isOnline: false });
    
    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);
    
    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  },

  queueFarmIntelligence: async (uid, payload) => {
    const db = await initDB();
    const item = {
      uid,
      payload,
      createdAt: Date.now(),
      deviceId: getDeviceId(),
      synced: false,
      attempts: 0,
      error: null,
    };
    await db.add(STORE_NAME, item);
    set(state => ({ pendingCount: state.pendingCount + 1 }));

    if (navigator.onLine) {
      get().flushQueue();
    }
  },

  flushQueue: async () => {
    if (get().isSyncing || !navigator.onLine) return;
    set({ isSyncing: true });

    try {
      const db = await initDB();
      const tx = db.transaction(STORE_NAME, 'readwrite');
      const store = tx.objectStore(STORE_NAME);
      const pending = await store.index('by_synced').getAll(0);

      for (const item of pending) {
        try {
          // Replace with your actual sync endpoint or Firestore write
          const token = await window.__firebase_auth?.currentUser?.getIdToken?.();
          const res = await fetch('/api/advisory/farm-intelligence', {
            method: 'POST',
            headers: { 
              'Content-Type': 'application/json',
              ...(token ? { 'Authorization': `Bearer ${token}` } : {}),
            },
            body: JSON.stringify({
              uid: item.uid,
              payload: item.payload,
              deviceId: item.deviceId,
              createdAt: item.createdAt,
            }),
          });
          
          if (res.ok) {
            item.synced = true;
            await store.put(item);
          } else {
            throw new Error(`HTTP ${res.status}`);
          }
        } catch (err) {
          item.attempts = (item.attempts || 0) + 1;
          item.error = err.message;
          await store.put(item);
          if (item.attempts >= 5) {
            item.synced = true;
            await store.put(item);
          }
        }
      }

      const remaining = (await store.index('by_synced').getAll(0)).length;
      set({ pendingCount: remaining, lastSyncAt: Date.now() });
    } finally {
      set({ isSyncing: false });
    }
  },

  getPendingForUid: async (uid) => {
    const db = await initDB();
    const tx = db.transaction(STORE_NAME, 'readonly');
    const store = tx.objectStore(STORE_NAME);
    const all = await store.index('by_uid').getAll(uid);
    return all.filter(i => !i.synced);
  },

  clearSynced: async () => {
    const db = await initDB();
    const tx = db.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);
    const all = await store.getAll();
    for (const item of all) {
      if (item.synced) {
        await store.delete(item.localId);
      }
    }
    const remaining = (await store.index('by_synced').getAll(0)).length;
    set({ pendingCount: remaining });
  },
}));