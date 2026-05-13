/**
 * Offline Map Service
 * Manages caching of map tiles and field data in IndexedDB for offline usage.
 * Uses the idb library already in the project's dependencies.
 */

const DB_NAME = 'fasal-sathi-offline-maps';
const DB_VERSION = 1;
const TILE_STORE = 'map-tiles';
const FIELD_STORE = 'field-boundaries';
const META_STORE = 'cache-metadata';

// Cache expiry: 7 days
const CACHE_EXPIRY_MS = 7 * 24 * 60 * 60 * 1000;

let dbInstance = null;

async function getDB() {
  if (dbInstance) return dbInstance;

  dbInstance = await new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onupgradeneeded = (event) => {
      const db = event.target.result;

      if (!db.objectStoreNames.contains(TILE_STORE)) {
        const tileStore = db.createObjectStore(TILE_STORE, { keyPath: 'key' });
        tileStore.createIndex('timestamp', 'timestamp', { unique: false });
        tileStore.createIndex('region', 'region', { unique: false });
      }

      if (!db.objectStoreNames.contains(FIELD_STORE)) {
        db.createObjectStore(FIELD_STORE, { keyPath: 'id' });
      }

      if (!db.objectStoreNames.contains(META_STORE)) {
        db.createObjectStore(META_STORE, { keyPath: 'key' });
      }
    };

    request.onsuccess = (event) => resolve(event.target.result);
    request.onerror = (event) => reject(event.target.error);
  });

  return dbInstance;
}

/**
 * Generate a unique key for a tile
 */
function tileKey(z, x, y, style = 'satellite') {
  return `${style}/${z}/${x}/${y}`;
}

/**
 * Fetch and cache a map tile
 */
async function cacheTile(z, x, y, url, region = 'default', style = 'satellite') {
  const key = tileKey(z, x, y, style);

  try {
    const response = await fetch(url, { mode: 'cors' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const blob = await response.blob();
    const db = await getDB();

    await new Promise((resolve, reject) => {
      const tx = db.transaction(TILE_STORE, 'readwrite');
      const store = tx.objectStore(TILE_STORE);
      const request = store.put({
        key,
        blob,
        timestamp: Date.now(),
        region,
        style,
        z, x, y,
        size: blob.size,
      });
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });

    return true;
  } catch (err) {
    console.warn(`Failed to cache tile ${key}:`, err.message);
    return false;
  }
}

/**
 * Retrieve a cached tile as an object URL
 */
async function getCachedTile(z, x, y, style = 'satellite') {
  const key = tileKey(z, x, y, style);

  try {
    const db = await getDB();
    const record = await new Promise((resolve, reject) => {
      const tx = db.transaction(TILE_STORE, 'readonly');
      const store = tx.objectStore(TILE_STORE);
      const request = store.get(key);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });

    if (!record) return null;

    // Check expiry
    if (Date.now() - record.timestamp > CACHE_EXPIRY_MS) {
      await deleteTile(z, x, y, style);
      return null;
    }

    return URL.createObjectURL(record.blob);
  } catch {
    return null;
  }
}

/**
 * Delete a single tile
 */
async function deleteTile(z, x, y, style = 'satellite') {
  const key = tileKey(z, x, y, style);
  const db = await getDB();
  await new Promise((resolve) => {
    const tx = db.transaction(TILE_STORE, 'readwrite');
    const store = tx.objectStore(TILE_STORE);
    const request = store.delete(key);
    request.onsuccess = () => resolve();
    request.onerror = () => resolve();
  });
}

/**
 * Download all tiles for a bounding box at specified zoom levels
 * Returns progress via callback: { downloaded, total, failed }
 */
async function downloadRegion({ bounds, zoomLevels = [10, 11, 12, 13], region = 'my-farm', style = 'satellite', onProgress }) {
  const { north, south, east, west } = bounds;

  // Generate tile list
  const tiles = [];
  for (const z of zoomLevels) {
    const n = Math.pow(2, z);
    const xMin = Math.floor(((west + 180) / 360) * n);
    const xMax = Math.floor(((east + 180) / 360) * n);
    const latRad = (lat) => (lat * Math.PI) / 180;
    const yMin = Math.floor((1 - Math.log(Math.tan(latRad(north)) + 1 / Math.cos(latRad(north))) / Math.PI) / 2 * n);
    const yMax = Math.floor((1 - Math.log(Math.tan(latRad(south)) + 1 / Math.cos(latRad(south))) / Math.PI) / 2 * n);

    for (let x = xMin; x <= xMax; x++) {
      for (let y = yMin; y <= yMax; y++) {
        tiles.push({ z, x, y });
      }
    }
  }

  const total = tiles.length;
  let downloaded = 0;
  let failed = 0;

  // Store metadata
  await saveMetadata(region, { bounds, zoomLevels, style, total, timestamp: Date.now(), status: 'downloading' });

  for (const { z, x, y } of tiles) {
    let url;
    if (style === 'satellite') {
      url = `https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/${z}/${y}/${x}`;
    } else {
      url = `https://tile.openstreetmap.org/${z}/${x}/${y}.png`;
    }

    const success = await cacheTile(z, x, y, url, region, style);
    if (success) {
      downloaded++;
    } else {
      failed++;
    }

    if (onProgress) {
      onProgress({ downloaded, total, failed, percent: Math.round((downloaded / total) * 100) });
    }

    // Small delay to avoid hammering servers
    await new Promise((r) => setTimeout(r, 20));
  }

  await saveMetadata(region, { bounds, zoomLevels, style, total, downloaded, failed, timestamp: Date.now(), status: 'ready' });

  return { downloaded, total, failed };
}

/**
 * Save field boundaries
 */
async function saveFieldBoundary(field) {
  const db = await getDB();
  await new Promise((resolve, reject) => {
    const tx = db.transaction(FIELD_STORE, 'readwrite');
    const store = tx.objectStore(FIELD_STORE);
    const request = store.put({ ...field, savedAt: Date.now() });
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
}

/**
 * Get all saved field boundaries
 */
async function getAllFields() {
  const db = await getDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(FIELD_STORE, 'readonly');
    const store = tx.objectStore(FIELD_STORE);
    const request = store.getAll();
    request.onsuccess = () => resolve(request.result || []);
    request.onerror = () => reject(request.error);
  });
}

/**
 * Delete a field boundary
 */
async function deleteField(id) {
  const db = await getDB();
  await new Promise((resolve) => {
    const tx = db.transaction(FIELD_STORE, 'readwrite');
    const store = tx.objectStore(FIELD_STORE);
    store.delete(id);
    tx.oncomplete = () => resolve();
  });
}

/**
 * Save region metadata
 */
async function saveMetadata(region, data) {
  const db = await getDB();
  await new Promise((resolve) => {
    const tx = db.transaction(META_STORE, 'readwrite');
    const store = tx.objectStore(META_STORE);
    store.put({ key: region, ...data });
    tx.oncomplete = () => resolve();
  });
}

/**
 * Get all cached region metadata
 */
async function getAllRegionMeta() {
  const db = await getDB();
  return new Promise((resolve) => {
    const tx = db.transaction(META_STORE, 'readonly');
    const store = tx.objectStore(META_STORE);
    const request = store.getAll();
    request.onsuccess = () => resolve(request.result || []);
    request.onerror = () => resolve([]);
  });
}

/**
 * Delete all tiles for a region
 */
async function deleteRegion(region) {
  const db = await getDB();

  // Get all tiles for this region
  const tiles = await new Promise((resolve) => {
    const tx = db.transaction(TILE_STORE, 'readonly');
    const store = tx.objectStore(TILE_STORE);
    const index = store.index('region');
    const request = index.getAll(region);
    request.onsuccess = () => resolve(request.result || []);
    request.onerror = () => resolve([]);
  });

  // Delete them
  await new Promise((resolve) => {
    const tx = db.transaction(TILE_STORE, 'readwrite');
    const store = tx.objectStore(TILE_STORE);
    tiles.forEach((tile) => store.delete(tile.key));
    tx.oncomplete = () => resolve();
  });

  // Delete metadata
  await new Promise((resolve) => {
    const tx = db.transaction(META_STORE, 'readwrite');
    const store = tx.objectStore(META_STORE);
    store.delete(region);
    tx.oncomplete = () => resolve();
  });

  return tiles.length;
}

/**
 * Get total cache storage usage estimate (bytes)
 */
async function getCacheStats() {
  const db = await getDB();
  const tiles = await new Promise((resolve) => {
    const tx = db.transaction(TILE_STORE, 'readonly');
    const store = tx.objectStore(TILE_STORE);
    const request = store.getAll();
    request.onsuccess = () => resolve(request.result || []);
    request.onerror = () => resolve([]);
  });

  const totalBytes = tiles.reduce((sum, t) => sum + (t.size || 0), 0);
  const regions = await getAllRegionMeta();

  return {
    tileCount: tiles.length,
    totalBytes,
    totalMB: (totalBytes / (1024 * 1024)).toFixed(2),
    regionCount: regions.length,
  };
}

/**
 * Clear all expired tiles
 */
async function pruneExpiredTiles() {
  const db = await getDB();
  const cutoff = Date.now() - CACHE_EXPIRY_MS;

  const tiles = await new Promise((resolve) => {
    const tx = db.transaction(TILE_STORE, 'readonly');
    const store = tx.objectStore(TILE_STORE);
    const request = store.getAll();
    request.onsuccess = () => resolve(request.result || []);
    request.onerror = () => resolve([]);
  });

  const expired = tiles.filter((t) => t.timestamp < cutoff);

  if (expired.length > 0) {
    await new Promise((resolve) => {
      const tx = db.transaction(TILE_STORE, 'readwrite');
      const store = tx.objectStore(TILE_STORE);
      expired.forEach((t) => store.delete(t.key));
      tx.oncomplete = () => resolve();
    });
  }

  return expired.length;
}

/**
 * Check if online
 */
function isOnline() {
  return navigator.onLine;
}

/**
 * Estimate download size for a region (bytes)
 */
function estimateDownloadSize(bounds, zoomLevels = [10, 11, 12, 13]) {
  const { north, south, east, west } = bounds;
  let tileCount = 0;

  for (const z of zoomLevels) {
    const n = Math.pow(2, z);
    const xMin = Math.floor(((west + 180) / 360) * n);
    const xMax = Math.floor(((east + 180) / 360) * n);
    const latRad = (lat) => (lat * Math.PI) / 180;
    const yMin = Math.floor((1 - Math.log(Math.tan(latRad(north)) + 1 / Math.cos(latRad(north))) / Math.PI) / 2 * n);
    const yMax = Math.floor((1 - Math.log(Math.tan(latRad(south)) + 1 / Math.cos(latRad(south))) / Math.PI) / 2 * n);
    tileCount += (xMax - xMin + 1) * (yMax - yMin + 1);
  }

  // Avg satellite tile ~30KB, OSM tile ~8KB
  const avgTileSize = 30 * 1024;
  return {
    tileCount,
    estimatedBytes: tileCount * avgTileSize,
    estimatedMB: ((tileCount * avgTileSize) / (1024 * 1024)).toFixed(1),
  };
}

const offlineMapService = {
  downloadRegion,
  getCachedTile,
  cacheTile,
  tileKey,
  saveFieldBoundary,
  getAllFields,
  deleteField,
  getAllRegionMeta,
  deleteRegion,
  getCacheStats,
  pruneExpiredTiles,
  estimateDownloadSize,
  isOnline,
};

export default offlineMapService;
