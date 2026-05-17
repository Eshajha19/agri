import { useEffect } from 'react';

const DEFAULT_USAGE_RATIO_LIMIT = 0.85;
const DEFAULT_LOW_PRIORITY_CACHES = [
  'unsplash-images',
  'images',
  'market-prices-api',
  'geocoding-api',
  'ip-geo-api',
];

export const estimateStorageUsageBytes = (storage) => {
  if (!storage) return 0;

  let total = 0;
  for (let index = 0; index < storage.length; index += 1) {
    const key = storage.key(index);
    const value = storage.getItem(key) || '';
    total += key.length + value.length;
  }
  return total;
};

export const getStorageUsageRatio = async () => {
  if (typeof navigator === 'undefined' || !navigator.storage?.estimate) {
    return { usage: 0, quota: 0, ratio: 0 };
  }

  const estimate = await navigator.storage.estimate();
  const usage = Number(estimate?.usage || 0);
  const quota = Number(estimate?.quota || 0);
  return {
    usage,
    quota,
    ratio: quota > 0 ? usage / quota : 0,
  };
};

export const enforceBrowserCacheBudget = async ({
  usageRatioLimit = DEFAULT_USAGE_RATIO_LIMIT,
  lowPriorityCaches = DEFAULT_LOW_PRIORITY_CACHES,
} = {}) => {
  const report = {
    usage: 0,
    quota: 0,
    ratio: 0,
    trimmed: false,
    deletedCaches: [],
  };

  const storage = await getStorageUsageRatio();
  report.usage = storage.usage;
  report.quota = storage.quota;
  report.ratio = storage.ratio;

  if (storage.ratio <= usageRatioLimit || typeof caches === 'undefined' || !caches.keys) {
    return report;
  }

  const cacheNames = await caches.keys();
  for (const cacheName of lowPriorityCaches) {
    if (cacheNames.includes(cacheName) && await caches.delete(cacheName)) {
      report.deletedCaches.push(cacheName);
      report.trimmed = true;
    }
  }

  return report;
};

export const useBrowserCacheBudget = ({
  enabled = true,
  usageRatioLimit = DEFAULT_USAGE_RATIO_LIMIT,
  lowPriorityCaches = DEFAULT_LOW_PRIORITY_CACHES,
  onReport,
} = {}) => {
  useEffect(() => {
    if (!enabled) return undefined;
    if (typeof window === 'undefined' || typeof document === 'undefined') return undefined;

    let cancelled = false;

    const run = async () => {
      const report = await enforceBrowserCacheBudget({
        usageRatioLimit,
        lowPriorityCaches,
      });

      if (!cancelled && typeof onReport === 'function') {
        onReport(report);
      }
    };

    run();

    const handleFocus = () => run();
    const handleVisibility = () => {
      if (document.visibilityState === 'visible') {
        run();
      }
    };

    window.addEventListener('focus', handleFocus);
    document.addEventListener('visibilitychange', handleVisibility);

    return () => {
      cancelled = true;
      window.removeEventListener('focus', handleFocus);
      document.removeEventListener('visibilitychange', handleVisibility);
    };
  }, [enabled, usageRatioLimit, lowPriorityCaches, onReport]);
};
