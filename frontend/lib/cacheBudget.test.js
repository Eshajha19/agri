import { describe, expect, it, vi } from 'vitest';
import { estimateStorageUsageBytes, enforceBrowserCacheBudget } from './cacheBudget';

const makeStorage = (entries) => ({
  length: entries.length,
  key: (index) => entries[index]?.[0] ?? null,
  getItem: (key) => entries.find(([entryKey]) => entryKey === key)?.[1] ?? null,
});

describe('cacheBudget', () => {
  it('estimates local storage usage in bytes', () => {
    const storage = makeStorage([
      ['a', '1234'],
      ['bb', '56'],
    ]);

    expect(estimateStorageUsageBytes(storage)).toBe(9);
  });

  it('trims low-priority caches when storage usage is high', async () => {
    const deleteCache = vi.fn(async () => true);
    vi.stubGlobal('navigator', {
      storage: {
        estimate: vi.fn(async () => ({ usage: 9, quota: 10 })),
      },
    });
    vi.stubGlobal('caches', {
      keys: vi.fn(async () => ['market-prices-api', 'images', 'offline-pages']),
      delete: deleteCache,
    });

    const report = await enforceBrowserCacheBudget({
      usageRatioLimit: 0.8,
      lowPriorityCaches: ['market-prices-api', 'images'],
    });

    expect(report.trimmed).toBe(true);
    expect(report.deletedCaches).toEqual(['market-prices-api', 'images']);
    expect(deleteCache).toHaveBeenCalledTimes(2);
  });
});
