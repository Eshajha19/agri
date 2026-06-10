import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const dbMocks = vi.hoisted(() => ({
  getOfflineRequests: vi.fn(),
  deleteOfflineRequest: vi.fn(),
}));

vi.mock('../lib/db.js', () => dbMocks);

describe('syncOfflineRequests', () => {
  beforeEach(() => {
    dbMocks.getOfflineRequests.mockReset();
    dbMocks.deleteOfflineRequest.mockReset();
    vi.stubGlobal('fetch', vi.fn());
    vi.stubGlobal('navigator', { onLine: true });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('does nothing while offline', async () => {
    navigator.onLine = false;

    const { syncOfflineRequests } = await import('../lib/syncOfflineRequests.js');

    const result = await syncOfflineRequests();

    expect(result).toEqual({ attempted: 0, synced: 0, failed: 0, remaining: 0 });
    expect(fetch).not.toHaveBeenCalled();
  });

  it('replays queued requests in order without double-stringifying bodies', async () => {
    const queuedRequests = [
      {
        id: 2,
        url: '/api/second',
        method: 'POST',
        body: '{"second":true}',
        headers: { 'Content-Type': 'application/json' },
        queuedAt: '2024-01-02T00:00:00.000Z',
      },
      {
        id: 1,
        url: '/api/first',
        method: 'POST',
        body: '{"first":true}',
        headers: { 'Content-Type': 'application/json' },
        queuedAt: '2024-01-01T00:00:00.000Z',
      },
      {
        id: 3,
        url: '/api/fail',
        method: 'PATCH',
        body: '{"fail":true}',
        headers: { 'Content-Type': 'application/json' },
        queuedAt: '2024-01-03T00:00:00.000Z',
      },
    ];

    dbMocks.getOfflineRequests.mockResolvedValueOnce(queuedRequests);
    dbMocks.deleteOfflineRequest.mockResolvedValue(true);

    fetch
      .mockResolvedValueOnce({ ok: true, status: 200 })
      .mockResolvedValueOnce({ ok: true, status: 200 })
      .mockResolvedValueOnce({ ok: false, status: 500 });

    const { syncOfflineRequests } = await import('../lib/syncOfflineRequests.js');

    const result = await syncOfflineRequests();

    expect(fetch).toHaveBeenNthCalledWith(1, '/api/first', expect.objectContaining({ body: '{"first":true}' }));
    expect(fetch).toHaveBeenNthCalledWith(2, '/api/second', expect.objectContaining({ body: '{"second":true}' }));
    expect(fetch).toHaveBeenNthCalledWith(3, '/api/fail', expect.objectContaining({ body: '{"fail":true}' }));
    expect(dbMocks.deleteOfflineRequest).toHaveBeenCalledTimes(2);
    expect(result).toEqual({ attempted: 3, synced: 2, failed: 1, remaining: 1 });
  });
});