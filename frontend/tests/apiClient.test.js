import { beforeEach, describe, expect, it, vi } from 'vitest';

const axiosMocks = vi.hoisted(() => {
  const request = vi.fn();
  const create = vi.fn(() => ({
    request,
    interceptors: {
      request: { use: vi.fn() },
      response: { use: vi.fn() },
    },
  }));

  return { request, create };
});

vi.mock('axios', () => ({
  default: {
    create: axiosMocks.create,
  },
}));

vi.mock('../stores/uiStore', () => ({
  useUiStore: {
    getState: () => ({
      incrementApiPendingRequests: vi.fn(),
      decrementApiPendingRequests: vi.fn(),
    }),
  },
}));

vi.mock('../utils/errorReporting', () => ({
  reportErrorToBackend: vi.fn(),
}));

vi.mock('../lib/firebase', () => ({
  auth: { currentUser: null },
}));

describe('services/api', () => {
  beforeEach(() => {
    vi.resetModules();
    axiosMocks.request.mockReset();
    axiosMocks.create.mockClear();
  });

  it('deduplicates concurrent in-flight GET requests', async () => {
    let resolveRequest;
    axiosMocks.request.mockImplementationOnce(() => {
      return new Promise((resolve) => {
        resolveRequest = resolve;
      });
    });

    const { default: apiClient } = await import('../services/api.js');

    const firstRequest = apiClient.get('/api/weather');
    const secondRequest = apiClient.get('/api/weather');

    expect(axiosMocks.request).toHaveBeenCalledTimes(1);
    expect(firstRequest).toBe(secondRequest);

    resolveRequest({ data: { ok: true }, config: {} });

    await expect(firstRequest).resolves.toEqual({ data: { ok: true }, config: {} });
    await expect(secondRequest).resolves.toEqual({ data: { ok: true }, config: {} });
  });

  it('opens the circuit breaker after repeated failures and recovers after the reset window', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-05-21T00:00:00.000Z'));

    axiosMocks.request
      .mockRejectedValueOnce(new Error('backend unavailable'))
      .mockResolvedValueOnce({ data: { ok: true }, config: {} });

    const { default: apiClient } = await import('../services/api.js');

    await expect(
      apiClient.get('/api/market/forecast', {
        circuitBreakerThreshold: 1,
        circuitBreakerResetMs: 1000,
        retries: 0,
        dedupe: false,
      })
    ).rejects.toThrow('backend unavailable');

    await expect(
      apiClient.get('/api/market/forecast', {
        circuitBreakerThreshold: 1,
        circuitBreakerResetMs: 1000,
        retries: 0,
        dedupe: false,
      })
    ).rejects.toMatchObject({ code: 'ERR_CIRCUIT_BREAKER_OPEN' });

    expect(axiosMocks.request).toHaveBeenCalledTimes(1);

    await vi.advanceTimersByTimeAsync(1001);

    await expect(
      apiClient.get('/api/market/forecast', {
        circuitBreakerThreshold: 1,
        circuitBreakerResetMs: 1000,
        retries: 0,
        dedupe: false,
      })
    ).resolves.toEqual({ data: { ok: true }, config: {} });

    expect(axiosMocks.request).toHaveBeenCalledTimes(2);
    vi.useRealTimers();
  });

  it('uses the configured backend base URL when provided', async () => {
    vi.stubEnv('VITE_API_BASE_URL', 'https://api.example.com/');

    await import('../services/api.js');

    expect(axiosMocks.create).toHaveBeenCalledWith(expect.objectContaining({
      baseURL: 'https://api.example.com',
    }));
  });
});
