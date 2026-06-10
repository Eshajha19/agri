import { beforeEach, describe, expect, it, vi } from 'vitest';

const dbMocks = vi.hoisted(() => ({
  saveOfflineSnapshot: vi.fn(),
  getOfflineSnapshot: vi.fn(),
  deleteOfflineSnapshot: vi.fn(),
}));

vi.mock('../lib/db.js', () => dbMocks);

describe('offlinePersistence', () => {
  beforeEach(() => {
    dbMocks.saveOfflineSnapshot.mockReset();
    dbMocks.getOfflineSnapshot.mockReset();
    dbMocks.deleteOfflineSnapshot.mockReset();
  });

  it('persists the app state snapshot under a stable key', async () => {
    const { persistAppState } = await import('../lib/offlinePersistence.js');

    dbMocks.saveOfflineSnapshot.mockResolvedValueOnce(true);

    await persistAppState({ preferredLang: 'hi' });

    expect(dbMocks.saveOfflineSnapshot).toHaveBeenCalledWith('offline:app-state', { preferredLang: 'hi' });
  });

  it('loads the user profile snapshot using the user-scoped key', async () => {
    const { loadUserProfileSnapshot } = await import('../lib/offlinePersistence.js');

    dbMocks.getOfflineSnapshot.mockResolvedValueOnce({ displayName: 'Offline Farmer' });

    const profile = await loadUserProfileSnapshot('user-123');

    expect(profile).toEqual({ displayName: 'Offline Farmer' });
    expect(dbMocks.getOfflineSnapshot).toHaveBeenCalledWith('offline:user-profile:user-123');
  });

  it('clears draft state with the namespaced draft key', async () => {
    const { clearDraft } = await import('../lib/offlinePersistence.js');

    dbMocks.deleteOfflineSnapshot.mockResolvedValueOnce(true);

    await clearDraft('soil-analysis');

    expect(dbMocks.deleteOfflineSnapshot).toHaveBeenCalledWith('offline:draft:soil-analysis');
  });
});