import {
  deleteOfflineSnapshot,
  getOfflineSnapshot,
  saveOfflineSnapshot,
} from './db.js';

const APP_STATE_KEY = 'offline:app-state';
const USER_PROFILE_PREFIX = 'offline:user-profile:';
const DRAFT_PREFIX = 'offline:draft:';

const keyForUserProfile = (uid) => `${USER_PROFILE_PREFIX}${uid}`;
const keyForDraft = (scope) => `${DRAFT_PREFIX}${scope}`;

export const persistAppState = async (state) => {
  const existingState = await loadAppState();
  return saveOfflineSnapshot(APP_STATE_KEY, {
    ...(existingState || {}),
    ...(state || {}),
  });
};

export const loadAppState = async () => {
  return getOfflineSnapshot(APP_STATE_KEY);
};

export const persistUserProfileSnapshot = async (uid, profile) => {
  if (!uid) return null;
  return saveOfflineSnapshot(keyForUserProfile(uid), profile);
};

export const loadUserProfileSnapshot = async (uid) => {
  if (!uid) return null;
  return getOfflineSnapshot(keyForUserProfile(uid));
};

export const clearUserProfileSnapshot = async (uid) => {
  if (!uid) return null;
  return deleteOfflineSnapshot(keyForUserProfile(uid));
};

export const persistDraft = async (scope, draft) => {
  return saveOfflineSnapshot(keyForDraft(scope), draft);
};

export const loadDraft = async (scope) => {
  return getOfflineSnapshot(keyForDraft(scope));
};

export const clearDraft = async (scope) => {
  return deleteOfflineSnapshot(keyForDraft(scope));
};