// frontend/featureFlags/featureFlagService.js
import axios from 'axios';
import { DEFAULT_FLAGS } from './flagConfig';

const FLAG_CACHE_KEY = 'fs_feature_flags';
const CACHE_TTL = 5 * 60 * 1000; // 5 minutes

class FeatureFlagService {
  constructor() {
    this.flags = DEFAULT_FLAGS;
    this.lastFetch = 0;
    this._initPromise = null;
  }

  /**
   * Initialize the service by fetching latest flags from the backend.
   */
  async init() {
    if (this._initPromise) return this._initPromise;

    this._initPromise = (async () => {
      try {
        // Try to load from localStorage first for immediate responsiveness
        const cached = localStorage.getItem(FLAG_CACHE_KEY);
        if (cached) {
          const { data, timestamp } = JSON.parse(cached);
          if (Date.now() - timestamp < CACHE_TTL) {
            this.flags = { ...DEFAULT_FLAGS, ...data };
            this.lastFetch = timestamp;
          }
        }

        // Fetch fresh flags if cache is old or missing
        if (Date.now() - this.lastFetch > CACHE_TTL) {
          await this.fetchFlags();
        }
      } catch (error) {
        console.error('FeatureFlagService init failed:', error);
        // We still have DEFAULT_FLAGS to fall back on
      }
    })();

    return this._initPromise;
  }

  async fetchFlags() {
    try {
      const response = await axios.get('/api/flags');
      const backendFlags = {};
      
      if (response.data && response.data.flags) {
        response.data.flags.forEach(f => {
          backendFlags[f.id] = f;
        });
      }

      this.flags = { ...DEFAULT_FLAGS, ...backendFlags };
      this.lastFetch = Date.now();

      localStorage.setItem(FLAG_CACHE_KEY, JSON.stringify({
        data: backendFlags,
        timestamp: this.lastFetch
      }));
    } catch (error) {
      console.warn('Could not fetch feature flags from backend, using defaults.', error);
      throw error;
    }
  }

  /**
   * Evaluate if a feature flag is enabled for a given user.
   * 
   * @param {string} flagId 
   * @param {string} userId 
   * @returns {boolean}
   */
  isEnabled(flagId, userId) {
    const flag = this.flags[flagId];
    if (!flag) return false;
    if (!flag.enabled) return false;
    if (flag.rollout_pct === 100) return true;
    if (flag.rollout_pct === 0) return false;

    // Deterministic rollout based on userId + flagId
    // Uses a simple FNV-1a hash implementation
    const hash = this._hashString(`${userId}:${flagId}`);
    const bucket = Math.abs(hash) % 100;

    return bucket < flag.rollout_pct;
  }

  /**
   * Simple hash function for deterministic assignment.
   */
  _hashString(str) {
    let hash = 0x811c9dc5;
    for (let i = 0; i < str.length; i++) {
      hash ^= str.charCodeAt(i);
      hash += (hash << 1) + (hash << 4) + (hash << 7) + (hash << 8) + (hash << 24);
    }
    return hash >>> 0;
  }

  getFlags() {
    return this.flags;
  }
}

export const featureFlagService = new FeatureFlagService();
