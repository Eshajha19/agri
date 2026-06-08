import axios from 'axios';

const ASSIGNMENT_CACHE_KEY = 'fs_exp_assignments';
const ASSIGNMENT_CACHE_TTL_MS = 30 * 60 * 1000; // 30 minutes

class ExperimentService {
  constructor() {
    this.assignments = {};
    this.pendingRequests = new Map();

    this.metrics = {
      cacheHits: 0,
      cacheMisses: 0,
      apiCalls: 0,
      failures: 0,
    };

    this._loadCachedAssignments();
  }

  _loadCachedAssignments() {
    try {
      const cached = sessionStorage.getItem(
        ASSIGNMENT_CACHE_KEY
      );

      if (!cached) {
        return;
      }

      const parsed = JSON.parse(cached);

      const now = Date.now();

      Object.entries(parsed).forEach(
        ([experimentId, value]) => {
          if (
            value?.timestamp &&
            now - value.timestamp <
              ASSIGNMENT_CACHE_TTL_MS
          ) {
            this.assignments[experimentId] = value;
          }
        }
      );
    } catch (e) {
      this.assignments = {};
    }
  }

  _saveCache() {
    try {
      sessionStorage.setItem(
        ASSIGNMENT_CACHE_KEY,
        JSON.stringify(this.assignments)
      );
    } catch (e) {
      console.error(
        'Failed to persist experiment cache:',
        e
      );
    }
  }

  _cleanupExpiredAssignments() {
    const now = Date.now();

    Object.keys(this.assignments).forEach(
      (experimentId) => {
        const assignment =
          this.assignments[experimentId];

        if (
          !assignment?.timestamp ||
          now - assignment.timestamp >
            ASSIGNMENT_CACHE_TTL_MS
        ) {
          delete this.assignments[
            experimentId
          ];
        }
      }
    );
  }

  /**
   * Assign a user to an experiment variant.
   *
   * @param {string} experimentId
   * @param {string} userId
   * @returns {Promise<string>}
   */
  async getVariant(
    experimentId,
    userId
  ) {
    if (!userId) {
      return 'control';
    }

    this._cleanupExpiredAssignments();

    const cached =
      this.assignments[experimentId];

    if (cached?.variant) {
      this.metrics.cacheHits++;

      return cached.variant;
    }

    this.metrics.cacheMisses++;

    if (
      this.pendingRequests.has(
        experimentId
      )
    ) {
      return this.pendingRequests.get(
        experimentId
      );
    }

    const requestPromise = axios
      .post('/api/experiments/assign', {
        user_id: userId,
        experiment_id: experimentId,
      })
      .then((response) => {
        this.metrics.apiCalls++;

        const variant =
          response.data.variant ||
          'control';

        this.assignments[
          experimentId
        ] = {
          variant,
          timestamp: Date.now(),
        };

        this._saveCache();

        return variant;
      })
      .catch((error) => {
        this.metrics.failures++;

        console.error(
          `Failed to assign experiment ${experimentId}:`,
          error
        );

        return 'control';
      })
      .finally(() => {
        this.pendingRequests.delete(
          experimentId
        );
      });

    this.pendingRequests.set(
      experimentId,
      requestPromise
    );

    return requestPromise;
  }

  /**
   * Preload assignments for multiple experiments.
   */
  async preloadAssignments(
    experimentIds,
    userId
  ) {
    if (
      !Array.isArray(experimentIds)
    ) {
      return {};
    }

    const results = {};

    await Promise.all(
      experimentIds.map(
        async (experimentId) => {
          results[experimentId] =
            await this.getVariant(
              experimentId,
              userId
            );
        }
      )
    );

    return results;
  }

  /**
   * Metrics useful for debugging
   * aggregation efficiency.
   */
  getMetrics() {
    return {
      ...this.metrics,
      cachedAssignments:
        Object.keys(
          this.assignments
        ).length,
      pendingRequests:
        this.pendingRequests.size,
    };
  }

  clearCache() {
    this.assignments = {};

    try {
      sessionStorage.removeItem(
        ASSIGNMENT_CACHE_KEY
      );
    } catch (e) {
      console.error(
        'Failed to clear experiment cache:',
        e
      );
    }
  }
}

export const experimentService =
  new ExperimentService();