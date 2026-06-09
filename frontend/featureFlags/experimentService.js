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
      totalAssignments: 0,
    };

    this._loadCachedAssignments();
  }

  _getCacheKey(experimentId, userId) {
    return `${userId}:${experimentId}`;
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
        ([cacheKey, value]) => {
          if (
            value?.timestamp &&
            now - value.timestamp <
              ASSIGNMENT_CACHE_TTL_MS
          ) {
            this.assignments[cacheKey] = value;
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
      (cacheKey) => {
        const assignment =
          this.assignments[cacheKey];

        if (
          !assignment?.timestamp ||
          now - assignment.timestamp >
            ASSIGNMENT_CACHE_TTL_MS
        ) {
          delete this.assignments[
            cacheKey
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

    const cacheKey =
      this._getCacheKey(
        experimentId,
        userId
      );

    const cached =
      this.assignments[cacheKey];

    if (cached?.variant) {
      this.metrics.cacheHits++;

      return cached.variant;
    }

    this.metrics.cacheMisses++;

    if (
      this.pendingRequests.has(
        cacheKey
      )
    ) {
      return this.pendingRequests.get(
        cacheKey
      );
    }

    const requestPromise = axios
      .post('/api/experiments/assign', {
        user_id: userId,
        experiment_id: experimentId,
      })
      .then((response) => {
        this.metrics.apiCalls++;
        this.metrics.totalAssignments++;

        const variant =
          response.data.variant ||
          'control';

        this.assignments[
          cacheKey
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
          cacheKey
        );
      });

    this.pendingRequests.set(
      cacheKey,
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

    /**
   * Warm cache for a collection of experiments.
   */
  async warmCache(
    experimentIds,
    userId
  ) {
    const results = {};

    for (const experimentId of experimentIds) {
      results[experimentId] =
        await this.getVariant(
          experimentId,
          userId
        );
    }

    return results;
  }

  /**
   * Remove a single assignment from cache.
   */
  invalidateAssignment(
    experimentId,
    userId
  ) {
    const cacheKey =
      this._getCacheKey(
        experimentId,
        userId
      );

    delete this.assignments[
      cacheKey
    ];

    this._saveCache();
  }

  /**
   * Assignment cache statistics.
   */
  getCacheStats() {
    const now = Date.now();

    const activeEntries =
      Object.values(
        this.assignments
      ).filter(
        (entry) =>
          entry?.timestamp &&
          now - entry.timestamp <
            ASSIGNMENT_CACHE_TTL_MS
      ).length;

    return {
      activeEntries,
      cacheHits:
        this.metrics.cacheHits,
      cacheMisses:
        this.metrics.cacheMisses,
      hitRate:
        this.metrics.cacheHits +
          this.metrics.cacheMisses >
        0
          ? (
              (this.metrics.cacheHits /
                (this.metrics.cacheHits +
                  this.metrics
                    .cacheMisses)) *
              100
            ).toFixed(2)
          : 0,
    };
  }

  /**
   * Aggregated assignment report.
   */
  getAssignmentReport() {
    return {
      totalAssignments:
        this.metrics
          .totalAssignments,
      apiCalls:
        this.metrics.apiCalls,
      failures:
        this.metrics.failures,
      cachedAssignments:
        Object.keys(
          this.assignments
        ).length,
      pendingRequests:
        this.pendingRequests.size,
      generatedAt:
        new Date().toISOString(),
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