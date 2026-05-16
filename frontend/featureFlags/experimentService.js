// frontend/featureFlags/experimentService.js
import axios from 'axios';

const ASSIGNMENT_CACHE_KEY = 'fs_exp_assignments';

class ExperimentService {
  constructor() {
    this.assignments = {};
    this._loadCachedAssignments();
  }

  _loadCachedAssignments() {
    try {
      const cached = sessionStorage.getItem(ASSIGNMENT_CACHE_KEY);
      if (cached) {
        this.assignments = JSON.parse(cached);
      }
    } catch (e) {
      this.assignments = {};
    }
  }

  /**
   * Assign a user to an experiment variant.
   * 
   * @param {string} experimentId 
   * @param {string} userId 
   * @returns {Promise<string>} variant id
   */
  async getVariant(experimentId, userId) {
    if (!userId) return 'control';

    // Check memory/session cache first to prevent flickering
    if (this.assignments[experimentId]) {
      return this.assignments[experimentId];
    }

    try {
      const response = await axios.post('/api/experiments/assign', {
        user_id: userId,
        experiment_id: experimentId
      });

      const variant = response.data.variant || 'control';
      
      this.assignments[experimentId] = variant;
      this._saveCache();
      
      return variant;
    } catch (error) {
      console.error(`Failed to assign experiment ${experimentId}:`, error);
      return 'control';
    }
  }

  _saveCache() {
    sessionStorage.setItem(ASSIGNMENT_CACHE_KEY, JSON.stringify(this.assignments));
  }
}

export const experimentService = new ExperimentService();
