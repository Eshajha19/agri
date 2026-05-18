// frontend/featureFlags/analyticsService.js
import axios from 'axios';

class AnalyticsService {
  /**
   * Log an event related to a feature flag or experiment.
   */
  async logEvent({ eventType, userId, experimentId, variant, flagId, metadata }) {
    try {
      await axios.post('/api/experiments/events', {
        event_type: eventType,
        user_id: userId,
        experiment_id: experimentId,
        variant: variant,
        flag_id: flagId,
        metadata: metadata || {},
        session_id: window.sessionStorage.getItem('fs_session_id') || this._generateSessionId()
      });
    } catch (error) {
      // Fail silently for analytics to avoid disrupting UX
      console.warn('Analytics event failed to log:', error);
    }
  }

  _generateSessionId() {
    const id = 'sess_' + Math.random().toString(36).substr(2, 9);
    window.sessionStorage.setItem('fs_session_id', id);
    return id;
  }

  /**
   * Specifically track an experiment impression.
   */
  trackImpression(experimentId, variant, userId) {
    return this.logEvent({
      eventType: 'impression',
      experimentId,
      variant,
      userId
    });
  }

  /**
   * Specifically track a conversion (goal met).
   */
  trackConversion(experimentId, variant, userId, goalName) {
    return this.logEvent({
      eventType: 'conversion',
      experimentId,
      variant,
      userId,
      metadata: { goal: goalName }
    });
  }
}

export const analyticsService = new AnalyticsService();
