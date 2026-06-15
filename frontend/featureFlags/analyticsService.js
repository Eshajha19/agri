// frontend/featureFlags/analyticsService.js
import axios from 'axios';

class AnalyticsService {
  constructor() {
    this.queueKey = 'analytics_event_queue';
    this.metricsKey = 'analytics_delivery_metrics';

    window.addEventListener('online', () => {
      this.flushQueue();
    });

    this.flushQueue();
  }

  /**
   * Log an event related to a feature flag or experiment.
   */
  async logEvent({
    eventType,
    userId,
    experimentId,
    variant,
    flagId,
    metadata
  }) {
    const payload = {
      event_type: eventType,
      user_id: userId,
      experiment_id: experimentId,
      variant,
      flag_id: flagId,
      metadata: metadata || {},
      session_id:
        window.sessionStorage.getItem('fs_session_id') ||
        this._generateSessionId()
    };

    try {
      const success = await this._sendWithRetry(payload);

      if (!success) {
        this._enqueue(payload);
      }
    } catch (error) {
      this._enqueue(payload);

      console.warn(
        'Analytics event buffered for later delivery:',
        error
      );
    }
  }

  /**
   * Generate persistent session id.
   */
  _generateSessionId() {
    const id =
      'sess_' +
      Math.random()
        .toString(36)
        .substr(2, 9);

    window.sessionStorage.setItem(
      'fs_session_id',
      id
    );

    return id;
  }

  /**
   * Load queued analytics events.
   */
  _loadQueue() {
    try {
      return JSON.parse(
        localStorage.getItem(this.queueKey) || '[]'
      );
    } catch {
      return [];
    }
  }

  /**
   * Persist queue.
   */
  _saveQueue(queue) {
    localStorage.setItem(
      this.queueKey,
      JSON.stringify(queue)
    );
  }

  /**
   * Store delivery metrics.
   */
  _updateMetrics(success) {
    const metrics = JSON.parse(
      localStorage.getItem(this.metricsKey) ||
        '{"success":0,"failed":0}'
    );

    if (success) {
      metrics.success += 1;
    } else {
      metrics.failed += 1;
    }

    localStorage.setItem(
      this.metricsKey,
      JSON.stringify(metrics)
    );
  }

  /**
   * Buffer failed event.
   */
  _enqueue(eventPayload) {
    const queue = this._loadQueue();

    queue.push({
      ...eventPayload,
      retryCount: 0,
      createdAt: Date.now()
    });

    this._saveQueue(queue);
  }

  /**
   * Retry failed submissions using exponential backoff.
   */
  async _sendWithRetry(payload, retries = 3) {
    let attempt = 0;

    while (attempt < retries) {
      try {
        await axios.post(
          '/api/experiments/events',
          payload
        );

        this._updateMetrics(true);
        return true;
      } catch (error) {
        attempt += 1;

        await new Promise(resolve =>
          setTimeout(
            resolve,
            Math.pow(2, attempt) * 1000
          )
        );
      }
    }

    this._updateMetrics(false);
    return false;
  }

  /**
   * Flush buffered events when connectivity returns.
   */
  async flushQueue() {
    if (!navigator.onLine) {
      return;
    }

    const queue = this._loadQueue();

    if (!queue.length) {
      return;
    }

    const remaining = [];

    for (const event of queue) {
      const success =
        await this._sendWithRetry(event);

      if (!success) {
        event.retryCount =
          (event.retryCount || 0) + 1;

        remaining.push(event);
      }
    }

    this._saveQueue(remaining);
  }

  /**
   * Delivery diagnostics.
   */
  getDeliveryMetrics() {
    return JSON.parse(
      localStorage.getItem(this.metricsKey) ||
        '{"success":0,"failed":0}'
    );
  }

  /**
   * Queue diagnostics.
   */
  getBufferedEventCount() {
    return this._loadQueue().length;
  }

  /**
   * Specifically track an experiment impression.
   */
  trackImpression(
    experimentId,
    variant,
    userId
  ) {
    return this.logEvent({
      eventType: 'impression',
      experimentId,
      variant,
      userId
    });
  }

  /**
   * Specifically track a conversion.
   */
  trackConversion(
    experimentId,
    variant,
    userId,
    goalName
  ) {
    return this.logEvent({
      eventType: 'conversion',
      experimentId,
      variant,
      userId,
      metadata: {
        goal: goalName
      }
    });
  }
}

export const analyticsService =
  new AnalyticsService();