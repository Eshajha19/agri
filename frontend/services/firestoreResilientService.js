/**
 * Frontend Firestore Resilience Service
 * Provides retry logic, exponential backoff, and offline queue support
 */

class FirestoreResilientService {
  constructor(maxRetries = 3) {
    this.maxRetries = maxRetries;
    this.writeQueue = [];
    this.isOffline = false;
    this.failureCount = 0;
    this.lastFailureTime = null;
    
    // Monitor connection status
    window.addEventListener('online', () => this.handleOnline());
    window.addEventListener('offline', () => this.handleOffline());
  }

  /**
   * Calculate exponential backoff with jitter
   */
  getBackoffDelay(attempt) {
    const baseDelay = 1000; // 1 second
    const maxDelay = 60000; // 60 seconds
    
    // Exponential: 1s, 2s, 4s, 8s...
    let delay = baseDelay * Math.pow(2, attempt);
    delay = Math.min(delay, maxDelay);
    
    // Add jitter (±20%)
    const jitter = delay * 0.2 * (Math.random() - 0.5);
    return Math.max(0, delay + jitter);
  }

  /**
   * Execute operation with retry logic
   */
  async executeWithRetry(operation, context = '') {
    let lastError = null;

    for (let attempt = 0; attempt < this.maxRetries; attempt++) {
      try {
        const result = await operation();
        this.failureCount = 0;
        this.isOffline = false;
        return result;
      } catch (error) {
        lastError = error;
        this.failureCount++;

        // Check if error is retryable
        const isRetryable = this.isRetryableError(error);

        if (!isRetryable || attempt === this.maxRetries - 1) {
          console.error(
            `Firestore operation failed after ${attempt + 1} attempts (${context}):`,
            error
          );
          throw error;
        }

        // Wait before retrying
        const delay = this.getBackoffDelay(attempt);
        console.warn(
          `Firestore operation failed (attempt ${attempt + 1}, ${context}). ` +
          `Retrying in ${(delay / 1000).toFixed(1)}s`
        );

        await this.sleep(delay);
      }
    }

    throw lastError;
  }

  /**
   * Check if error is retryable (transient)
   */
  isRetryableError(error) {
    const retryableMessages = [
      'deadline exceeded',
      'unavailable',
      'temporarily unavailable',
      'connection',
      'network',
      'timeout',
      'ECONNREFUSED',
      'ETIMEDOUT'
    ];

    const errorMsg = (error.message || '').toLowerCase();
    return retryableMessages.some(msg => errorMsg.includes(msg));
  }

  /**
   * Queue a write operation for offline
   */
  queueWrite(collection, docId, data, operation = 'set') {
    this.writeQueue.push({
      collection,
      docId,
      data,
      operation,
      timestamp: new Date().toISOString()
    });

    console.log(`Queued ${operation} operation for ${collection}/${docId}`);
    
    // Try to flush immediately if online
    if (!this.isOffline) {
      this.flushQueue();
    }
  }

  /**
   * Attempt to flush queued writes
   */
  async flushQueue() {
    if (this.writeQueue.length === 0 || this.isOffline) {
      return;
    }

    const queue = [...this.writeQueue];
    this.writeQueue = [];

    let flushed = 0;

    for (const op of queue) {
      try {
        if (op.operation === 'set') {
          await this.executeWithRetry(
            () => db.collection(op.collection).doc(op.docId).set(op.data, { merge: true }),
            `flush/${op.collection}/${op.docId}`
          );
        } else if (op.operation === 'delete') {
          await this.executeWithRetry(
            () => db.collection(op.collection).doc(op.docId).delete(),
            `flush/${op.collection}/${op.docId}`
          );
        }
        flushed++;
      } catch (error) {
        console.error(`Failed to flush ${op.operation} for ${op.collection}/${op.docId}:`, error);
        // Re-queue failed operation
        this.writeQueue.push(op);
      }
    }

    if (flushed > 0) {
      console.log(`Flushed ${flushed} queued operations`);
    }
  }

  /**
   * Get document
   */
  async getDoc(collection, docId) {
    return this.executeWithRetry(
      async () => {
        const doc = await db.collection(collection).doc(docId).get();
        return doc.exists ? doc.data() : null;
      },
      `getDoc/${collection}/${docId}`
    );
  }

  /**
   * Set document
   */
  async setDoc(collection, docId, data) {
    try {
      return await this.executeWithRetry(
        () => db.collection(collection).doc(docId).set(data, { merge: true }),
        `setDoc/${collection}/${docId}`
      );
    } catch (error) {
      if (this.isOffline) {
        this.queueWrite(collection, docId, data, 'set');
      } else {
        throw error;
      }
    }
  }

  /**
   * Add document
   */
  async addDoc(collection, data) {
    return this.executeWithRetry(
      async () => {
        const docRef = await db.collection(collection).add(data);
        return docRef.id;
      },
      `addDoc/${collection}`
    );
  }

  /**
   * Delete document
   */
  async deleteDoc(collection, docId) {
    try {
      return await this.executeWithRetry(
        () => db.collection(collection).doc(docId).delete(),
        `deleteDoc/${collection}/${docId}`
      );
    } catch (error) {
      if (this.isOffline) {
        this.queueWrite(collection, docId, {}, 'delete');
      } else {
        throw error;
      }
    }
  }

  /**
   * Query collection
   */
  async queryCollection(collection, filters = []) {
    return this.executeWithRetry(
      async () => {
        let query = db.collection(collection);

        for (const { field, operator, value } of filters) {
          query = query.where(field, operator, value);
        }

        const snapshot = await query.get();
        return snapshot.docs.map(doc => ({ id: doc.id, ...doc.data() }));
      },
      `queryCollection/${collection}`
    );
  }

  /**
   * Batch write operations
   */
  async batchWrite(operations) {
    return this.executeWithRetry(
      async () => {
        const batch = db.batch();

        for (const { type, collection, docId, data } of operations) {
          const docRef = db.collection(collection).doc(docId);
          
          if (type === 'set') {
            batch.set(docRef, data, { merge: true });
          } else if (type === 'delete') {
            batch.delete(docRef);
          }
        }

        return batch.commit();
      },
      `batchWrite`
    );
  }

  /**
   * Handle online status
   */
  handleOnline() {
    console.log('Connection restored');
    this.isOffline = false;
    this.flushQueue();
  }

  /**
   * Handle offline status
   */
  handleOffline() {
    console.log('Connection lost');
    this.isOffline = true;
  }

  /**
   * Sleep helper
   */
  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Get queue status
   */
  getQueueStatus() {
    return {
      queuedOperations: this.writeQueue.length,
      isOffline: this.isOffline,
      failureCount: this.failureCount,
      lastFailureTime: this.lastFailureTime
    };
  }
}

// Export singleton instance
export const firestoreService = new FirestoreResilientService();
