import React, { useState, useEffect } from 'react';
import { AlertCircle, CheckCircle, Wifi, WifiOff } from 'lucide-react';
import './OfflineIndicator.css';
import { firestoreService } from '../services/firestoreResilientService';

export default function OfflineIndicator() {
  const [isOnline, setIsOnline] = useState(navigator.onLine);
  const [queueStatus, setQueueStatus] = useState({
    queuedOperations: 0,
    isOffline: false,
    failureCount: 0
  });
  const [showDetails, setShowDetails] = useState(false);

  useEffect(() => {
    const updateStatus = () => {
      setIsOnline(navigator.onLine);
      setQueueStatus(firestoreService.getQueueStatus());
    };

    // Update on connection change
    window.addEventListener('online', updateStatus);
    window.addEventListener('offline', updateStatus);

    // Poll queue status
    const interval = setInterval(updateStatus, 2000);

    return () => {
      window.removeEventListener('online', updateStatus);
      window.removeEventListener('offline', updateStatus);
      clearInterval(interval);
    };
  }, []);

  if (isOnline && queueStatus.queuedOperations === 0) {
    return null; // Don't show if everything is fine
  }

  const handleSync = async () => {
    await firestoreService.flushQueue();
    setQueueStatus(firestoreService.getQueueStatus());
  };

  return (
    <div className="offline-indicator">
      {!isOnline ? (
        <div className="indicator offline">
          <WifiOff size={16} />
          <span>Offline - Changes will sync when connected</span>
          {queueStatus.queuedOperations > 0 && (
            <span className="queue-badge">{queueStatus.queuedOperations} pending</span>
          )}
        </div>
      ) : queueStatus.queuedOperations > 0 ? (
        <div className="indicator syncing">
          <AlertCircle size={16} />
          <span>{queueStatus.queuedOperations} changes syncing...</span>
          <button onClick={handleSync} className="sync-btn">
            Sync Now
          </button>
        </div>
      ) : (
        <div className="indicator online" onClick={() => setShowDetails(!showDetails)}>
          <CheckCircle size={16} />
          <span>All changes synced</span>
        </div>
      )}

      {showDetails && (
        <div className="details">
          <div className="detail-item">
            <span>Status:</span>
            <span>{isOnline ? 'Online' : 'Offline'}</span>
          </div>
          <div className="detail-item">
            <span>Pending Operations:</span>
            <span>{queueStatus.queuedOperations}</span>
          </div>
          <div className="detail-item">
            <span>Failures:</span>
            <span>{queueStatus.failureCount}</span>
          </div>
        </div>
      )}
    </div>
  );
}
