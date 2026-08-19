import React, { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { AlertTriangle, CheckCircle, RefreshCw, ChevronDown, ChevronUp } from 'lucide-react';
import { syncService } from '../services/syncConflictService';
import './SyncStateManager.css';

export default function SyncStateManager() {
  const { t } = useTranslation();
  const [syncState, setSyncState] = useState('synced');
  const [pendingCount, setPendingCount] = useState(0);
  const [showDetails, setShowDetails] = useState(false);
  const [conflicts, setConflicts] = useState([]);

  useEffect(() => {
    // Subscribe to sync events
    const unsubscribe = syncService.subscribe((eventType, data) => {
      if (eventType === 'sync-state-changed') {
        setSyncState(data.state);
      } else if (eventType === 'conflict-detected') {
        setConflicts(prev => [...prev, data]);
      }
    });

    // Update status periodically
    const interval = setInterval(() => {
      const status = syncService.getSyncStatus();
      setPendingCount(status.pendingUpdates);
    }, 1000);

    return () => {
      unsubscribe();
      clearInterval(interval);
    };
  }, []);

  const handleRetryConflict = (conflictIndex) => {
    const conflict = conflicts[conflictIndex];
    syncService.retryLocalChanges(conflict.docId);
    setConflicts(prev => prev.filter((_, i) => i !== conflictIndex));
  };

  const handleAcceptServer = (conflictIndex) => {
    const conflict = conflicts[conflictIndex];
    syncService.acceptServerVersion(conflict.docId, conflict.server);
    setConflicts(prev => prev.filter((_, i) => i !== conflictIndex));
  };

  const renderStateIndicator = () => {
    switch (syncState) {
      case 'synced':
        return (
          <div className="sync-state synced">
            <CheckCircle size={16} />
            <span>All synced</span>
          </div>
        );
      
      case 'syncing':
        return (
          <div className="sync-state syncing">
            <RefreshCw size={16} className="spinning" />
            <span>{pendingCount} syncing...</span>
          </div>
        );
      
      case 'conflict':
        return (
          <div className="sync-state conflict">
            <AlertTriangle size={16} />
            <span>{conflicts.length} conflict{conflicts.length !== 1 ? 's' : ''}</span>
            <button 
              className="expand-btn"
              onClick={() => setShowDetails(!showDetails)}
            >
              {showDetails ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
            </button>
          </div>
        );
      
      case 'error':
        return (
          <div className="sync-state error">
            <AlertTriangle size={16} />
            <span>Sync error</span>
          </div>
        );
      
      default:
        return null;
    }
  };

  if (syncState === 'synced' && conflicts.length === 0) {
    return null;
  }

  return (
    <div className="sync-state-manager">
      {renderStateIndicator()}

      {showDetails && conflicts.length > 0 && (
        <div className="conflicts-panel">
          <div className="conflicts-header">
            <h4>Sync Conflicts</h4>
            <p>Choose how to resolve conflicts in your changes</p>
          </div>

          <div className="conflicts-list">
            {conflicts.map((conflict, index) => (
              <div key={index} className="conflict-item">
                <div className="conflict-info">
                  <h5>{conflict.docId}</h5>
                  <p>Conflicting fields: {conflict.conflictingFields.join(', ')}</p>
                </div>

                <div className="conflict-actions">
                  <button
                    className="action-btn retry"
                    onClick={() => handleRetryConflict(index)}
                    title="Keep your changes and retry"
                  >
                    Keep Local
                  </button>
                  <button
                    className="action-btn accept"
                    onClick={() => handleAcceptServer(index)}
                    title="Use server version"
                  >
                    Use Server
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
