import React from 'react';
import { useSyncStatus } from '../hooks/useSyncStatus';
import { Wifi, WifiOff, CloudUpload, CheckCircle } from 'lucide-react';

export const SyncBadge = () => {
  const { isOnline, pendingCount, isSyncing } = useSyncStatus();

  const dotColor = !isOnline ? '#ef4444' : isSyncing ? '#3b82f6' : pendingCount > 0 ? '#f59e0b' : '#22c55e';
  const Icon = !isOnline ? WifiOff : isSyncing ? CloudUpload : pendingCount > 0 ? CloudUpload : CheckCircle;
  const label = !isOnline 
    ? 'Offline' 
    : isSyncing 
    ? 'Syncing...' 
    : pendingCount > 0 
    ? `${pendingCount} pending` 
    : 'All synced';

  return (
    <div style={{ 
      display: 'inline-flex', 
      alignItems: 'center', 
      gap: 6, 
      padding: '4px 10px', 
      borderRadius: 16, 
      background: '#f3f4f6', 
      fontSize: 12, 
      fontWeight: 500,
      color: '#374151',
    }}>
      <span style={{ 
        width: 8, 
        height: 8, 
        borderRadius: '50%', 
        background: dotColor, 
        display: 'inline-block',
        flexShrink: 0,
      }} />
      <Icon size={13} />
      <span>{label}</span>
    </div>
  );
};