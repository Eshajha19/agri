import { useEffect, useState } from 'react';
import { useOfflineStore } from '../store/offlineStore';

export const useSyncStatus = () => {
  const [healthSync, setHealthSync] = useState(null);
  
  const pendingCount = useOfflineStore(s => s.pendingCount);
  const isOnline = useOfflineStore(s => s.isOnline);
  const isSyncing = useOfflineStore(s => s.isSyncing);
  const lastSyncAt = useOfflineStore(s => s.lastSyncAt);
  const init = useOfflineStore(s => s.init);

  useEffect(() => {
    let cleanup;
    
    const setup = async () => {
      cleanup = await init();
    };
    setup();
    
    const interval = setInterval(async () => {
      if (!navigator.onLine) return;
      try {
        const res = await fetch('/health/sync');
        if (res.ok) {
          const data = await res.json();
          setHealthSync(data.sync);
        }
      } catch {
        // ignore — offline or server down
      }
    }, 30000);
    
    return () => {
      clearInterval(interval);
      cleanup?.();
    };
  }, [init]);

  return {
    isOnline,
    pendingCount,
    isSyncing,
    lastSyncAt,
    backendPending: healthSync?.pending_sync ?? null,
    backendFailed: healthSync?.failed_permanently ?? null,
  };
};