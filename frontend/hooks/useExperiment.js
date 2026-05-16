// frontend/hooks/useExperiment.js
import { useState, useEffect } from 'react';
import { experimentService } from '../featureFlags/experimentService';
import { analyticsService } from '../featureFlags/analyticsService';
import { auth } from '../lib/firebase';

/**
 * Custom hook to participate in an A/B experiment.
 * 
 * @param {string} experimentId 
 * @returns {{variant: string, loading: boolean}}
 */
export function useExperiment(experimentId) {
  const [variant, setVariant] = useState('control');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let mounted = true;

    const assign = async () => {
      try {
        const user = auth?.currentUser;
        const userId = user ? user.uid : 'anonymous';
        
        const assignedVariant = await experimentService.getVariant(experimentId, userId);
        
        if (mounted) {
          setVariant(assignedVariant);
          setLoading(false);
          
          // Track impression automatically
          analyticsService.trackImpression(experimentId, assignedVariant, userId);
        }
      } catch (error) {
        if (mounted) setLoading(false);
      }
    };

    assign();

    return () => { mounted = false; };
  }, [experimentId]);

  return { variant, loading };
}
