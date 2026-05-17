// frontend/hooks/useFeatureFlag.js
import { useState, useEffect } from 'react';
import { featureFlagService } from '../featureFlags/featureFlagService';
import { auth } from '../lib/firebase';

/**
 * Custom hook to evaluate a feature flag.
 * 
 * @param {string} flagId 
 * @returns {boolean}
 */
export function useFeatureFlag(flagId) {
  const [isEnabled, setIsEnabled] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let mounted = true;

    const evaluate = async () => {
      try {
        await featureFlagService.init();
        
        const user = auth?.currentUser;
        const userId = user ? user.uid : 'anonymous';
        
        const result = featureFlagService.isEnabled(flagId, userId);
        
        if (mounted) {
          setIsEnabled(result);
          setLoading(false);
        }
      } catch (error) {
        if (mounted) setLoading(false);
      }
    };

    evaluate();

    return () => { mounted = false; };
  }, [flagId]);

  return isEnabled;
}
