import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export const usePerformanceStore = create(
  persist(
    (set, get) => ({
      liteMode: false,
      autoDetected: false,
      
      setLiteMode: (enabled) => set({ liteMode: enabled, autoDetected: false }),
      
      detectAndSetLiteMode: () => {
        try {
          // Skip if user has already manually set a preference
          const storedRaw = localStorage.getItem('performance-storage');
          if (storedRaw) {
            const stored = JSON.parse(storedRaw);
            if (stored && stored.state && stored.state.autoDetected === false) return;
          }

          let shouldBeLite = false;

          // 1. Check Network Speed (Network Information API)
          if (navigator && navigator.connection) {
            const type = navigator.connection.effectiveType;
            if (['slow-2g', '2g', '3g'].includes(type)) {
              shouldBeLite = true;
            }
          }

          // 2. Check Device Memory (RAM)
          if (navigator && navigator.deviceMemory && navigator.deviceMemory < 4) {
            shouldBeLite = true;
          }

          // 3. Check Reduced Motion Preference
          if (window && window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
            shouldBeLite = true;
          }

          if (shouldBeLite) {
            set({ liteMode: true, autoDetected: true });
          }
        } catch (err) {
          console.warn("Performance detection failed (non-critical):", err);
        }
      }
    }),
    {
      name: 'performance-storage',
    }
  )
);
