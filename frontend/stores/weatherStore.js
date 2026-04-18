import { create } from 'zustand';
import { getStoredWeatherSnapshot } from '../weather/weatherService';

export const useWeatherStore = create((set) => ({
  // Weather snapshot
  snapshot: getStoredWeatherSnapshot(),
  setSnapshot: (snapshot) => set({ snapshot }),

  // Selected crop for warnings
  selectedCrop: 'paddy',
  setSelectedCrop: (crop) => set({ selectedCrop: crop }),

  // Search query
  searchQuery: '',
  setSearchQuery: (query) => set({ searchQuery: query }),

  // Error handling
  weatherError: '',
  setWeatherError: (error) => set({ weatherError: error }),

  // Loading state
  weatherLoading: false,
  setWeatherLoading: (loading) => set({ weatherLoading: loading }),

  // Notification permission
  notificationPermission:
    typeof Notification === 'undefined' ? 'unsupported' : Notification.permission,
  setNotificationPermission: (permission) =>
    set({ notificationPermission: permission }),

  // UI state for showing weather details
  showWeatherDetails: false,
  setShowWeatherDetails: (show) => set({ showWeatherDetails: show }),

  // Reset store
  resetWeatherStore: () =>
    set({
      snapshot: null,
      selectedCrop: 'paddy',
      searchQuery: '',
      weatherError: '',
      weatherLoading: false,
      showWeatherDetails: false,
    }),
}));
