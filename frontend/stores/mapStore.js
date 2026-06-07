import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export const useMapStore = create(
  persist(
    (set, get) => ({
  // User location
  userLocation: null,
  setUserLocation: (location) => set({ userLocation: location }),

  // Map center and zoom
  mapCenter: [20.5937, 78.9629], // Default to India center
  mapZoom: 5,
  setMapCenter: (center) => set({ mapCenter: center }),
  setMapZoom: (zoom) => set({ mapZoom: zoom }),

  // Weather data points (markers)
  weatherMarkers: [],
  setWeatherMarkers: (markers) => set({ weatherMarkers: markers }),
  addWeatherMarker: (marker) =>
    set((state) => ({
      weatherMarkers: [
        ...state.weatherMarkers.filter((m) => m.id !== marker.id),
        marker,
      ],
    })),

  // Crop/Field data points (markers)
  cropMarkers: [],
  setCropMarkers: (markers) => set({ cropMarkers: markers }),
  addCropMarker: (marker) =>
    set((state) => ({
      cropMarkers: [
        ...state.cropMarkers.filter((m) => m.id !== marker.id),
        marker,
      ],
    })),

  // Alert zones (areas with weather alerts)
  alertZones: [],
  setAlertZones: (zones) => set({ alertZones: zones }),

  // Selected marker for details
  selectedMarker: null,
  setSelectedMarker: (marker) => set({ selectedMarker: marker }),

  // Loading and error states
  mapLoading: false,
  setMapLoading: (loading) => set({ mapLoading: loading }),

  mapError: null,
  setMapError: (error) => set({ mapError: error }),

  // Map type (satellite, terrain, etc.)
  mapType: 'default',
  setMapType: (type) => set({ mapType: type }),

  // Show/hide layers
  showWeatherLayer: true,
  setShowWeatherLayer: (show) => set({ showWeatherLayer: show }),

  showCropLayer: true,
  setShowCropLayer: (show) => set({ showCropLayer: show }),

  showAlertLayer: true,
  setShowAlertLayer: (show) => set({ showAlertLayer: show }),

  // Fetch user location
  fetchUserLocation: () => {
    set({ mapLoading: true, mapError: null });
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const { latitude, longitude } = position.coords;
          set({
            userLocation: [latitude, longitude],
            mapCenter: [latitude, longitude],
            mapZoom: 12,
            mapLoading: false,
          });
        },
        (error) => {
          console.error('Error getting user location:', error);
          set({
            mapError: 'Unable to access user location',
            mapLoading: false,
          });
        }
      );
    } else {
      set({
        mapError: 'Geolocation is not supported',
        mapLoading: false,
      });
    }
  },

  validateMapState: () => {
    const state = get();

    return (
      Array.isArray(state.mapCenter) &&
      state.mapCenter.length === 2 &&
      typeof state.mapZoom === 'number'
    );
  },

  recoverMapState: () => {
    try {
      const saved = localStorage.getItem('map-store');
      if (!saved) return false;

      const parsed = JSON.parse(saved);

      if (!parsed?.state) {
        return false;
      }

      if (!get().validateMapState()) {
        return false;
      }

      set(parsed.state);
      return true;
    } catch (error) {
      console.error('State recovery failed:', error);
      return false;
    }
  },

  // Reset store
  resetMapStore: () =>
    set({
      userLocation: null,
      mapCenter: [20.5937, 78.9629],
      mapZoom: 5,
      weatherMarkers: [],
      cropMarkers: [],
      alertZones: [],
      selectedMarker: null,
      mapLoading: false,
      mapError: null,
      mapType: 'default',
      showWeatherLayer: true,
      showCropLayer: true,
      showAlertLayer: true,
    }),
  }),
  {
    name: 'map-store',
    partialize: (state) => ({
      userLocation: state.userLocation,
      mapCenter: state.mapCenter,
      mapZoom: state.mapZoom,
      mapType: state.mapType,
      showWeatherLayer: state.showWeatherLayer,
      showCropLayer: state.showCropLayer,
      showAlertLayer: state.showAlertLayer,
    }),
  }
  )
  );