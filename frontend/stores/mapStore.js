import { create } from 'zustand';

export const useMapStore = create((set) => ({
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
      weatherMarkers: [...state.weatherMarkers, marker],
    })),

  // Crop/Field data points (markers)
  cropMarkers: [],
  setCropMarkers: (markers) => set({ cropMarkers: markers }),
  addCropMarker: (marker) =>
    set((state) => ({
      cropMarkers: [...state.cropMarkers, marker],
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
}));
