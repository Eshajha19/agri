import { create } from 'zustand';
import { persist } from 'zustand/middleware';

const STORE_VERSION = 1;

const DEFAULT_MAP_STATE = {
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
  lastRecoveryTime: null,
};

export const useMapStore = create(
  persist(
    (set, get) => ({
      ...DEFAULT_MAP_STATE,

      // User location
      setUserLocation: (location) =>
        set({ userLocation: location }),

      // Map center and zoom
      setMapCenter: (center) =>
        set({ mapCenter: center }),

      setMapZoom: (zoom) =>
        set({ mapZoom: zoom }),

      // Weather markers
      setWeatherMarkers: (markers) =>
        set({ weatherMarkers: markers }),

      addWeatherMarker: (marker) =>
        set((state) => ({
          weatherMarkers: [
            ...state.weatherMarkers.filter(
              (m) => m.id !== marker.id
            ),
            marker,
          ],
        })),

      // Crop markers
      setCropMarkers: (markers) =>
        set({ cropMarkers: markers }),

      addCropMarker: (marker) =>
        set((state) => ({
          cropMarkers: [
            ...state.cropMarkers.filter(
              (m) => m.id !== marker.id
            ),
            marker,
          ],
        })),

      // Alert zones
      setAlertZones: (zones) =>
        set({ alertZones: zones }),

      // Selected marker
      setSelectedMarker: (marker) =>
        set({ selectedMarker: marker }),

      // Loading state
      setMapLoading: (loading) =>
        set({ mapLoading: loading }),

      // Error state
      setMapError: (error) =>
        set({ mapError: error }),

      // Map type
      setMapType: (type) =>
        set({ mapType: type }),

      // Layer visibility
      setShowWeatherLayer: (show) =>
        set({ showWeatherLayer: show }),

      setShowCropLayer: (show) =>
        set({ showCropLayer: show }),

      setShowAlertLayer: (show) =>
        set({ showAlertLayer: show }),

      // Fetch user location
      fetchUserLocation: () => {
        set({
          mapLoading: true,
          mapError: null,
        });

        if (navigator.geolocation) {
          navigator.geolocation.getCurrentPosition(
            (position) => {
              const {
                latitude,
                longitude,
              } = position.coords;

              set({
                userLocation: [
                  latitude,
                  longitude,
                ],
                mapCenter: [
                  latitude,
                  longitude,
                ],
                mapZoom: 12,
                mapLoading: false,
              });
            },
            (error) => {
              console.error(
                'Error getting user location:',
                error
              );

              set({
                mapError:
                  'Unable to access user location',
                mapLoading: false,
              });
            }
          );
        } else {
          set({
            mapError:
              'Geolocation is not supported',
            mapLoading: false,
          });
        }
      },

      // Persistence validation
      validatePersistedState: (state) => {
        if (!state) return false;

        return (
          Array.isArray(state.mapCenter) &&
          state.mapCenter.length === 2 &&
          typeof state.mapCenter[0] ===
            'number' &&
          typeof state.mapCenter[1] ===
            'number' &&
          typeof state.mapZoom ===
            'number' &&
          state.mapZoom >= 1 &&
          state.mapZoom <= 22
        );
      },

      validateMapState: () => {
        const state = get();

        return (
          Array.isArray(state.mapCenter) &&
          state.mapCenter.length === 2 &&
          typeof state.mapZoom ===
            'number'
        );
      },

      // Backup synchronization
      syncPersistedState: () => {
        try {
          const state = get();

          localStorage.setItem(
            'map-store-backup',
            JSON.stringify({
              version: STORE_VERSION,
              timestamp: Date.now(),
              state,
            })
          );

          return true;
        } catch (error) {
          console.error(
            '[MAP_STORE] Backup sync failed:',
            error
          );

          return false;
        }
      },

      // State recovery
      recoverMapState: () => {
        try {
          const saved =
            localStorage.getItem(
              'map-store'
            );

          const backup =
            localStorage.getItem(
              'map-store-backup'
            );

          if (!saved && backup) {
            const parsedBackup =
              JSON.parse(backup);

            if (
              get().validatePersistedState(
                parsedBackup.state
              )
            ) {
              set({
                ...parsedBackup.state,
                lastRecoveryTime:
                  Date.now(),
              });

              console.info(
                '[MAP_STORE] Recovered from backup'
              );

              return true;
            }
          }

          if (!saved) {
            return false;
          }

          const parsed =
            JSON.parse(saved);

          if (!parsed?.state) {
            return false;
          }

          if (
            !get().validatePersistedState(
              parsed.state
            )
          ) {
            console.warn(
              '[MAP_STORE] Invalid persisted state detected'
            );

            localStorage.removeItem(
              'map-store'
            );

            return false;
          }

          set({
            ...parsed.state,
            lastRecoveryTime:
              Date.now(),
          });

          console.info(
            '[MAP_STORE] State restored successfully'
          );

          return true;
        } catch (error) {
          console.error(
            'State recovery failed:',
            error
          );

          return false;
        }
      },

      // State integrity diagnostics
      verifyStateIntegrity: () => {
        const state = get();

        const isValid =
          get().validatePersistedState(
            state
          );

        return {
          valid: isValid,
          timestamp: Date.now(),
          version: STORE_VERSION,
          hasLocation:
            !!state.userLocation,
          weatherMarkers:
            state.weatherMarkers.length,
          cropMarkers:
            state.cropMarkers.length,
        };
      },

      // Reset store
      resetMapStore: () =>
        set({
          ...DEFAULT_MAP_STATE,
        }),
    }),
    {
      name: 'map-store',

      partialize: (state) => ({
        userLocation:
          state.userLocation,
        mapCenter: state.mapCenter,
        mapZoom: state.mapZoom,
        mapType: state.mapType,
        showWeatherLayer:
          state.showWeatherLayer,
        showCropLayer:
          state.showCropLayer,
        showAlertLayer:
          state.showAlertLayer,
        lastRecoveryTime:
          state.lastRecoveryTime,
        storeVersion:
          STORE_VERSION,
      }),
    }
  )
);