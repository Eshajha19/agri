import { useCallback } from 'react';
import { useWeatherStore } from '../stores/weatherStore';
import { useErrorHandler } from './useErrorHandler';

export const useWeatherManagement = () => {
  const { handleSilentError } = useErrorHandler();
  const {
    snapshot,
    setSnapshot,
    selectedCrop,
    setSelectedCrop,
    searchQuery,
    setSearchQuery,
    weatherError,
    setWeatherError,
    weatherLoading,
    setWeatherLoading,
    notificationPermission,
    setNotificationPermission,
    resetWeatherStore,
  } = useWeatherStore();

  const loadWeather = useCallback(
    async (loader) => {
      setWeatherLoading(true);
      setWeatherError('');

      try {
        const latestSnapshot = await loader();
        setSnapshot(latestSnapshot);
      } catch (loadError) {
        setWeatherError(loadError.message || 'Unable to load weather data.');
      } finally {
        setWeatherLoading(false);
      }
    },
    [setWeatherLoading, setWeatherError, setSnapshot]
  );

  const handleSearchQuery = useCallback(
    (query) => {
      setSearchQuery(query);
    },
    [setSearchQuery]
  );

  const handleCropSelection = useCallback(
    (crop) => {
      setSelectedCrop(crop);
    },
    [setSelectedCrop]
  );

  const requestNotificationPermission = useCallback(async () => {
    if (typeof Notification === 'undefined') {
      setNotificationPermission('unsupported');
      return;
    }

    if (Notification.permission === 'granted') {
      setNotificationPermission('granted');
      return;
    }

    if (Notification.permission === 'denied') {
      setNotificationPermission('denied');
      return;
    }

    try {
      const permission = await Notification.requestPermission();
      setNotificationPermission(permission);
    } catch (error) {
      handleSilentError(error, 'notification-permission');
    }
  }, [handleSilentError]);

  return {
    snapshot,
    setSnapshot,
    selectedCrop,
    setSelectedCrop: handleCropSelection,
    searchQuery,
    setSearchQuery: handleSearchQuery,
    weatherError,
    setWeatherError,
    weatherLoading,
    setWeatherLoading,
    loadWeather,
    notificationPermission,
    setNotificationPermission,
    requestNotificationPermission,
    resetWeatherStore,
  };
};
