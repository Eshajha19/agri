import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  FaCloudSun,
  FaCrosshairs,
  FaMapMarkerAlt,
  FaTimes,
} from "react-icons/fa";
import {
  WEATHER_SNAPSHOT_EVENT,
  fetchWeatherByLocation,
  getCurrentPosition,
  getStoredWeatherSnapshot,
  notifyWeatherSnapshotUpdated,
} from "./weatherService";
import "./WeatherQuickWidget.css";

const WEATHER_CACHE_KEY = "agriWeatherSnapshot";
const LEGACY_WIDGET_DISMISS_KEY = "agriWeatherWidgetDismissed";
const ALERT_BAR_SHOWN_KEY = "agriAlertBarActive";

export default function WeatherQuickWidget() {
  const [snapshot, setSnapshot] = useState(() => getStoredWeatherSnapshot());
  const [dismissed, setDismissed] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    // Always show on refresh
    setDismissed(false);

    try {
      localStorage.removeItem(LEGACY_WIDGET_DISMISS_KEY);
    } catch {
      // ignore
    }

    const handlePageShow = () => setDismissed(false);
    window.addEventListener("pageshow", handlePageShow);

    return () => {
      window.removeEventListener("pageshow", handlePageShow);
    };
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;

    const handleWeatherUpdate = (event) => {
      if (!event.detail?.location) return;
      setSnapshot(event.detail);
      setError("");
    };

    const handleStorage = (event) => {
      if (event.key !== WEATHER_CACHE_KEY) return;

      const latestSnapshot = getStoredWeatherSnapshot();
      if (latestSnapshot?.location) {
        setSnapshot(latestSnapshot);
        setError("");
      }
    };

    window.addEventListener(WEATHER_SNAPSHOT_EVENT, handleWeatherUpdate);
    window.addEventListener("storage", handleStorage);

    return () => {
      window.removeEventListener(WEATHER_SNAPSHOT_EVENT, handleWeatherUpdate);
      window.removeEventListener("storage", handleStorage);
    };
  }, []);

  const usePreciseGps = useCallback(async () => {
    setLoading(true);
    setError("");

    try {
      const preciseLocation = await getCurrentPosition();
      const latestSnapshot = await fetchWeatherByLocation(preciseLocation);

      setSnapshot(latestSnapshot);
      notifyWeatherSnapshotUpdated(latestSnapshot);
    } catch (locationError) {
      const permissionDenied =
        locationError?.code === 1 ||
        /denied|permission/i.test(locationError?.message || "");

      setError(
        permissionDenied
          ? "Location access denied. Please allow location and try again."
          : locationError.message || "Unable to fetch GPS weather."
      );
    } finally {
      setLoading(false);
    }
  }, []);

  const dismissWidget = () => {
    setDismissed(true);
  };

  const locationSource = snapshot?.location?.source || "manual";
  const showUseGps = !snapshot || locationSource !== "gps";
  const roundedTemperature = Math.round(
    snapshot?.current?.temperature_2m || 0
  );

  const locationLabel = useMemo(() => {
    return (
      snapshot?.location?.name ||
      snapshot?.location?.city ||
      "Your area"
    );
  }, [snapshot?.location?.city, snapshot?.location?.name]);

  // Only hide if dismissed or alert bar is active
  const isAlertBarActive = localStorage.getItem(ALERT_BAR_SHOWN_KEY);
  if (dismissed || isAlertBarActive) {
    return null;
  }

  return (
    <aside className="weather-quick-widget" aria-live="polite">
      {/* Close Button */}
      <button
        className="weather-quick-widget__dismiss"
        onClick={dismissWidget}
        aria-label="Close weather widget"
      >
        <FaTimes />
      </button>

      {/* Header */}
      <div className="weather-quick-widget__head">
        <span className="weather-quick-widget__eyebrow">
          <FaCloudSun />
          <span>Landing Weather</span>
        </span>
      </div>

      {/* Weather Content */}
      {snapshot ? (
        <>
          <div className="weather-quick-widget__location">
            <FaMapMarkerAlt />
            <span>{locationLabel}</span>
          </div>

          <div className="weather-quick-widget__temp-row highlight">
            <strong>
              {roundedTemperature}
              {snapshot?.units?.temperature_2m || "°C"}
            </strong>
            <span>
              {snapshot?.summary || "Current conditions"}
            </span>
          </div>
        </>
      ) : (
        <p className="weather-quick-widget__placeholder emphasized">
          📍 Enable location to see real-time weather updates
        </p>
      )}

      {/* GPS Button */}
      {showUseGps && (
        <button
          className="weather-quick-widget__gps-btn primary"
          onClick={usePreciseGps}
          disabled={loading}
        >
          <FaCrosshairs />
          <span>
            {loading ? "Fetching weather..." : "Use My Location"}
          </span>
        </button>
      )}

      {/* Error */}
      {error && (
        <p className="weather-quick-widget__error">
          {error}
        </p>
      )}
    </aside>
  );
}