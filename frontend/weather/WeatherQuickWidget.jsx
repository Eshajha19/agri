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

export default function WeatherQuickWidget() {
  const [snapshot, setSnapshot] = useState(() => getStoredWeatherSnapshot());
  const [dismissed, setDismissed] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    // Force-show after refresh and clear legacy persisted dismiss flag.
    setDismissed(false);
    try {
      localStorage.removeItem(LEGACY_WIDGET_DISMISS_KEY);
    } catch {
      // Ignore storage access failures.
    }

    const handlePageShow = () => setDismissed(false);
    window.addEventListener("pageshow", handlePageShow);
    return () => {
      window.removeEventListener("pageshow", handlePageShow);
    };
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") {
      return undefined;
    }

    const handleWeatherUpdate = (event) => {
      if (!event.detail?.location) {
        return;
      }
      setSnapshot(event.detail);
      setError("");
    };

    const handleStorage = (event) => {
      if (event.key !== WEATHER_CACHE_KEY) {
        return;
      }
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
    try {
      const preciseLocation = await getCurrentPosition();
      const latestSnapshot = await fetchWeatherByLocation(preciseLocation);
      setSnapshot(latestSnapshot);
      setError("");
      notifyWeatherSnapshotUpdated(latestSnapshot);
    } catch (locationError) {
      const permissionDenied =
        locationError?.code === 1 ||
        /denied|permission/i.test(locationError?.message || "");
      setError(
        permissionDenied
          ? "Location access denied. Click Fetch GPS again after allowing location in your browser."
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
  const roundedTemperature = Math.round(snapshot?.current?.temperature_2m || 0);

  const locationLabel = useMemo(() => {
    return snapshot?.location?.name || snapshot?.location?.city || "Your area";
  }, [snapshot?.location?.city, snapshot?.location?.name]);

  if (dismissed) {
    return null;
  }

  return (
    <aside className="weather-quick-widget" aria-live="polite">
      <button
        className="weather-quick-widget__dismiss"
        onClick={dismissWidget}
        aria-label="Close weather widget"
      >
        <FaTimes />
      </button>

      <div className="weather-quick-widget__head">
        <span className="weather-quick-widget__eyebrow">
          <FaCloudSun />
          <span>Landing Weather</span>
        </span>
      </div>

      {snapshot ? (
        <>
          <div className="weather-quick-widget__location">
            <FaMapMarkerAlt />
            <span>{locationLabel}</span>
          </div>

          <div className="weather-quick-widget__temp-row">
            <strong>
              {roundedTemperature}
              {snapshot?.units?.temperature_2m || "°C"}
            </strong>
            <span>{snapshot?.summary || "Current conditions"}</span>
          </div>
        </>
      ) : (
        <p className="weather-quick-widget__placeholder">
          Allow location to load local weather for your area.
        </p>
      )}

      {showUseGps && (
        <button className="weather-quick-widget__gps-btn" onClick={usePreciseGps} disabled={loading}>
          <FaCrosshairs />
          <span>{loading ? "Fetching..." : "Fetch GPS"}</span>
        </button>
      )}

      {error && <p className="weather-quick-widget__error">{error}</p>}
    </aside>
  );
}
