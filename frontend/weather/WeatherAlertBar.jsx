import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  FaMapMarkerAlt,
  FaTimes,
} from "react-icons/fa";
import {
  WEATHER_SNAPSHOT_EVENT,
  fetchWeatherByLocation,
  getCurrentPosition,
  getCropWarnings,
  notifyWeatherSnapshotUpdated,
} from "./weatherService";
import { useWeatherStore } from "../stores/weatherStore";
import "./WeatherAlertBar.css";

const REFRESH_INTERVAL_MS = 15 * 60 * 1000;
const ALERT_BAR_ACTIVE_KEY = "agri:alertBarActive";

export default function WeatherAlertBar() {
  const [snapshot, setSnapshot] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [dismissed, setDismissed] = useState(false);
  const selectedCrop = useWeatherStore((state) => state.selectedCrop);

  const cropWarnings = useMemo(
    () => getCropWarnings(snapshot?.alerts || [], selectedCrop),
    [snapshot, selectedCrop]
  );
  const applySnapshot = useCallback((latestSnapshot, shouldBroadcast = true) => {
    setSnapshot(latestSnapshot);
    setError("");
    if (shouldBroadcast) {
      notifyWeatherSnapshotUpdated(latestSnapshot);
    }
  }, []);

  useEffect(() => {
    setDismissed(false);
    try {
      localStorage.setItem(ALERT_BAR_ACTIVE_KEY, "true");
    } catch {
      // Ignore storage access failures.
    }

    const handlePageShow = () => setDismissed(false);
    window.addEventListener("pageshow", handlePageShow);
    return () => {
      window.removeEventListener("pageshow", handlePageShow);
      try {
        localStorage.removeItem(ALERT_BAR_ACTIVE_KEY);
      } catch {
        // Ignore storage access failures.
      }
    };
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") {
      return undefined;
    }

    const handleExternalSnapshot = (event) => {
      const latestSnapshot = event.detail;
      if (!latestSnapshot?.location) {
        return;
      }
      setSnapshot((currentSnapshot) => {
        if (
          currentSnapshot?.fetchedAt === latestSnapshot.fetchedAt &&
          currentSnapshot?.location?.source === latestSnapshot.location?.source
        ) {
          return currentSnapshot;
        }
        return latestSnapshot;
      });
      setError("");
    };

    window.addEventListener(WEATHER_SNAPSHOT_EVENT, handleExternalSnapshot);
    return () => {
      window.removeEventListener(WEATHER_SNAPSHOT_EVENT, handleExternalSnapshot);
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    let refreshIntervalId = 0;

    const updateFromLocation = async (location, errorMessage) => {
      try {
        const latest = await fetchWeatherByLocation(location);
        if (!cancelled) {
          applySnapshot(latest);
        }
      } catch (refreshError) {
        if (!cancelled) {
          setError(refreshError.message || errorMessage);
        }
      }
    };

    const initializeWeather = async () => {
      setLoading(true);

      try {
        const preciseLocation = await getCurrentPosition();
        await updateFromLocation(preciseLocation, "Unable to refresh weather alerts.");
      } catch (error) {
        if (!cancelled) {
          setError(error?.message || "Unable to load weather alerts. Allow GPS access to receive real-time farm weather alerts.");
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };

    const refreshSnapshot = async () => {
      if (!snapshot?.location) {
        return;
      }
      await updateFromLocation(snapshot.location, "Unable to refresh weather alerts.");
    };

    void initializeWeather();
    refreshIntervalId = window.setInterval(() => {
      void refreshSnapshot();
    }, REFRESH_INTERVAL_MS);

    return () => {
      cancelled = true;
      window.clearInterval(refreshIntervalId);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [applySnapshot]);

  const dismissBar = () => {
    setDismissed(true);
    try {
      localStorage.removeItem(ALERT_BAR_ACTIVE_KEY);
    } catch {
      // Ignore storage access failures.
    }
  };

  if (dismissed) {
    return null;
  }

  const topAlert = snapshot?.alerts?.[0];
  const activeCropWarning = cropWarnings[0];
  const alertTitle = activeCropWarning?.title || topAlert?.title || (loading ? "Checking local weather alerts" : "Weather alert update");
  const alertMessage = topAlert
    ? `${snapshot?.location?.city || "Your area"}: ${activeCropWarning?.message || topAlert.message}`
    : error || "Allow location access to receive real-time farm weather alerts.";
  const roundedTemperature = Math.round(snapshot?.current?.temperature_2m || 0);
  const weatherSummary = snapshot?.summary || "Current conditions";

  return (
    <section
      className={`weather-alert-bar ${topAlert ? `severity-${topAlert.severity}` : "severity-info"}`}
      aria-live="polite"
    >
      <div className="weather-alert-bar__content">
        <div className="weather-alert-bar__left">
<span className="weather-alert-bar__icon" aria-hidden="true">
            <FaMapMarkerAlt />
          </span>

          <div className="weather-alert-bar__text">
            <strong>{alertTitle}</strong>
            <span>{alertMessage}</span>
          </div>
        </div>

        {snapshot && (
          <div className="weather-alert-bar__temp-display">
            <strong className="weather-alert-bar__temp-value">
              {roundedTemperature}
              {snapshot?.units?.temperature_2m || "°C"}
            </strong>
            <span className="weather-alert-bar__temp-summary">{weatherSummary}</span>
          </div>
        )}

        <button className="weather-alert-bar__dismiss" onClick={dismissBar} aria-label="Dismiss alerts">
          <FaTimes />
        </button>
      </div>
    </section>
  );
}
