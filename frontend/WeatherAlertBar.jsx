import React, { useEffect, useState } from "react";
import { FaBell, FaCloudSunRain, FaMapMarkerAlt, FaTimes } from "react-icons/fa";
import {
  fetchWeatherByLocation,
  getCurrentPosition,
  getStoredWeatherSnapshot,
} from "./weatherService";

const NOTIFICATION_STORAGE_KEY = "agriLiveAlertDismissed";

export default function WeatherAlertBar() {
  const [snapshot, setSnapshot] = useState(() => getStoredWeatherSnapshot());
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [dismissed, setDismissed] = useState(
    () => localStorage.getItem(NOTIFICATION_STORAGE_KEY) === "true"
  );

  useEffect(() => {
    if (!snapshot?.location) {
      return;
    }

    let cancelled = false;

    const refreshSnapshot = async () => {
      setLoading(true);
      try {
        const latest = await fetchWeatherByLocation(snapshot.location);
        if (!cancelled) {
          setSnapshot(latest);
          setError("");
        }
      } catch (refreshError) {
        if (!cancelled) {
          setError(refreshError.message);
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };

    refreshSnapshot();
    const interval = window.setInterval(refreshSnapshot, 15 * 60 * 1000);

    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [snapshot?.location?.latitude, snapshot?.location?.longitude]);

  const enableLiveAlerts = async () => {
    setLoading(true);
    try {
      const location = await getCurrentPosition();
      const latest = await fetchWeatherByLocation(location);
      setSnapshot(latest);
      setError("");
    } catch (locationError) {
      setError(locationError.message || "Unable to enable live alerts.");
    } finally {
      setLoading(false);
    }
  };

  const dismissBar = () => {
    localStorage.setItem(NOTIFICATION_STORAGE_KEY, "true");
    setDismissed(true);
  };

  if (dismissed) {
    return null;
  }

  const topAlert = snapshot?.alerts?.[0];

  return (
    <div className={`alert-bar ${topAlert ? `severity-${topAlert.severity}` : ""}`}>
      <div className="alert-bar__content">
        {topAlert ? (
          <>
            <span className="alert-bar__icon">
              <FaBell />
            </span>
            <div className="alert-bar__text">
              <strong>{topAlert.title}</strong>
              <span>
                {snapshot.location?.city || "Your area"}: {topAlert.message}
              </span>
            </div>
            <div className="alert-bar__meta">
              <FaCloudSunRain />
              <span>{snapshot.summary}</span>
            </div>
          </>
        ) : (
          <>
            <span className="alert-bar__icon">
              <FaMapMarkerAlt />
            </span>
            <div className="alert-bar__text">
              <strong>Enable live farm weather alerts</strong>
              <span>
                Use your location to get hyperlocal rain, frost, and heatwave warnings.
              </span>
            </div>
          </>
        )}

        {!topAlert && (
          <button className="alert-bar__action" onClick={enableLiveAlerts} disabled={loading}>
            {loading ? "Checking..." : "Use My Location"}
          </button>
        )}

        {topAlert && !snapshot.location?.city && (
          <button className="alert-bar__action" onClick={enableLiveAlerts} disabled={loading}>
            Refresh Location
          </button>
        )}

        <button className="alert-bar__dismiss" onClick={dismissBar} aria-label="Dismiss alerts">
          <FaTimes />
        </button>
      </div>

      {error && <div className="alert-bar__error">{error}</div>}
    </div>
  );
}
