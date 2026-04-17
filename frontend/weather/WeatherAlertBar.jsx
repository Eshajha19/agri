import React, { useCallback, useEffect, useState } from "react";
import {
  FaBell,
  FaMapMarkerAlt,
  FaTimes,
} from "react-icons/fa";
import {
  WEATHER_SNAPSHOT_EVENT,
  fetchWeatherByIP,
  fetchWeatherByLocation,
  getCurrentPosition,
  getStoredWeatherSnapshot,
  notifyWeatherSnapshotUpdated,
} from "./weatherService";
import "./WeatherAlertBar.css";

const REFRESH_INTERVAL_MS = 15 * 60 * 1000;
const LEGACY_DISMISS_KEY = "agriLiveAlertDismissed";

export default function WeatherAlertBar() {
  const [snapshot, setSnapshot] = useState(() => getStoredWeatherSnapshot());
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [dismissed, setDismissed] = useState(false);
  const applySnapshot = useCallback((latestSnapshot, shouldBroadcast = true) => {
    setSnapshot(latestSnapshot);
    setError("");
    if (shouldBroadcast) {
      notifyWeatherSnapshotUpdated(latestSnapshot);
    }
  }, []);

  useEffect(() => {
    // Force-show after refresh and clear legacy persisted dismiss flag.
    setDismissed(false);
    try {
      localStorage.removeItem(LEGACY_DISMISS_KEY);
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
      const ipPromise = fetchWeatherByIP()
        .then((ipSnapshot) => ({ ipSnapshot, ipError: null }))
        .catch((ipError) => ({ ipSnapshot: null, ipError }));

      try {
        let geolocationDenied = false;

        if (typeof navigator !== "undefined" && navigator.permissions?.query) {
          try {
            const permissionStatus = await navigator.permissions.query({
              name: "geolocation",
            });
            geolocationDenied = permissionStatus.state === "denied";
          } catch {
            // Ignore unsupported permission-query environments.
          }
        }

        if (geolocationDenied) {
          throw new Error("Geolocation permission denied.");
        }

        const preciseLocation = await getCurrentPosition();
        await updateFromLocation(preciseLocation, "Unable to refresh weather alerts.");
      } catch {
        const { ipSnapshot, ipError } = await ipPromise;

        if (ipSnapshot) {
          if (!cancelled) {
            applySnapshot(ipSnapshot);
          }
        } else {
          if (!cancelled) {
            setError(ipError.message || "Unable to load weather alerts.");
          }
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

    if (!snapshot?.location) {
      void initializeWeather();
      return () => {
        cancelled = true;
      };
    }

    void refreshSnapshot();
    refreshIntervalId = window.setInterval(() => {
      void refreshSnapshot();
    }, REFRESH_INTERVAL_MS);

    return () => {
      cancelled = true;
      window.clearInterval(refreshIntervalId);
    };
  }, [applySnapshot, snapshot?.location?.latitude, snapshot?.location?.longitude, snapshot?.location?.source]);

  useEffect(() => {
    if (!snapshot?.location || snapshot.location.source === "gps") {
      return;
    }
    if (typeof navigator === "undefined" || !navigator.permissions?.query) {
      return;
    }

    let cancelled = false;
    let removePermissionListener = () => {};

    const syncGpsLocation = async () => {
      try {
        const preciseLocation = await getCurrentPosition();
        const latest = await fetchWeatherByLocation(preciseLocation);
        if (!cancelled) {
          applySnapshot(latest);
        }
      } catch {
        // Ignore silent GPS-sync failures in background mode.
      }
    };

    const monitorPermission = async () => {
      try {
        const permissionStatus = await navigator.permissions.query({
          name: "geolocation",
        });

        if (permissionStatus.state === "granted") {
          await syncGpsLocation();
        }

        const onPermissionChange = () => {
          if (permissionStatus.state === "granted") {
            void syncGpsLocation();
          }
        };

        permissionStatus.addEventListener?.("change", onPermissionChange);
        removePermissionListener = () =>
          permissionStatus.removeEventListener?.("change", onPermissionChange);
      } catch {
        // Ignore unsupported permission-query environments.
      }
    };

    void monitorPermission();

    return () => {
      cancelled = true;
      removePermissionListener();
    };
  }, [applySnapshot, snapshot?.location?.source, snapshot?.location?.latitude, snapshot?.location?.longitude]);

  const dismissBar = () => {
    setDismissed(true);
  };

  if (dismissed) {
    return null;
  }

  const topAlert = snapshot?.alerts?.[0];
  const alertTitle = topAlert?.title || (loading ? "Checking local weather alerts" : "Weather alert update");
  const alertMessage = topAlert
    ? `${snapshot?.location?.city || "Your area"}: ${topAlert.message}`
    : error || "Allow location access to receive real-time farm weather alerts.";

  return (
    <section
      className={`weather-alert-bar ${topAlert ? `severity-${topAlert.severity}` : "severity-info"}`}
      aria-live="polite"
    >
      <div className="weather-alert-bar__content">
        <div className="weather-alert-bar__left">
          <span className="weather-alert-bar__icon" aria-hidden="true">
            {topAlert ? <FaBell /> : <FaMapMarkerAlt />}
          </span>

          <div className="weather-alert-bar__text">
            <strong>{alertTitle}</strong>
            <span>{alertMessage}</span>
          </div>
        </div>

        <button className="weather-alert-bar__dismiss" onClick={dismissBar} aria-label="Dismiss alerts">
          <FaTimes />
        </button>
      </div>
    </section>
  );
}
