import React, { useCallback, useEffect, useMemo, useState } from "react";
import { toast } from "react-toastify";
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
  getCropWarnings,
  notifyWeatherSnapshotUpdated,
} from "./weatherService";
import { useWeatherStore } from "../stores/weatherStore";
import "./WeatherAlertBar.css";

const REFRESH_INTERVAL_MS = 15 * 60 * 1000;
const ALERT_SIGNATURE_KEY = "agri:weatherAlertSignature";
const ALERT_BAR_ACTIVE_KEY = "agri:alertBarActive";

export default function WeatherAlertBar() {
  const [snapshot, setSnapshot] = useState(() => getStoredWeatherSnapshot());
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
    // Keep the alert bar active while this view is mounted so the compact
    // widget stays hidden behind the live weather banner.
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
    if (!snapshot?.alerts?.length) {
      return;
    }

    const topAlert = snapshot.alerts.find((alert) =>
      ["watch", "warning", "critical"].includes(alert.severity)
    );
    const cropWarning = cropWarnings[0];
    const alertToNotify = cropWarning || topAlert;

    if (!alertToNotify) {
      return;
    }

    const signature = [
      snapshot.location?.name || snapshot.location?.city || "unknown-location",
      selectedCrop || "no-crop",
      alertToNotify.type || "alert",
      alertToNotify.severity || "info",
      alertToNotify.title || "",
      alertToNotify.message || "",
      cropWarning?.message || "",
    ].join("|");

    try {
      if (localStorage.getItem(ALERT_SIGNATURE_KEY) === signature) {
        return;
      }
    } catch {
      // Ignore storage access failures.
    }

    const body = cropWarning?.message || alertToNotify.message;
    const isSevere = ["warning", "critical"].includes(alertToNotify.severity);

    if (typeof Notification !== "undefined" && Notification.permission === "granted" && isSevere) {
      const notification = new Notification(alertToNotify.title, {
        body: `${body}\nTake action immediately.`,
        tag: signature,
      });
      notification.onclick = () => window.focus();
    }

    if (isSevere) {
      toast.warning(`${alertToNotify.title}: ${body}`, {
        position: "top-right",
        autoClose: 5000,
      });
    }

    try {
      localStorage.setItem(ALERT_SIGNATURE_KEY, signature);
    } catch {
      // Ignore storage access failures.
    }
  }, [cropWarnings, selectedCrop, snapshot]);

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
    // eslint-disable-next-line react-hooks/exhaustive-deps
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [applySnapshot, snapshot?.location?.source, snapshot?.location?.latitude, snapshot?.location?.longitude]);

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
            {topAlert ? <FaBell /> : <FaMapMarkerAlt />}
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
