import React, { useEffect, useMemo, useState } from "react";
import {
  FaBell,
  FaCrosshairs,
  FaMapMarkerAlt,
  FaSearch,
  FaTemperatureHigh,
  FaTint,
  FaWind,
} from "react-icons/fa";
import "./WeatherCard.css";
import {
  fetchWeatherByLocation,
  getAvailableCrops,
  getCropWarnings,
  getCurrentPosition,
  getStoredWeatherSnapshot,
  searchLocationByName,
} from "./weatherService";

const SENT_NOTIFICATION_KEY = "agriWeatherNotificationSignature";

function formatSeverity(severity) {
  return severity.charAt(0).toUpperCase() + severity.slice(1);
}

export default function WeatherCard({
  onClose,
  embedded = false,
  title = "Hyperlocal Weather Intelligence",
  subtitle = "Use your farm location for real-time alerts, crop advisories, and next-step recommendations.",
}) {
  const cropOptions = getAvailableCrops();
  const [query, setQuery] = useState("");
  const [crop, setCrop] = useState("paddy");
  const [snapshot, setSnapshot] = useState(() => getStoredWeatherSnapshot());
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [notificationPermission, setNotificationPermission] = useState(
    typeof Notification === "undefined" ? "unsupported" : Notification.permission
  );

  const cropWarnings = useMemo(
    () => getCropWarnings(snapshot?.alerts || [], crop),
    [snapshot, crop]
  );

  useEffect(() => {
    if (!embedded || snapshot) {
      return;
    }

    handleUseMyLocation();
  }, []);

  useEffect(() => {
    if (!snapshot?.alerts?.length || notificationPermission !== "granted") {
      return;
    }

    const topAlert = snapshot.alerts[0];
    if (topAlert.severity === "info") {
      return;
    }

    const signature = `${snapshot.location?.name}-${topAlert.type}-${topAlert.severity}`;
    const lastSent = localStorage.getItem(SENT_NOTIFICATION_KEY);

    if (lastSent === signature) {
      return;
    }

    const warning = cropWarnings[0]?.message || topAlert.message;
    const notification = new Notification(topAlert.title, {
      body: `${snapshot.location?.city || "Your area"}: ${warning}`,
      tag: signature,
    });

    notification.onclick = () => window.focus();
    localStorage.setItem(SENT_NOTIFICATION_KEY, signature);
  }, [snapshot, cropWarnings, notificationPermission]);

  const loadWeather = async (loader) => {
    setLoading(true);
    setError("");

    try {
      const latestSnapshot = await loader();
      setSnapshot(latestSnapshot);
    } catch (loadError) {
      setError(loadError.message || "Unable to load weather data.");
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = async () => {
    if (!query.trim()) {
      setError("Enter a city or district name first.");
      return;
    }

    await loadWeather(async () => {
      const location = await searchLocationByName(query);
      return fetchWeatherByLocation(location);
    });
  };

  const handleUseMyLocation = async () => {
    await loadWeather(async () => {
      const location = await getCurrentPosition();
      return fetchWeatherByLocation(location);
    });
  };

  const enableNotifications = async () => {
    if (typeof Notification === "undefined") {
      setNotificationPermission("unsupported");
      return;
    }

    const permission = await Notification.requestPermission();
    setNotificationPermission(permission);
  };

  const topAlert = snapshot?.alerts?.[0];
  const units = snapshot?.units || {};
  const locationLabel = snapshot?.location?.name || "Set your farm location";

  return (
    <div className={`weather-card ${embedded ? "weather-card--embedded" : ""}`}>
      {!embedded && onClose && (
        <button className="weather-card__close-btn" onClick={onClose} aria-label="Close weather panel">
          x
        </button>
      )}

      <div className="weather-card__header">
        <span className="weather-card__eyebrow">Real-time alerts</span>
        <h2>{title}</h2>
        <p className="subtitle">{subtitle}</p>
      </div>

      <div className="weather-card__controls">
        <div className="input-group">
          <input
            type="text"
            placeholder="Search city or district"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                handleSearch();
              }
            }}
          />
          <button onClick={handleSearch} disabled={loading}>
            <FaSearch />
            <span>{loading ? "Loading..." : "Search"}</span>
          </button>
        </div>

        <div className="weather-card__actions">
          <button className="location-btn" onClick={handleUseMyLocation} disabled={loading}>
            <FaCrosshairs />
            <span>Use My Location</span>
          </button>
          <button
            className="notify-btn"
            onClick={enableNotifications}
            disabled={notificationPermission === "granted"}
          >
            <FaBell />
            <span>
              {notificationPermission === "granted"
                ? "Alerts Enabled"
                : notificationPermission === "unsupported"
                ? "Browser Alerts Unavailable"
                : "Enable Push Alerts"}
            </span>
          </button>
        </div>

        <div className="weather-card__crop-selector">
          <label htmlFor={`crop-${embedded ? "landing" : "modal"}`}>Crop</label>
          <select
            id={`crop-${embedded ? "landing" : "modal"}`}
            value={crop}
            onChange={(event) => setCrop(event.target.value)}
          >
            {cropOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {error && <p className="error">{error}</p>}

      {snapshot ? (
        <>
          <div className="weather-summary">
            <div className="weather-summary__location">
              <FaMapMarkerAlt />
              <span>{locationLabel}</span>
            </div>

            <div className="weather-summary__top">
              <div>
                <div className="weather-summary__temp">
                  {Math.round(snapshot.current?.temperature_2m || 0)}
                  {units.temperature_2m || "°C"}
                </div>
                <p className="weather-summary__status">{snapshot.summary}</p>
              </div>

              <div className={`weather-alert-pill severity-${topAlert?.severity || "info"}`}>
                {topAlert?.title || "Weather update"}
              </div>
            </div>

            <div className="weather-metrics">
              <div className="weather-metric">
                <FaTemperatureHigh />
                <span>
                  Feels like {Math.round(snapshot.current?.apparent_temperature || 0)}
                  {units.apparent_temperature || "°C"}
                </span>
              </div>
              <div className="weather-metric">
                <FaTint />
                <span>
                  Humidity {Math.round(snapshot.current?.relative_humidity_2m || 0)}
                  {units.relative_humidity_2m || "%"}
                </span>
              </div>
              <div className="weather-metric">
                <FaWind />
                <span>
                  Wind {Math.round(snapshot.current?.wind_speed_10m || 0)}
                  {units.wind_speed_10m || " km/h"}
                </span>
              </div>
            </div>
          </div>

          <div className="weather-panels">
            <section className="weather-panel">
              <div className="weather-panel__title-row">
                <h3>Extreme Weather Alerts</h3>
                {topAlert && <span>{formatSeverity(topAlert.severity)}</span>}
              </div>
              <div className="alert-list">
                {snapshot.alerts.map((alert) => (
                  <article key={`${alert.type}-${alert.title}`} className={`alert-item severity-${alert.severity}`}>
                    <h4>{alert.title}</h4>
                    <p>{alert.message}</p>
                  </article>
                ))}
              </div>
            </section>

            <section className="weather-panel">
              <div className="weather-panel__title-row">
                <h3>{cropOptions.find((option) => option.value === crop)?.label} Warnings</h3>
                <span>Field action</span>
              </div>
              <div className="alert-list">
                {cropWarnings.length ? (
                  cropWarnings.map((warning) => (
                    <article
                      key={`${warning.type}-${warning.message}`}
                      className={`alert-item severity-${warning.severity}`}
                    >
                      <h4>{warning.title}</h4>
                      <p>{warning.message}</p>
                    </article>
                  ))
                ) : (
                  <article className="alert-item severity-info">
                    <h4>No crop-specific warning right now</h4>
                    <p>Current conditions look suitable for routine monitoring and scheduled field work.</p>
                  </article>
                )}
              </div>
            </section>
          </div>
        </>
      ) : (
        <div className="weather-empty-state">
          <h3>Live farm weather will appear here</h3>
          <p>Search your district or use device location to unlock hyperlocal alerts and crop guidance.</p>
        </div>
      )}
    </div>
  );
}
