import React, { useEffect, useMemo } from "react";
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
  searchLocationByName,
  getWeatherLabel,
} from "./weatherService";
import { useWeatherManagement } from "../hooks/useWeatherManagement";
import LastUpdated from "../LastUpdated";

const SENT_NOTIFICATION_KEY = "agri:weatherNotificationSignature";

/* 🌦️ Weather Icon Helper */
const getWeatherIcon = (summary = "") => {
  const s = summary.toLowerCase();
  if (s.includes("rain")) return "🌧️";
  if (s.includes("cloud")) return "☁️";
  if (s.includes("storm")) return "⛈️";
  if (s.includes("sun")) return "☀️";
  return "🌤️";
};

function formatSeverity(severity) {
  return severity.charAt(0).toUpperCase() + severity.slice(1);
}

export default function WeatherCard({
  onClose,
  embedded = false,
  title = "Hyperlocal Weather Intelligence",
  subtitle = "AI-powered alerts, crop advisories & farming insights.",
}) {
  const cropOptions = getAvailableCrops();

  const {
    snapshot,
    selectedCrop,
    setSelectedCrop,
    searchQuery,
    setSearchQuery,
    weatherError,
    setWeatherError,
    weatherLoading,
    loadWeather,
    notificationPermission,
    requestNotificationPermission,
  } = useWeatherManagement();

  const cropWarnings = useMemo(
    () => getCropWarnings(snapshot?.alerts || [], selectedCrop),
    [snapshot, selectedCrop]
  );

  /* 🌾 Smart Farming Advice Engine */
  const getFarmingAdvice = () => {
    if (!snapshot) return [];

    const advice = [];
    const temp = snapshot.current?.temperature_2m;
    const humidity = snapshot.current?.relative_humidity_2m;
    const wind = snapshot.current?.wind_speed_10m;

    if (temp > 35) advice.push("🌡️ Heat stress: Water crops early morning/evening.");
    if (humidity > 80) advice.push("💧 High humidity: Watch for fungal infections.");
    if (wind > 25) advice.push("🌬️ Strong winds: Avoid spraying pesticides.");
    if (snapshot.alerts?.some(a => a.type === "rain"))
      advice.push("🌧️ Rain alert: Delay irrigation & fertilizer use.");

    return advice;
  };

  /* 📡 Auto location load */
  useEffect(() => {
    if (!embedded || snapshot) return;
    handleUseMyLocation();
  }, []);

  /* 🔔 Smart Notifications */
  useEffect(() => {
    if (!snapshot?.alerts?.length || notificationPermission !== "granted") return;

    const topAlert = snapshot.alerts[0];
    if (topAlert.severity === "info") return;

    const signature = `${snapshot.location?.name}-${topAlert.type}-${topAlert.severity}`;
    const lastSent = localStorage.getItem(SENT_NOTIFICATION_KEY);

    if (lastSent === signature) return;

    const warning = cropWarnings[0]?.message || topAlert.message;

    const notification = new Notification(`⚠️ ${topAlert.title}`, {
      body: `${warning}\nTake action immediately.`,
      tag: signature,
    });

    notification.onclick = () => window.focus();
    localStorage.setItem(SENT_NOTIFICATION_KEY, signature);
  }, [snapshot, cropWarnings, notificationPermission]);

  const handleSearch = async () => {
    if (!searchQuery.trim()) {
      setWeatherError("Enter a city or district name first.");
      return;
    }

    await loadWeather(async () => {
      const location = await searchLocationByName(searchQuery);
      return fetchWeatherByLocation(location);
    });
  };

  const handleUseMyLocation = async () => {
    await loadWeather(async () => {
      const location = await getCurrentPosition();
      return fetchWeatherByLocation(location);
    });
  };

  const topAlert = snapshot?.alerts?.[0];
  const units = snapshot?.units || {};
  const locationLabel = snapshot?.location?.name || "Set your farm location";

  // In embedded mode, hide controls and panels - just show summary
  const isCompact = embedded;

  return (
    <div
      className={`weather-card ${embedded ? "weather-card--embedded" : ""}`}
      onClick={(e) => {
        if (e.target.closest('button, input, select, textarea, a')) {
          e.stopPropagation();
        }
      }}
    >

      {/* CLOSE */}
      {!embedded && onClose && (
        <button className="weather-card__close-btn" onClick={onClose}>×</button>
      )}

      {/* HEADER */}
      {isCompact ? (
        <div className="weather-card__compact-header">
          <span className="weather-card__compact-location">
            <FaMapMarkerAlt /> {locationLabel}
          </span>
          {snapshot?.fetchedAt && (
            <LastUpdated timestamp={snapshot.fetchedAt} />
          )}
        </div>
      ) : (
        <div className="weather-card__header">
          <span className="weather-card__eyebrow">🌾 Real-time farm intelligence</span>
          <h2>{title}</h2>
          <p className="subtitle">{subtitle}</p>
          {snapshot?.fetchedAt && (
            <LastUpdated timestamp={snapshot.fetchedAt} />
          )}
        </div>
      )}

      {/* CONTROLS - Hidden in compact mode */}
      {!isCompact && (
        <div className="weather-card__controls">

        <div className="input-group">
          <input
            type="text"
            placeholder="Search village / district"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
          <button onClick={handleSearch} disabled={weatherLoading}>
            <FaSearch /> {weatherLoading ? "Loading..." : "Search"}
          </button>
        </div>

        <div className="weather-card__actions">
          <button onClick={handleUseMyLocation}>
            <FaCrosshairs /> Use My Location
          </button>

          <button onClick={requestNotificationPermission}>
            <FaBell />
            {notificationPermission === "granted"
              ? "Alerts ON"
              : "Enable Alerts"}
          </button>
        </div>

        <div className="weather-card__crop-selector">
          <label>Crop</label>
          <select
            value={selectedCrop}
            onChange={(e) => setSelectedCrop(e.target.value)}
          >
            {cropOptions.map((c) => (
              <option key={c.value} value={c.value}>
                {c.label}
              </option>
            ))}
          </select>
         </div>
       </div>
       )}

       {weatherError && <p className="error">{weatherError}</p>}

      {/* MAIN DATA */}
      {snapshot ? (
        <>
          {/* SUMMARY */}
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

                <p className="weather-summary__status">
                  {getWeatherIcon(snapshot.summary)} {snapshot.summary}
                </p>
              </div>

              <div className={`weather-alert-pill severity-${topAlert?.severity}`}>
                {topAlert?.title || "Stable weather"}
              </div>
            </div>

            {/* METRICS */}
            <div className="weather-metrics">
              <div><FaTemperatureHigh /> Feels {snapshot.current?.apparent_temperature}°</div>
              <div><FaTint /> Humidity {snapshot.current?.relative_humidity_2m}%</div>
              <div><FaWind /> Wind {snapshot.current?.wind_speed_10m} km/h</div>
            </div>
          </div>

           {/* FORECAST */}
          <div className="forecast-strip">
            {snapshot.daily && snapshot.daily.weather_code && snapshot.daily.weather_code.length > 0 ? (
              snapshot.daily.weather_code.slice(0, 5).map((_, i) => {
                const date = new Date();
                date.setDate(date.getDate() + i);
                const dayName = i === 0 ? 'Today' : i === 1 ? 'Tomorrow' : date.toLocaleDateString('en-US', { weekday: 'short' });

                return (
                  <div key={i} className="forecast-day">
                    <p>{dayName}</p>
                    <p>{Math.round(snapshot.daily.temperature_2m_max[i])}° / {Math.round(snapshot.daily.temperature_2m_min[i])}°</p>
                    <p>{getWeatherIcon(getWeatherLabel(snapshot.daily.weather_code[i]))}</p>
                  </div>
                );
              })
            ) : null}
          </div>

          {/* PANELS - Hidden in compact mode */}
          {!isCompact && (
            <div className="weather-panels">

            {/* ALERTS */}
            <section className="weather-panel">
              <h3>⚠️ Extreme Alerts</h3>
              {snapshot.alerts.map((a) => (
                <div key={a.title} className={`alert-item severity-${a.severity}`}>
                  <h4>{a.title}</h4>
                  <p>{a.message}</p>
                </div>
              ))}
            </section>

            {/* CROP WARNINGS */}
            <section className="weather-panel">
              <h3>🌾 Crop Warnings</h3>

              {cropWarnings.length ? cropWarnings.map((w) => (
                <div key={w.message} className="alert-item severity-info">
                  <h4>{w.title}</h4>
                  <p>{w.message}</p>
                </div>
              )) : (
                <p>No crop risk detected</p>
              )}
            </section>

            {/* SMART ADVICE */}
            <section className="weather-panel">
              <h3>🧠 Smart Farming Advice</h3>

              {getFarmingAdvice().map((tip, i) => (
                <div key={i} className="alert-item severity-info">
                  <p>{tip}</p>
                </div>
              ))}
             </section>

            </div>
          )}
        </>
      ) : (
        <div className="weather-empty-state">
          <h3>🌾 Live farm intelligence ready</h3>
          <p>Search your location to unlock AI farming insights.</p>
        </div>
      )}
    </div>
  );
}