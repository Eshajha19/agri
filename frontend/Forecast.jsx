import React, { useState, useEffect } from "react";
import { CloudSun, Sun, CloudRain, Cloud, CloudLightning, CloudSnow, Wind, Droplets, MapPin, RefreshCw, AlertTriangle } from "lucide-react";
import {
  fetchWeatherByLocation,
  getStoredWeatherSnapshot,
  getWeatherLabel,
  WEATHER_SNAPSHOT_EVENT,
} from "./weather/weatherService";

// ---------------------------------------------------------------------------
// WMO weather code → icon + colour mapping
// Open-Meteo uses WMO 4677 codes in the `daily.weather_code` field.
// ---------------------------------------------------------------------------
function weatherIcon(code, size = 24) {
  if (code === 0 || code === 1) return <Sun size={size} color="#f59e0b" />;
  if (code === 2 || code === 3) return <CloudSun size={size} color="#fb923c" />;
  if (code >= 51 && code <= 67) return <CloudRain size={size} color="#60a5fa" />;
  if (code >= 71 && code <= 77) return <CloudSnow size={size} color="#93c5fd" />;
  if (code >= 80 && code <= 82) return <CloudRain size={size} color="#3b82f6" />;
  if (code >= 85 && code <= 86) return <CloudSnow size={size} color="#93c5fd" />;
  if (code >= 95 && code <= 99) return <CloudLightning size={size} color="#4b5563" />;
  if (code === 45 || code === 48) return <Wind size={size} color="#9ca3af" />;
  return <Cloud size={size} color="#9ca3af" />;
}

// Return the short day name (Mon, Tue …) for a date string like "2025-06-10"
function shortDay(dateStr, index) {
  if (!dateStr) return ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][index % 7];
  const d = new Date(dateStr + "T12:00:00");
  return d.toLocaleDateString("en-US", { weekday: "short" });
}

// Build a per-day agricultural insight based on the dominant condition
function buildInsight(days) {
  if (!days.length) return null;
  const rainDay = days.find((d) => d.code >= 51 && d.code <= 99);
  const hotDay = days.find((d) => d.max >= 38);
  const coldDay = days.find((d) => d.min <= 5);
  if (rainDay)
    return `🌧️ Rain expected around ${rainDay.label}. Consider delaying irrigation and check field drainage.`;
  if (hotDay)
    return `🌡️ High temperatures (~${Math.round(hotDay.max)}°C) forecast around ${hotDay.label}. Irrigate early morning and protect seedlings.`;
  if (coldDay)
    return `❄️ Cold nights (~${Math.round(coldDay.min)}°C) expected around ${coldDay.label}. Protect frost-sensitive crops overnight.`;
  return "✅ Conditions look stable for routine field work this week.";
}

export default function Forecast() {
  const [days, setDays] = useState([]);
  const [locationName, setLocationName] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [fetchedAt, setFetchedAt] = useState(null);

  function applySnapshot(snapshot) {
    if (!snapshot?.daily?.time?.length) return false;
    const { daily, location, fetchedAt: ts } = snapshot;
    const built = daily.time.slice(0, 7).map((date, i) => ({
      label: shortDay(date, i),
      date,
      max: daily.temperature_2m_max?.[i] ?? null,
      min: daily.temperature_2m_min?.[i] ?? null,
      code: daily.weather_code?.[i] ?? 0,
      rain: daily.precipitation_sum?.[i] ?? 0,
    }));
    setDays(built);
    setLocationName(location?.name || null);
    setFetchedAt(ts ? new Date(ts) : null);
    setError(null);
    return true;
  }

  async function loadForecast(forceRefresh = false) {
    setLoading(true);
    setError(null);

    // 1. Try the cached snapshot first (instant render)
    if (!forceRefresh) {
      const cached = getStoredWeatherSnapshot();
      if (applySnapshot(cached)) {
        setLoading(false);
        // Still refresh in the background if the cache is older than 30 min
        const age = cached.fetchedAt ? Date.now() - new Date(cached.fetchedAt).getTime() : Infinity;
        if (age > 30 * 60 * 1000) loadForecast(true);
        return;
      }
    }

    // 2. Try to get the user's location and fetch live data
    try {
      let location = null;

      if (navigator.geolocation) {
        location = await new Promise((resolve) => {
          navigator.geolocation.getCurrentPosition(
            async (pos) => {
              const { latitude, longitude } = pos.coords;
              resolve({ latitude, longitude, source: "gps" });
            },
            () => resolve(null),
            { timeout: 6000, maximumAge: 300000 }
          );
        });
      }

      // Fall back to IP-based location if GPS is unavailable/denied
      if (!location) {
        const { getLocationByIP } = await import("./weather/weatherService");
        location = await getLocationByIP().catch(() => null);
      }

      if (!location) throw new Error("Could not determine your location.");

      const snapshot = await fetchWeatherByLocation(location);
      applySnapshot(snapshot);
    } catch (err) {
      setError(err.message || "Unable to load forecast. Check your connection.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadForecast();

    // Keep in sync when another component (e.g. WeatherAlerts) fetches fresh data
    const handler = (e) => applySnapshot(e.detail);
    window.addEventListener(WEATHER_SNAPSHOT_EVENT, handler);
    return () => window.removeEventListener(WEATHER_SNAPSHOT_EVENT, handler);
  }, []);

  const insight = buildInsight(days);

  return (
    <div className="forecast-modal-content">
      <div style={{ padding: "20px", color: "#1f2937" }}>

        {/* Header */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "16px" }}>
          <h2 style={{ margin: 0, display: "flex", alignItems: "center", gap: "10px", color: "#166534" }}>
            <CloudSun size={28} /> 7-Day Weather Forecast
          </h2>
          <button
            onClick={() => loadForecast(true)}
            disabled={loading}
            title="Refresh forecast"
            style={{
              background: "none", border: "1px solid #d1fae5", borderRadius: "8px",
              padding: "6px 10px", cursor: loading ? "not-allowed" : "pointer",
              color: "#166534", display: "flex", alignItems: "center", gap: "6px",
              fontSize: "0.8rem", opacity: loading ? 0.6 : 1,
            }}
          >
            <RefreshCw size={14} style={{ animation: loading ? "spin 1s linear infinite" : "none" }} />
            {loading ? "Loading…" : "Refresh"}
          </button>
        </div>

        {/* Location badge */}
        {locationName && (
          <div style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "14px", color: "#6b7280", fontSize: "0.85rem" }}>
            <MapPin size={14} />
            <span>{locationName}</span>
            {fetchedAt && (
              <span style={{ marginLeft: "auto", fontSize: "0.75rem" }}>
                Updated {fetchedAt.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
              </span>
            )}
          </div>
        )}

        {/* Error state */}
        {error && (
          <div style={{
            display: "flex", alignItems: "center", gap: "10px",
            padding: "12px 16px", background: "#fef2f2", borderRadius: "10px",
            border: "1px solid #fecaca", color: "#dc2626", marginBottom: "16px", fontSize: "0.9rem",
          }}>
            <AlertTriangle size={18} />
            <span>{error}</span>
          </div>
        )}

        {/* Skeleton while loading with no cached data */}
        {loading && !days.length && (
          <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
            {Array.from({ length: 7 }).map((_, i) => (
              <div key={i} style={{
                height: "52px", borderRadius: "12px",
                background: "linear-gradient(90deg,#f3f4f6 25%,#e5e7eb 50%,#f3f4f6 75%)",
                backgroundSize: "200% 100%", animation: "shimmer 1.4s infinite",
              }} />
            ))}
          </div>
        )}

        {/* Forecast rows */}
        {days.length > 0 && (
          <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
            {days.map((day, index) => (
              <div
                key={index}
                style={{
                  display: "flex", alignItems: "center", justifyContent: "space-between",
                  padding: "12px 20px", background: "rgba(0,0,0,0.03)",
                  borderRadius: "12px", border: "1px solid rgba(0,0,0,0.05)",
                }}
              >
                <div style={{ width: "48px", fontWeight: "600", fontSize: "0.95rem" }}>{day.label}</div>
                <div style={{ display: "flex", alignItems: "center", gap: "10px", flex: 1, justifyContent: "center" }}>
                  {weatherIcon(day.code)}
                  <span style={{ fontSize: "0.85rem", color: "#6b7280" }}>{getWeatherLabel(day.code)}</span>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                  {day.rain > 0 && (
                    <span style={{ display: "flex", alignItems: "center", gap: "3px", fontSize: "0.78rem", color: "#60a5fa" }}>
                      <Droplets size={12} />{day.rain.toFixed(1)}mm
                    </span>
                  )}
                  <div style={{ textAlign: "right", minWidth: "72px" }}>
                    <span style={{ fontWeight: "700", color: "#166534" }}>
                      {day.max !== null ? `${Math.round(day.max)}°` : "—"}
                    </span>
                    <span style={{ color: "#9ca3af", fontSize: "0.85rem", marginLeft: "4px" }}>
                      {day.min !== null ? `${Math.round(day.min)}°` : ""}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Agricultural insight */}
        {insight && days.length > 0 && (
          <div style={{
            marginTop: "18px", padding: "14px 16px", background: "#f0fdf4",
            borderRadius: "10px", border: "1px solid #bbf7d0",
          }}>
            <p style={{ margin: 0, fontSize: "0.85rem", color: "#166534", lineHeight: "1.6" }}>
              <strong>🌱 Agricultural Insight:</strong> {insight}
            </p>
          </div>
        )}

        {/* Data source attribution */}
        <p style={{ margin: "14px 0 0", fontSize: "0.72rem", color: "#9ca3af", textAlign: "center" }}>
          Weather data from{" "}
          <a href="https://open-meteo.com" target="_blank" rel="noopener noreferrer" style={{ color: "#6b7280" }}>
            Open-Meteo
          </a>{" "}
          — free, no API key required
        </p>
      </div>

      <style>{`
        @keyframes shimmer {
          0% { background-position: 200% 0; }
          100% { background-position: -200% 0; }
        }
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}
