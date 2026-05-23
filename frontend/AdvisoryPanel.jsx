import React, { useEffect, useMemo, useState } from "react";
import {
  FaBell,
  FaCheckCircle,
  FaCloudRain,
  FaExclamationTriangle,
  FaFlask,
  FaSeedling,
  FaThermometerHalf,
} from "react-icons/fa";
import apiClient from "./lib/apiClient";
import { getStoredWeatherSnapshot, WEATHER_SNAPSHOT_EVENT } from "./weather/weatherService";
import "./AdvisoryPanel.css";

const severityIcon = {
  critical: <FaExclamationTriangle />,
  warning: <FaExclamationTriangle />,
  info: <FaBell />,
  success: <FaCheckCircle />,
};

const categoryIcon = {
  weather: <FaCloudRain />,
  soil: <FaFlask />,
  crop: <FaSeedling />,
  general: <FaCheckCircle />,
};

function sumNext24Hours(values) {
  if (!Array.isArray(values)) return 0;
  return values.slice(0, 24).reduce((total, value) => {
    const number = Number(value);
    return Number.isFinite(number) ? total + number : total;
  }, 0);
}

function maxNext24Hours(values) {
  if (!Array.isArray(values)) return 0;
  return values.slice(0, 24).reduce((max, value) => {
    const number = Number(value);
    return Number.isFinite(number) ? Math.max(max, number) : max;
  }, 0);
}

function buildWeatherPayload(snapshot) {
  if (!snapshot?.current) return {};

  return {
    temperature: snapshot.current.temperature_2m,
    humidity: snapshot.current.relative_humidity_2m,
    rainfall: snapshot.current.rain || snapshot.current.precipitation || 0,
    rainfall_next_24h:
      sumNext24Hours(snapshot.hourly?.precipitation) ||
      Number(snapshot.daily?.precipitation_sum?.[0] || 0),
    rain_probability: maxNext24Hours(snapshot.hourly?.precipitation_probability),
  };
}

function buildSoilPayload(userData) {
  const soil = userData?.soilData || userData?.soilParameters || {};
  const payload = {
    nitrogen: soil.nitrogen ?? userData?.nitrogen,
    phosphorus: soil.phosphorus ?? userData?.phosphorus,
    potassium: soil.potassium ?? userData?.potassium,
    moisture: soil.moisture ?? soil.soil_moisture ?? userData?.soilMoisture,
    ph: soil.ph ?? soil.soilPH ?? userData?.soilPH,
  };

  return Object.fromEntries(
    Object.entries(payload).filter(([, value]) => value !== undefined && value !== null && value !== "")
  );
}

export default function AdvisoryPanel({ userData }) {
  const [weatherSnapshot, setWeatherSnapshot] = useState(() => getStoredWeatherSnapshot());
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const requestPayload = useMemo(() => ({
    crop_type: userData?.cropType || "",
    weather: buildWeatherPayload(weatherSnapshot),
    soil: buildSoilPayload(userData),
    user_id: userData?.uid || userData?.id,
    store_alerts: Boolean(userData?.uid || userData?.id),
  }), [userData, weatherSnapshot]);

  useEffect(() => {
    const handleWeatherUpdate = (event) => {
      setWeatherSnapshot(event.detail || getStoredWeatherSnapshot());
    };

    window.addEventListener(WEATHER_SNAPSHOT_EVENT, handleWeatherUpdate);
    return () => window.removeEventListener(WEATHER_SNAPSHOT_EVENT, handleWeatherUpdate);
  }, []);

  useEffect(() => {
    let cancelled = false;

    const loadAdvisories = async () => {
      setLoading(true);
      setError("");
      try {
        const response = await apiClient.post("/api/advisory", requestPayload, {
          skipGlobalLoader: true,
          logError: false,
          retries: 1,
        });
        if (!cancelled) {
          setAlerts(response.data?.data || []);
        }
      } catch {
        if (!cancelled) {
          setError("Advisories are temporarily unavailable.");
          setAlerts([]);
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };

    loadAdvisories();
    return () => {
      cancelled = true;
    };
  }, [requestPayload]);

  return (
    <div className="advisory-panel dashboard-section-card">
      <div className="section-card-header">
        <h2><FaThermometerHalf /> Farmer Advisory</h2>
        <span className="section-badge">Rule Based</span>
      </div>

      {loading ? (
        <div className="advisory-state">Preparing field actions...</div>
      ) : error ? (
        <div className="advisory-state advisory-state-error">{error}</div>
      ) : (
        <div className="advisory-list">
          {alerts.slice(0, 5).map((alert) => (
            <article className={`advisory-card advisory-${alert.severity || alert.type}`} key={`${alert.title}-${alert.action}`}>
              <div className="advisory-card-icon" aria-hidden="true">
                {categoryIcon[alert.category] || severityIcon[alert.severity] || <FaBell />}
              </div>
              <div className="advisory-card-content">
                <div className="advisory-card-header">
                  <h3>{alert.title}</h3>
                  <span>{alert.category || "advisory"}</span>
                </div>
                <p>{alert.message}</p>
                <strong>{alert.action}</strong>
              </div>
            </article>
          ))}
        </div>
      )}
    </div>
  );
}
