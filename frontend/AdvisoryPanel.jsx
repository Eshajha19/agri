import React, { useEffect, useMemo, useState, useRef } from "react";
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
import { generateRecommendations } from "./utils/recommendationEngine";
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

function deriveSeasonFromCalendar() {
  const month = new Date().getMonth() + 1;
  if (month >= 6 && month <= 10) return "Kharif";
  if (month >= 3 && month <= 5) return "Zaid";
  return "Rabi";
}

function buildLocalFallbackAdvisories(userData, weatherSnapshot) {
  const recommendations = generateRecommendations({
    weatherData: weatherSnapshot,
    cropType: userData?.cropType,
    season: userData?.season || deriveSeasonFromCalendar(),
  });

  const mapped = recommendations.map((item) => ({
    severity: item.type === "warning" ? "warning" : item.type === "heat" || item.type === "frost" ? "info" : "success",
    category: item.type === "warning" || item.type === "heat" || item.type === "frost" ? "weather" : item.type === "crop" ? "crop" : "general",
    title: item.title,
    message: item.text,
    action: item.type === "crop"
      ? "Review your crop care plan"
      : item.type === "season"
        ? "Align field work with the season"
        : "Take action now",
  }));

  if (mapped.length > 0) {
    return mapped;
  }

  return [
    {
      severity: "info",
      category: "general",
      title: "Profile-based advisory",
      message: userData?.cropType
        ? `Your ${userData.cropType} profile is saved. Update weather or soil details to unlock more precise advice.`
        : "Complete your profile to get more precise field advisories.",
      action: "Review profile settings",
    },
  ];
}

export default function AdvisoryPanel({ userData }) {
  const [weatherSnapshot, setWeatherSnapshot] = useState(() => getStoredWeatherSnapshot());
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const mountedRef = useRef(true);
  const requestIdRef = useRef(0);

  useEffect(() => {
    mountedRef.current = true;

    return () => {
      mountedRef.current = false;
      requestIdRef.current++;
    };
  }, []);

  const fallbackAdvisories = useMemo(
    () => buildLocalFallbackAdvisories(userData, weatherSnapshot),
    [userData, weatherSnapshot],
  );

  const requestPayload = useMemo(() => ({
    crop_type: userData?.cropType || "",
    weather: buildWeatherPayload(weatherSnapshot),
    soil: buildSoilPayload(userData),
    farm_area: userData?.farmArea || userData?.farmSize || "",
    irrigation_type: userData?.irrigationType || userData?.irrigationMethod || "",
    season: userData?.season || "",
    location: userData?.address || (typeof userData?.location === "string" ? userData.location : ""),
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
    const requestId = ++requestIdRef.current;

    const loadAdvisories = async () => {
      setLoading(true);
      setError("");

      try {
        const response = await apiClient.post(
          "/api/advisory",
          requestPayload,
          {
            skipGlobalLoader: true,
            logError: false,
            retries: 1,
          }
        );

        if (
          !mountedRef.current ||
          requestId !== requestIdRef.current
        ) {
          return;
        }

        const nextAlerts =
          Array.isArray(response.data?.data) &&
          response.data.data.length > 0
            ? response.data.data
            : fallbackAdvisories;

        setAlerts(nextAlerts);
        setError("");
      } catch {
        if (
          mountedRef.current &&
          requestId === requestIdRef.current
        ) {
          setAlerts(fallbackAdvisories);
          setError("");
        }
      } finally {
        if (
          mountedRef.current &&
          requestId === requestIdRef.current
        ) {
          setLoading(false);
        }
      }
    };

    loadAdvisories();
  }, [requestPayload, fallbackAdvisories]);

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
