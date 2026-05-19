import React, { useState, useEffect, useCallback } from "react";
import { toast } from "react-toastify";
import {
  FaThermometerHalf,
  FaCloudRain,
  FaWind,
  FaExclamationTriangle,
  FaTint,
  FaEye,
} from "react-icons/fa";
import "./WeatherAlerts.css";

/**
 * WeatherAlerts Component
 * 
 * Displays real-time weather alerts for a farm location with crop-specific warnings.
 * Features:
 * - Real-time weather data (temperature, humidity, rainfall, wind speed)
 * - Severity-based alert system (low, medium, high, critical)
 * - Crop-specific recommendations
 * - Secure API (no exposed API keys)
 * - Alert history tracking
 */
const WeatherAlerts = ({ latitude, longitude, location, crop }) => {
  const [weather, setWeather] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [showHistory, setShowHistory] = useState(false);
  const [alertHistory, setAlertHistory] = useState([]);
  const [selectedAlert, setSelectedAlert] = useState(null);

  // Color mapping for severity levels
  const severityColors = {
    low: "#3498db",      // Blue
    medium: "#f39c12",   // Orange
    high: "#e74c3c",     // Red
    critical: "#c0392b", // Dark Red
  };

  const severityIcons = {
    low: "ℹ️",
    medium: "⚠️",
    high: "🔴",
    critical: "🚨",
  };

  // Fetch weather alerts
  const fetchWeatherAlerts = useCallback(async () => {
    if (!latitude || !longitude || !location) {
      setError("Location information is required");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch("/api/weather/alerts", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          latitude,
          longitude,
          location,
          crop: crop || null,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to fetch weather alerts");
      }

      const data = await response.json();

      if (data.success) {
        setWeather(data.weather);
        setAlerts(data.alerts.alerts || []);
        setLastUpdated(new Date());

        // Show critical/high alerts as toasts
        const criticalAlerts = data.alerts.alerts.filter(
          (a) => a.severity === "critical" || a.severity === "high"
        );
        criticalAlerts.forEach((alert) => {
          toast.warning(alert.title, {
            position: "top-right",
            autoClose: 5000,
            hideProgressBar: false,
          });
        });
      }
    } catch (err) {
      const errorMsg = err.message || "Failed to fetch weather data";
      setError(errorMsg);
      toast.error(errorMsg, {
        position: "top-right",
        autoClose: 4000,
      });
    } finally {
      setLoading(false);
    }
  }, [latitude, longitude, location, crop]);

  // Fetch alert history
  const fetchAlertHistory = useCallback(async () => {
    try {
      const response = await fetch("/api/weather/alerts/history");
      if (response.ok) {
        const data = await response.json();
        setAlertHistory(data.recent_alerts || []);
      }
    } catch (err) {
      console.error("Failed to fetch alert history:", err);
    }
  }, []);

  // Initial fetch and auto-refresh
  useEffect(() => {
    fetchWeatherAlerts();
    
    // Auto-refresh every 30 minutes
    const interval = setInterval(fetchWeatherAlerts, 30 * 60 * 1000);
    
    return () => clearInterval(interval);
  }, [fetchWeatherAlerts]);

  // Weather info card component
  const WeatherInfoCard = ({ icon, label, value, unit }) => (
    <div className="weather-info-item">
      <div className="info-icon">{icon}</div>
      <div className="info-content">
        <p className="info-label">{label}</p>
        <p className="info-value">
          {value}
          {unit && <span className="info-unit">{unit}</span>}
        </p>
      </div>
    </div>
  );

  // Alert card component
  const AlertCard = ({ alert }) => (
    <div
      className="alert-card"
      style={{ borderLeftColor: severityColors[alert.severity] }}
      onClick={() => setSelectedAlert(alert)}
    >
      <div className="alert-header">
        <span className="alert-severity-icon">
          {severityIcons[alert.severity]}
        </span>
        <h4 className="alert-title">{alert.title}</h4>
        <span className="alert-severity" style={{ color: severityColors[alert.severity] }}>
          {alert.severity.toUpperCase()}
        </span>
      </div>
      <p className="alert-message">{alert.message}</p>
      {alert.crop && <p className="alert-crop">🌾 {alert.crop}</p>}
      {alert.recommended_action && (
        <div className="alert-action">
          <strong>💡 Action:</strong> {alert.recommended_action}
        </div>
      )}
    </div>
  );

  // Alert detail modal
  const AlertDetailModal = ({ alert, onClose }) => (
    <div className="alert-modal-overlay" onClick={onClose}>
      <div className="alert-modal-content" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close-btn" onClick={onClose}>×</button>
        <h3>{alert.title}</h3>
        <div className="modal-details">
          <p>
            <strong>Severity:</strong>{" "}
            <span style={{ color: severityColors[alert.severity] }}>
              {alert.severity.toUpperCase()}
            </span>
          </p>
          <p>
            <strong>Message:</strong> {alert.message}
          </p>
          {alert.crop && (
            <p>
              <strong>Crop:</strong> {alert.crop}
            </p>
          )}
          <p>
            <strong>Time:</strong>{" "}
            {new Date(alert.timestamp).toLocaleString()}
          </p>
          {alert.expires_at && (
            <p>
              <strong>Expires:</strong>{" "}
              {new Date(alert.expires_at).toLocaleString()}
            </p>
          )}
          {alert.recommended_action && (
            <div className="modal-action">
              <strong>💡 Recommended Action:</strong>
              <p>{alert.recommended_action}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );

  if (!latitude || !longitude) {
    return (
      <div className="weather-alerts-container">
        <div className="error-message">
          📍 Please set your farm location to view weather alerts
        </div>
      </div>
    );
  }

  return (
    <div className="weather-alerts-container">
      <div className="weather-alerts-header">
        <h2>🌤️ Real-Time Weather Alerts</h2>
        <div className="header-actions">
          <button
            className="refresh-btn"
            onClick={fetchWeatherAlerts}
            disabled={loading}
          >
            {loading ? "Fetching..." : "🔄 Refresh"}
          </button>
          <button
            className="history-btn"
            onClick={() => {
              setShowHistory(!showHistory);
              if (!showHistory) fetchAlertHistory();
            }}
          >
            📋 History
          </button>
        </div>
      </div>

      {error && (
        <div className="error-message">
          ⚠️ {error}
        </div>
      )}

      {weather && (
        <div className="weather-display">
          <div className="location-info">
            <h3>📍 {location}</h3>
            <p className="last-updated">
              Last updated: {lastUpdated?.toLocaleTimeString()}
            </p>
          </div>

          <div className="weather-grid">
            <WeatherInfoCard
              icon={<FaThermometerHalf />}
              label="Temperature"
              value={weather.temperature}
              unit="°C"
            />
            <WeatherInfoCard
              icon={<FaTint />}
              label="Humidity"
              value={weather.humidity}
              unit="%"
            />
            <WeatherInfoCard
              icon={<FaCloudRain />}
              label="Rainfall"
              value={weather.rainfall}
              unit="mm"
            />
            <WeatherInfoCard
              icon={<FaWind />}
              label="Wind Speed"
              value={weather.wind_speed}
              unit="km/h"
            />
            <WeatherInfoCard
              icon={<FaEye />}
              label="Cloud Cover"
              value={weather.cloud_cover}
              unit="%"
            />
          </div>
        </div>
      )}

      <div className="alerts-section">
        <div className="alerts-header">
          <h3>
            <FaExclamationTriangle /> Active Alerts ({alerts.length})
          </h3>
          {alerts.length > 0 && (
            <div className="alert-summary">
              <span className="critical-count">
                🚨 Critical: {alerts.filter((a) => a.severity === "critical").length}
              </span>
              <span className="high-count">
                🔴 High: {alerts.filter((a) => a.severity === "high").length}
              </span>
            </div>
          )}
        </div>

        {alerts.length === 0 ? (
          <div className="no-alerts">✅ No weather alerts at this time</div>
        ) : (
          <div className="alerts-list">
            {alerts.map((alert) => (
              <AlertCard key={alert.id} alert={alert} />
            ))}
          </div>
        )}
      </div>

      {showHistory && (
        <div className="history-section">
          <h3>📋 Alert History</h3>
          {alertHistory.length === 0 ? (
            <p>No alert history available</p>
          ) : (
            <div className="history-list">
              {alertHistory.slice(0, 10).map((alert) => (
                <div key={alert.id} className="history-item">
                  <span className="history-severity" style={{ color: severityColors[alert.severity] }}>
                    {alert.severity}
                  </span>
                  <span className="history-title">{alert.title}</span>
                  <span className="history-time">
                    {new Date(alert.timestamp).toLocaleDateString()}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {selectedAlert && (
        <AlertDetailModal alert={selectedAlert} onClose={() => setSelectedAlert(null)} />
      )}
    </div>
  );
};

export default WeatherAlerts;
