import React, { useState, useEffect } from "react";
import {
  Cloud,
  CloudRain,
  AlertTriangle,
  Check,
  X,
  Loader,
  Bell,
  Calendar,
  Droplet,
  Shield,
  MapPin,
} from "lucide-react";
import "./SprayReminder.css";

export default function SprayReminder({ onClose, location = "Default" }) {
  const [cropName, setCropName] = useState("");
  const [cropStage, setCropStage] = useState("seedling");
  const [sprayType, setSprayType] = useState("pesticide");
  const [notificationEnabled, setNotificationEnabled] = useState(true);
  const [selectedDate, setSelectedDate] = useState("");
  const [weatherData, setWeatherData] = useState(null);
  const [recommendations, setRecommendations] = useState(null);
  const [scheduleHistory, setScheduleHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [activeTab, setActiveTab] = useState("schedule");

  const persistSchedule = (item) => {
    try {
      const raw = localStorage.getItem("agri_spray_schedules");
      const existing = raw ? JSON.parse(raw) : [];
      localStorage.setItem("agri_spray_schedules", JSON.stringify([item, ...(Array.isArray(existing) ? existing : [])]));
    } catch (err) {
      console.error("Failed to persist spray schedule:", err);
    }
  };

  const saveSchedule = persistSchedule;

  // Crop stages
  const cropStages = [
    { value: "seedling", label: "Seedling (0-2 weeks)" },
    { value: "vegetative", label: "Vegetative (2-4 weeks)" },
    { value: "flowering", label: "Flowering (4-8 weeks)" },
    { value: "fruiting", label: "Fruiting/Grain Filling (8-12 weeks)" },
    { value: "mature", label: "Mature/Harvest Ready" },
  ];

  const sprayTypes = [
    { value: "pesticide", label: "Pesticide" },
    { value: "fungicide", label: "Fungicide" },
    { value: "herbicide", label: "Herbicide" },
    { value: "insecticide", label: "Insecticide" },
    { value: "fertilizer", label: "Liquid Fertilizer" },
  ];

  // Fetch weather data
  useEffect(() => {
    const fetchWeather = async () => {
      try {
        const apiKey = import.meta.env.VITE_OPENWEATHER_API_KEY;
        if (!apiKey) {
          setError("Weather API key not configured");
          return;
        }

        const response = await fetch(
          `https://api.openweathermap.org/data/2.5/weather?q=${location}&appid=${apiKey}&units=metric`
        );
        const data = await response.json();

        if (response.ok) {
          setWeatherData({
            temperature: Math.round(data.main.temp),
            humidity: data.main.humidity,
            windSpeed: Math.round(data.wind.speed * 3.6), // m/s to km/h
            description: data.weather[0].main,
            rainChance: data.clouds.all,
            feelsLike: Math.round(data.main.feels_like),
          });
        }
      } catch (err) {
        console.error("Weather fetch error:", err);
      }
    };

    fetchWeather();
  }, [location]);

  // Generate recommendations
  const generateRecommendations = async () => {
    if (!cropName.trim()) {
      setError("Please enter a crop name");
      return;
    }

    setLoading(true);
    setError("");
    setSuccess("");

    try {
      const apiKey = import.meta.env.VITE_GEMINI_API_KEY;
      if (!apiKey) {
        setError("Gemini API key not configured");
        setLoading(false);
        return;
      }

      const prompt = `You are an expert agricultural spray scheduler. A farmer is growing ${cropName} at the ${cropStage} stage and wants to apply ${sprayType}.

Current weather: ${weatherData ? `Temperature: ${weatherData.temperature}°C, Humidity: ${weatherData.humidity}%, Wind: ${weatherData.windSpeed} km/h, Rain chance: ${weatherData.rainChance}%` : "Unknown"}

Provide recommendations ONLY in this JSON format:
{
  "isSuitableToday": true/false,
  "suitabilityReason": "Brief reason",
  "weatherWarnings": ["Warning 1", "Warning 2"],
  "bestSprayTime": "Morning/Afternoon/Evening or specific time",
  "rotationTip": "Safe pesticide rotation recommendation",
  "safetyPrecautions": ["Precaution 1", "Precaution 2"],
  "nextBestDay": "When to spray if not today"
}`;

      const response = await fetch(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            contents: [{ parts: [{ text: prompt }] }],
            generationConfig: {
              temperature: 0.7,
              maxOutputTokens: 500,
            },
          }),
        }
      );

      const result = await response.json();

      if (
        result.candidates &&
        result.candidates[0]?.content?.parts[0]?.text
      ) {
        const text = result.candidates[0].content.parts[0].text;
        const jsonMatch = text.match(/\{[\s\S]*\}/);

        if (jsonMatch) {
          const recommendations = JSON.parse(jsonMatch[0]);
          setRecommendations(recommendations);
          setSuccess("Recommendations generated successfully!");
        } else {
          setError("Failed to parse recommendations");
        }
      } else {
        setError("Failed to generate recommendations");
      }
    } catch (err) {
      console.error("Recommendation error:", err);
      setError("Error generating recommendations: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  // Schedule spray reminder
  const handleScheduleSpray = async () => {
    if (!selectedDate) {
      setError("Please select a date for the spray");
      return;
    }

    try {
      // Save to local schedule history
      const newSchedule = {
        id: Date.now(),
        crop: cropName,
        cropName,
        stage: cropStage,
        sprayType,
        type: sprayType,
        date: selectedDate,
        scheduledAt: selectedDate,
        notification: notificationEnabled,
        createdAt: new Date().toLocaleString(),
      };

      setScheduleHistory([newSchedule, ...scheduleHistory]);
      persistSchedule(newSchedule);

      setSuccess("Spray reminder scheduled successfully!");

      if (notificationEnabled) {
        if ("Notification" in window && Notification.permission === "granted") {
          const sprayDate = new Date(selectedDate);
          const timeUntil = sprayDate - new Date();

          if (timeUntil > 0) {
            setTimeout(() => {
              new Notification(`Spray Reminder: ${cropName}`, {
                body: `Time to apply ${sprayType} to your ${cropName} field!`,
                icon: "/agri-icon.png",
              });
            }, Math.min(timeUntil, 2147483647));
          }
        }
      }

      setCropName("");
      setSelectedDate("");
      setSprayType("pesticide");
      setCropStage("seedling");
    } catch (err) {
      setError("Failed to schedule spray: " + err.message);
    }
  };

  const requestNotificationPermission = async () => {
    if ("Notification" in window) {
      if (Notification.permission === "granted") {
        setSuccess("Notifications already enabled!");
      } else if (Notification.permission !== "denied") {
        const permission = await Notification.requestPermission();
        if (permission === "granted") {
          setSuccess("Notifications enabled successfully!");
        } else {
          setError("Notification permission denied");
        }
      }
    } else {
      setError("Notifications not supported in this browser");
    }
  };

  // Handle escape key
  useEffect(() => {
    const handleEsc = (e) => {
      if (e.key === "Escape") {
        onClose?.();
      }
    };
    window.addEventListener("keydown", handleEsc);
    return () => window.removeEventListener("keydown", handleEsc);
  }, [onClose]);

  const isSuitableWeather =
    weatherData &&
    weatherData.rainChance < 30 &&
    weatherData.temperature >= 15 &&
    weatherData.temperature <= 35 &&
    weatherData.windSpeed < 20;

  return (
    <div className="spray-reminder-modal">
      <div className="modal-header">
        <div className="header-title">
          <Droplet className="header-icon" size={28} />
          <div>
            <h2 className="modal-title">Spray Scheduler</h2>
            <p className="modal-subtitle">
              Weather-aware spray scheduling for optimal results
            </p>
          </div>
        </div>
        <button
          className="close-btn"
          onClick={onClose}
          aria-label="Close spray reminder"
        >
          <X size={24} />
        </button>
      </div>

      {/* Tabs */}
      <div className="tabs">
        <button
          className={`tab ${activeTab === "schedule" ? "active" : ""}`}
          onClick={() => setActiveTab("schedule")}
        >
          Schedule
        </button>
        <button
          className={`tab ${activeTab === "history" ? "active" : ""}`}
          onClick={() => setActiveTab("history")}
        >
          History ({scheduleHistory.length})
        </button>
      </div>

      {activeTab === "schedule" && (
        <div className="modal-content">
          {/* Weather Alert */}
          {weatherData && (
            <div className={`weather-alert ${isSuitableWeather ? "good" : "caution"}`}>
              <div className="weather-status">
                {isSuitableWeather ? (
                  <Check size={20} />
                ) : (
                  <AlertTriangle size={20} />
                )}
                <span>
                  {isSuitableWeather
                    ? "Good conditions for spraying"
                    : "Weather conditions may not be ideal"}
                </span>
              </div>
              <div className="weather-details">
                <span>🌡️ {weatherData.temperature}°C</span>
                <span>💨 {weatherData.windSpeed} km/h</span>
                <span>💧 {weatherData.humidity}%</span>
                <span>🌧️ {weatherData.rainChance}% rain</span>
              </div>
            </div>
          )}

          {/* Input Fields */}
          <div className="input-group">
            <label htmlFor="crop">Crop Name</label>
            <input
              id="crop"
              type="text"
              placeholder="e.g., Rice, Wheat, Cotton"
              value={cropName}
              onChange={(e) => setCropName(e.target.value)}
              className="sr-input"
            />
          </div>

          <div className="input-group">
            <label htmlFor="stage">Crop Stage</label>
            <select
              id="stage"
              value={cropStage}
              onChange={(e) => setCropStage(e.target.value)}
              className="sr-select"
            >
              {cropStages.map((stage) => (
                <option key={stage.value} value={stage.value}>
                  {stage.label}
                </option>
              ))}
            </select>
          </div>

          <div className="input-group">
            <label htmlFor="spray-type">Spray Type</label>
            <select
              id="spray-type"
              value={sprayType}
              onChange={(e) => setSprayType(e.target.value)}
              className="sr-select"
            >
              {sprayTypes.map((type) => (
                <option key={type.value} value={type.value}>
                  {type.label}
                </option>
              ))}
            </select>
          </div>

          {/* Generate Recommendations Button */}
          <button
            onClick={generateRecommendations}
            disabled={loading || !cropName.trim()}
            className="sr-btn primary"
          >
            {loading ? (
              <>
                <Loader size={18} className="spinner" />
                Analyzing...
              </>
            ) : (
              <>
                <Shield size={18} />
                Get Recommendations
              </>
            )}
          </button>

          {/* Recommendations Display */}
          {recommendations && (
            <div className="recommendations-box">
              <h3 className="rec-title">📋 Spray Recommendations</h3>

              <div
                className={`recommendation-item ${recommendations.isSuitableToday ? "suitable" : "not-suitable"}`}
              >
                <div className="rec-header">
                  {recommendations.isSuitableToday ? (
                    <Check size={20} className="check-icon" />
                  ) : (
                    <AlertTriangle size={20} className="warning-icon" />
                  )}
                  <span className="rec-label">Spray Suitability Today</span>
                </div>
                <p className="rec-text">{recommendations.suitabilityReason}</p>
              </div>

              {recommendations.weatherWarnings &&
                recommendations.weatherWarnings.length > 0 && (
                  <div className="recommendation-item warning">
                    <div className="rec-header">
                      <AlertTriangle size={20} />
                      <span className="rec-label">⚠️ Weather Warnings</span>
                    </div>
                    <ul className="rec-list">
                      {recommendations.weatherWarnings.map((warning, idx) => (
                        <li key={idx}>{warning}</li>
                      ))}
                    </ul>
                  </div>
                )}

              {recommendations.bestSprayTime && (
                <div className="recommendation-item info">
                  <div className="rec-header">
                    <Calendar size={20} />
                    <span className="rec-label">🕐 Best Spray Time</span>
                  </div>
                  <p className="rec-text">{recommendations.bestSprayTime}</p>
                </div>
              )}

              {recommendations.rotationTip && (
                <div className="recommendation-item info">
                  <div className="rec-header">
                    <Shield size={20} />
                    <span className="rec-label">🔄 Pesticide Rotation</span>
                  </div>
                  <p className="rec-text">{recommendations.rotationTip}</p>
                </div>
              )}

              {recommendations.safetyPrecautions &&
                recommendations.safetyPrecautions.length > 0 && (
                  <div className="recommendation-item safety">
                    <div className="rec-header">
                      <Shield size={20} />
                      <span className="rec-label">🛡️ Safety Precautions</span>
                    </div>
                    <ul className="rec-list">
                      {recommendations.safetyPrecautions.map((precaution, idx) => (
                        <li key={idx}>{precaution}</li>
                      ))}
                    </ul>
                  </div>
                )}

              {recommendations.nextBestDay && (
                <div className="recommendation-item info">
                  <div className="rec-header">
                    <Calendar size={20} />
                    <span className="rec-label">📅 Next Best Day</span>
                  </div>
                  <p className="rec-text">{recommendations.nextBestDay}</p>
                </div>
              )}
            </div>
          )}

          {/* Date Selection */}
          <div className="input-group">
            <label htmlFor="spray-date">Schedule Spray Date</label>
            <input
              id="spray-date"
              type="datetime-local"
              value={selectedDate}
              onChange={(e) => setSelectedDate(e.target.value)}
              className="sr-input"
              min={new Date().toISOString().slice(0, 16)}
            />
          </div>

          {/* Notification Toggle */}
          <div className="notification-group">
            <input
              id="notify"
              type="checkbox"
              checked={notificationEnabled}
              onChange={(e) => setNotificationEnabled(e.target.checked)}
              className="sr-checkbox"
            />
            <label htmlFor="notify">
              <Bell size={16} />
              Enable notifications for scheduled spray
            </label>
            <button
              onClick={requestNotificationPermission}
              className="sr-btn secondary"
              title="Enable browser notifications"
            >
              Enable Notifications
            </button>
          </div>

          {/* Schedule Button */}
          <button
            onClick={handleScheduleSpray}
            disabled={!cropName.trim() || !selectedDate}
            className="sr-btn primary full-width"
          >
            <Calendar size={18} />
            Schedule Spray Reminder
          </button>

          {/* Messages */}
          {error && (
            <div className="message error">
              <X size={16} />
              {error}
            </div>
          )}
          {success && (
            <div className="message success">
              <Check size={16} />
              {success}
            </div>
          )}
        </div>
      )}

      {activeTab === "history" && (
        <div className="modal-content">
          {scheduleHistory.length === 0 ? (
            <div className="empty-state">
              <Calendar size={48} />
              <p>No spray schedules yet</p>
              <p className="text-secondary">Schedule your first spray to see it here</p>
            </div>
          ) : (
            <div className="history-list">
              {scheduleHistory.map((schedule) => (
                <div key={schedule.id} className="history-item">
                  <div className="history-header">
                    <div className="history-crop">
                      <span className="crop-name">{schedule.crop}</span>
                      <span className="crop-stage">
                        {cropStages.find((s) => s.value === schedule.stage)
                          ?.label || schedule.stage}
                      </span>
                    </div>
                    <div className="history-badges">
                      <span className="badge type">{schedule.type}</span>
                      {schedule.notification && (
                        <span className="badge notify">
                          <Bell size={12} /> Notify
                        </span>
                      )}
                    </div>
                  </div>
                  <div className="history-footer">
                    <span className="date">📅 {schedule.date}</span>
                    <span className="created">Added: {schedule.createdAt}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
