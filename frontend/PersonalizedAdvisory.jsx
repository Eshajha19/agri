import React, { useMemo } from "react";
import {
  CloudRain,
  Sun,
  AlertTriangle,
  Sprout,
  ShieldCheck,
  Thermometer,
  Wind,
  Droplets,
} from "lucide-react";

import "./PersonalizedAdvisory.css";
import { generateRecommendations } from "./utils/recommendationEngine";
import { 
  AlertTriangle, 
  ThermometerSun, 
  Snowflake, 
  Wheat, 
  Sprout, 
  Leaf, 
  CloudRain, 
  CloudSnow, 
  Sun,
  Info,
  Droplets,
  Calendar
} from "lucide-react";

/**
 * Derive the current Indian agricultural season from the calendar month
 * when the user's profile does not have an explicit season set.
 *
 * Indian season calendar:
 *   Kharif  — June  to October  (months 6–10)
 *   Rabi    — November to February (months 11–2)
 *   Zaid    — March to May       (months 3–5)
 *
 * @returns {"Kharif" | "Rabi" | "Zaid"}
 */
function deriveSeasonFromCalendar() {
  const month = new Date().getMonth() + 1; // 1-indexed
  if (month >= 6 && month <= 10) return "Kharif";
  if (month >= 3 && month <= 5) return "Zaid";
  return "Rabi"; // November – February
}

const TYPE_CONFIG = {
  warning: {
    icon: AlertTriangle,
    label: "Weather Alert",
    gradient: "linear-gradient(135deg, #fef2f2 0%, #fecaca 100%)",
    borderColor: "#ef4444",
    iconBg: "#fef2f2",
    iconColor: "#ef4444",
  },
  heat: {
    icon: ThermometerSun,
    label: "Heat Advisory",
    gradient: "linear-gradient(135deg, #fffbeb 0%, #fde68a 100%)",
    borderColor: "#f59e0b",
    iconBg: "#fffbeb",
    iconColor: "#f59e0b",
  },
  frost: {
    icon: Snowflake,
    label: "Frost Warning",
    gradient: "linear-gradient(135deg, #f0f9ff 0%, #bae6fd 100%)",
    borderColor: "#0ea5e9",
    iconBg: "#f0f9ff",
    iconColor: "#0ea5e9",
  },
  crop: {
    icon: Wheat,
    label: "Crop Care",
    gradient: "linear-gradient(135deg, #f0fdf4 0%, #bbf7d0 100%)",
    borderColor: "#22c55e",
    iconBg: "#f0fdf4",
    iconColor: "#22c55e",
  },
  season: {
    icon: Calendar,
    label: "Seasonal Tip",
    gradient: "linear-gradient(135deg, #faf5ff 0%, #e9d5ff 100%)",
    borderColor: "#a855f7",
    iconBg: "#faf5ff",
    iconColor: "#a855f7",
  },
};

function getCropIcon(cropType) {
  const crop = (cropType || "").toLowerCase();
  if (crop.includes("paddy") || crop.includes("rice")) return Wheat;
  if (crop.includes("cotton")) return Sprout;
  return Leaf;
}

export default function PersonalizedRecommendations({
  userProfile,
  weatherData,
}) {
export default function PersonalizedRecommendations({ userProfile, weatherData }) {
export default function PersonalizedRecommendations({ userData, weatherData }) {

  const resolvedSeason = useMemo(() => {
    if (userData?.season) return userData.season;
    return deriveSeasonFromCalendar();
  }, [userData?.season]);

  const recommendations = useMemo(() => {
    if (!userData) return [];

    return generateRecommendations({
      weatherData,
      cropType: userData.cropType,
      season: resolvedSeason,
    });
  }, [userProfile, weatherData]);

  }, [userData, weatherData, resolvedSeason]);

  if (!userData) {
    return (
      <div className="personalized-section">
        <div className="section-header">
          <h2><Info className="section-icon" /> Personalized Recommendations</h2>
        </div>
        <div className="empty-state">
          <div className="empty-icon">👤</div>
          <p>Complete your profile to get personalized farming advice</p>
          <button 
            className="complete-profile-btn"
            onClick={() => window.location.href = '/profile-settings'}
          >
            Complete Profile
          </button>
        </div>
      </div>
    );
  }

  if (recommendations.length === 0) {
    return (
      <div className="personalized-section">
        <div className="section-header">
          <h2><Info className="section-icon" /> Personalized Recommendations</h2>
        </div>
        <div className="empty-state">
          <div className="empty-icon">✓</div>
          <p>All clear! No urgent recommendations at this time.</p>
          <p className="empty-subtext">Check back later for updated advice based on weather and crop conditions.</p>
        </div>
      </div>
    );
  }

  const sortedRecs = [...recommendations].sort((a, b) => {
    const priority = { warning: 1, heat: 2, frost: 2, crop: 3, season: 4 };
    return (priority[a.type] || 99) - (priority[b.type] || 99);
  });

  /* =========================================================
      ICON HANDLER
  ========================================================= */

  const getIcon = (type) => {
    switch (type) {
      case "weather":
        return <CloudRain size={28} />;

      case "irrigation":
        return <Droplets size={28} />;

      case "temperature":
        return <Thermometer size={28} />;

      case "wind":
        return <Wind size={28} />;

      case "warning":
        return <AlertTriangle size={28} />;

      case "crop":
        return <Sprout size={28} />;

      case "security":
        return <ShieldCheck size={28} />;

      default:
        return <Sun size={28} />;
    }
  };

  return (
    <section className="personalized-section">
      {/* HEADER */}

      <div className="advisory-header">
        <div>
          <span className="section-badge">
            Smart AI Advisory
          </span>

          <h2>
            🎯 Personalized Recommendations
          </h2>

          <p>
            Real-time recommendations based on
            your crop profile and weather
            conditions.
          </p>
        </div>
      </div>

      {/* EMPTY PROFILE */}

      {!userProfile ? (
        <div className="empty-state">
          <div className="empty-icon">
            🌱
          </div>

          <h3>Profile Incomplete</h3>

          <p>
            Complete your farming profile to
            receive personalized crop insights
            and smart recommendations.
          </p>

          <button className="complete-btn">
            Complete Profile
          </button>
        </div>
      ) : recommendations.length === 0 ? (
        /* EMPTY RECOMMENDATIONS */

        <div className="empty-state">
          <div className="empty-icon">
            📭
          </div>

          <h3>No Recommendations</h3>

          <p>
            No recommendations are available at
            the moment.
          </p>
        </div>
      ) : (
        <>
          {/* WEATHER SUMMARY */}

          <div className="weather-summary">
            <div className="weather-card">
              <span>🌡 Temperature</span>
              <h3>
                {weatherData?.temperature || "--"}
                °C
    <div className="personalized-section">
      <div className="section-header">
        <h2>
          <span className="header-icon-wrap">🎯</span>
          Recommendations for You
        </h2>
        <div className="recommendation-meta">
          {userData.cropType && (
            <span className="crop-badge">
              <Wheat size={14} />
              {userData.cropType}
            </span>
          )}
          <span className="season-badge">
            <Calendar size={14} />
            {resolvedSeason}
          </span>
        </div>
      </div>

      <div className="recommendation-grid">
        {sortedRecs.map((rec, index) => {
          const config = TYPE_CONFIG[rec.type];
          const IconComponent = config.icon;
          const isCropType = rec.type === 'crop' && userData?.cropType;
          const CropIcon = isCropType ? getCropIcon(userData.cropType) : null;

          return (
            <div 
              key={index} 
              className={`recommendation-card ${rec.type}`}
              style={{
                background: config.gradient,
                borderLeft: `4px solid ${config.borderColor}`,
                animationDelay: `${index * 0.1}s`
              }}
              role="alert"
              aria-live={rec.type === 'warning' || rec.type === 'heat' || rec.type === 'frost' ? 'polite' : 'off'}
            >
              <div className="card-header">
                <div 
                  className="icon-wrapper"
                  style={{ background: config.iconBg, color: config.iconColor }}
                >
                  {isCropType && CropIcon ? (
                    <CropIcon size={24} />
                  ) : (
                    <IconComponent size={24} />
                  )}
                </div>
                <div className="card-badge" style={{ background: config.borderColor }}>
                  {config.label}
                </div>
              </div>

              <h3 className="card-title">
                {rec.type === 'warning' && '⚠️ '}
                {rec.type === 'heat' && '☀️ '}
                {rec.type === 'frost' && '❄️ '}
                {rec.type === 'crop' && userProfile?.cropType ? `${userProfile.cropType}: ` : ''}
                {rec.title || rec.type.charAt(0).toUpperCase() + rec.type.slice(1)}
              </h3>
            </div>

            <div className="weather-card">
              <span>💧 Humidity</span>
              <h3>
                {weatherData?.humidity || "--"}%
              </h3>
            </div>

            <div className="weather-card">
              <span>🌤 Condition</span>
              <h3>
                {weatherData?.condition ||
                  "Unknown"}
              </h3>
            </div>
          </div>
<h3 className="card-title">
                    {rec.type === 'warning' && '⚠️ '}
                    {rec.type === 'heat' && '☀️ '}
                    {rec.type === 'frost' && '❄️ '}
                    {rec.type === 'crop' && userData?.cropType ? `${userData.cropType}: ` : ''}
                    {rec.title || rec.type.charAt(0).toUpperCase() + rec.type.slice(1)}
                  </h3>

          {/* RECOMMENDATIONS */}

          <div className="recommendation-grid">
            {recommendations.map((rec, index) => (
              <div
                key={index}
                className={`recommendation-card ${
                  rec.type || "default"
                }`}
              >
                {/* ICON */}

                <div className="rec-icon">
                  {rec.icon || getIcon(rec.type)}
                </div>

                {/* TYPE */}

                <div className="rec-type">
                  {rec.type || "General"}
                </div>

                {/* CONTENT */}

                <p>{rec.text}</p>

                {/* FOOTER */}

                <div className="rec-footer">
                  <span>AI Advisory</span>

                  <button>
                    View Details
                  </button>
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </section>
              <p className="card-text">{rec.text}</p>

              <div className="card-footer">
                <span className="priority-indicator">
                  Priority: {rec.type === 'warning' ? 'High' : rec.type === 'heat' || rec.type === 'frost' ? 'Medium' : 'Info'}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
