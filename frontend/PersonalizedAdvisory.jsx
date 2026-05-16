import React, { useMemo } from "react";
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

export default function PersonalizedRecommendations({ userProfile, weatherData }) {

  const resolvedSeason = useMemo(() => {
    if (userProfile?.season) return userProfile.season;
    return deriveSeasonFromCalendar();
  }, [userProfile?.season]);

  const recommendations = useMemo(() => {
    if (!userProfile) return [];

    return generateRecommendations({
      weatherData,
      cropType: userProfile.cropType,
      season: resolvedSeason,
    });

  }, [userProfile, weatherData, resolvedSeason]);

  if (!userProfile) {
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
            onClick={() => window.location.href = '/profile-setup'}
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

  return (
    <div className="personalized-section">
      <div className="section-header">
        <h2>
          <span className="header-icon-wrap">🎯</span>
          Recommendations for You
        </h2>
        <div className="recommendation-meta">
          {userProfile.cropType && (
            <span className="crop-badge">
              <Wheat size={14} />
              {userProfile.cropType}
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
          const isCropType = rec.type === 'crop' && userProfile?.cropType;
          const CropIcon = isCropType ? getCropIcon(userProfile.cropType) : null;

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
