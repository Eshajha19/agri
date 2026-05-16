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

export default function PersonalizedRecommendations({
  userProfile,
  weatherData,
}) {
  const recommendations = useMemo(() => {
    if (!userProfile) return [];

    return generateRecommendations({
      weatherData,
      cropType: userProfile.cropType,
    });
  }, [userProfile, weatherData]);

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
  );
}