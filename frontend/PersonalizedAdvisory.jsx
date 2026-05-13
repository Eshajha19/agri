import React, { useMemo } from "react";
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

  return (
    <section className="personalized-section">

      <div className="section-header">
        <h2>🎯 Personalized Recommendations</h2>

        {weatherData && (
          <div className="weather-summary">
            <span>🌡 {weatherData.temp}°C</span>
            <span>💧 {weatherData.humidity}%</span>
            <span>🌧 {weatherData.condition}</span>
          </div>
        )}
      </div>

      {!userProfile ? (

        <div className="empty-state">
          <p>Complete your profile to unlock smart farming suggestions.</p>
        </div>

      ) : recommendations.length === 0 ? (

        <div className="empty-state">
          <p>No recommendations available right now.</p>
        </div>

      ) : (

        <div className="recommendation-grid">

          {recommendations.map((rec, index) => (
            <div
              key={index}
              className={`recommendation-card ${rec.type}`}
            >

              <div className="card-top">

                <div className="rec-icon">
                  {rec.icon}
                </div>

                <span className={`priority ${rec.priority || "medium"}`}>
                  {rec.priority || "medium"}
                </span>

              </div>

              <h3>
                {rec.title || rec.type}
              </h3>

              <p>{rec.text}</p>

              <button className="learn-btn">
                Learn More →
              </button>

            </div>
          ))}

        </div>
      )}
    </section>
  );
}