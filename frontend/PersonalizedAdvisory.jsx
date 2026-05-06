import React, { useMemo } from "react";
import "./PersonalizedAdvisory.css";
import { generateRecommendations } from "./utils/recommendationEngine";

export default function PersonalizedRecommendations({ userProfile, weatherData }) {

  const recommendations = useMemo(() => {
    if (!userProfile) return [];

    return generateRecommendations({
      weatherData,
      cropType: userProfile.cropType,
    });

  }, [userProfile, weatherData]);

  return (
    <div className="personalized-section">
      <h2>🎯 Personalized Recommendations</h2>

      {!userProfile ? (
        <p>Complete your profile to get recommendations.</p>
      ) : recommendations.length === 0 ? (
        <p>No recommendations available.</p>
      ) : (
        <div className="recommendation-grid">
          {recommendations.map((rec, index) => (
            <div key={index} className={`recommendation-card ${rec.type}`}>
              
              <div className="rec-icon">{rec.icon}</div>

              {/* Optional: show type as title */}
              <h3 style={{ textTransform: "capitalize" }}>
                {rec.type}
              </h3>

              {/* ✅ Correct field from engine */}
              <p>{rec.text}</p>

            </div>
          ))}
        </div>
      )}
    </div>
  );
}