import React, { useMemo } from "react";
import "./PersonalizedAdvisory.css";
import { generateRecommendations } from "./utils/recommendationEngine";

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

export default function PersonalizedRecommendations({ userProfile, weatherData }) {

  /**
   * Resolve the season to pass to the recommendation engine.
   *
   * Priority:
   *   1. userProfile.season  — explicit user preference (most accurate)
   *   2. Calendar fallback   — derived from current month so seasonal
   *                            recommendation branches never stay silent
   *                            when the profile field is missing.
   */
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

              <h3 style={{ textTransform: "capitalize" }}>
                {rec.type}
              </h3>

              <p>{rec.text}</p>

            </div>
          ))}
        </div>
      )}
    </div>
  );
}
