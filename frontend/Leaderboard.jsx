import React, { useState } from "react";
import { FaTrophy, FaMedal, FaStar, FaSeedling, FaCloudSun, FaUsers } from "react-icons/fa";
import { useTheme } from "./ThemeContext";
import "./Leaderboard.css";

const INITIAL_LEADERBOARD = [
  { id: 1, name: "Ramesh Kumar", location: "Maharashtra", points: 2850, badges: ["Top Farmer", "Weather Expert"], avatar: "👨‍🌾" },
  { id: 2, name: "Lakshmi Devi", location: "Tamil Nadu", points: 2640, badges: ["Crop Master", "Community Leader"], avatar: "👩‍🌾" },
  { id: 3, name: "Suresh Patel", location: "Gujarat", points: 2420, badges: ["Soil Expert"], avatar: "👨‍🌾" },
  { id: 4, name: "Priya Singh", location: "Punjab", points: 2180, badges: ["Irrigation Pro"], avatar: "👩‍🌾" },
  { id: 5, name: "Amit Sharma", location: "Uttar Pradesh", points: 1950, badges: ["Rising Star"], avatar: "👨‍🌾" },
  { id: 6, name: "Kavitha Reddy", location: "Karnataka", points: 1820, badges: ["Organic Farmer"], avatar: "👩‍🌾" },
  { id: 7, name: "Rajesh Verma", location: "Madhya Pradesh", points: 1690, badges: ["Tech Adopter"], avatar: "👨‍🌾" },
  { id: 8, name: "Meena Joshi", location: "Maharashtra", points: 1540, badges: ["Knowledge Sharer"], avatar: "👩‍🌾" },
];

export default function Leaderboard() {
  const { theme } = useTheme();
  const [timeFilter, setTimeFilter] = useState("all");

  const getRankIcon = (rank) => {
    if (rank === 1) return <FaTrophy className="rank-icon gold" />;
    if (rank === 2) return <FaMedal className="rank-icon silver" />;
    if (rank === 3) return <FaMedal className="rank-icon bronze" />;
    return <span className="rank-number">#{rank}</span>;
  };

  return (
    <div className={`leaderboard-page ${theme === "dark" ? "theme-dark" : ""}`}>
      <div className="leaderboard-container">
        {/* Header */}
        <div className="leaderboard-header">
          <div className="leaderboard-icon-wrap">
            <FaTrophy size={40} className="leaderboard-main-icon" />
          </div>
          <h1>Farmer Leaderboard 🏆</h1>
          <p className="leaderboard-subtitle">
            Celebrating our top contributors who are transforming agriculture through smart farming
          </p>
        </div>

        {/* Time Filter */}
        <div className="leaderboard-filters">
          {["all", "monthly", "weekly"].map((filter) => (
            <button
              key={filter}
              className={`filter-btn ${timeFilter === filter ? "active" : ""}`}
              onClick={() => setTimeFilter(filter)}
            >
              {filter.charAt(0).toUpperCase() + filter.slice(1)}
            </button>
          ))}
        </div>

        {/* Leaderboard List */}
        <div className="leaderboard-list">
          {INITIAL_LEADERBOARD.map((farmer, index) => (
            <div
              key={farmer.id}
              className={`leaderboard-card ${index < 3 ? `top-${index + 1}` : ""}`}
            >
              <div className="card-rank">{getRankIcon(index + 1)}</div>
              
              <div className="card-avatar">{farmer.avatar}</div>
              
              <div className="card-info">
                <h3 className="farmer-name">{farmer.name}</h3>
                <p className="farmer-location">📍 {farmer.location}</p>
                <div className="farmer-badges">
                  {farmer.badges.map((badge, i) => (
                    <span key={i} className="badge">{badge}</span>
                  ))}
                </div>
              </div>

              <div className="card-points">
                <span className="points-value">{farmer.points.toLocaleString()}</span>
                <span className="points-label">points</span>
              </div>
            </div>
          ))}
        </div>

        {/* Stats Section */}
        <div className="leaderboard-stats">
          <div className="stat-card">
            <FaUsers className="stat-icon" />
            <h3>1,250+</h3>
            <p>Active Farmers</p>
          </div>
          <div className="stat-card">
            <FaSeedling className="stat-icon" />
            <h3>850+</h3>
            <p>Crops Tracked</p>
          </div>
          <div className="stat-card">
            <FaCloudSun className="stat-icon" />
            <h3>500+</h3>
            <p>Weather Alerts</p>
          </div>
          <div className="stat-card">
            <FaStar className="stat-icon" />
            <h3>15,000+</h3>
            <p>Total Points</p>
          </div>
        </div>

        {/* Call to Action */}
        <div className="leaderboard-cta">
          <h2>Want to Climb the Leaderboard?</h2>
          <p>
            Earn points by completing farming tasks, sharing knowledge, and helping the community
          </p>
          <a href="/advisor" className="cta-button">Start Earning Points</a>
        </div>
      </div>
    </div>
  );
}
