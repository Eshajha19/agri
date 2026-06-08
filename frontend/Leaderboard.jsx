import React, { useState, useEffect } from "react";
import { FaTrophy, FaMedal, FaStar, FaSeedling, FaCloudSun, FaUsers } from "react-icons/fa";
import { useTheme } from "./ThemeContext";
import { db, isFirebaseConfigured } from "./lib/firebase";
import { collection, query, orderBy, limit, getDocs } from "firebase/firestore";
import "./Leaderboard.css";

// Number of farmers shown on the leaderboard
const PAGE_SIZE = 10;

export default function Leaderboard() {
  const { theme } = useTheme();
  const [timeFilter, setTimeFilter] = useState("all");
  const [farmers, setFarmers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!isFirebaseConfigured()) {
      setLoading(false);
      return;
    }

    let cancelled = false;
    setLoading(true);
    setError(null);

    const fetchLeaderboard = async () => {
      try {
        // Determine which points field to sort by based on the active filter.
        const sortField =
          timeFilter === "monthly"
            ? "monthlyReputation"
            : timeFilter === "weekly"
            ? "weeklyReputation"
            : "reputation";

        const q = query(
          collection(db, "users"),
          orderBy(sortField, "desc"),
          limit(PAGE_SIZE)
        );

        const snapshot = await getDocs(q);

        if (cancelled) return;

        const results = snapshot.docs.map((docSnap) => {
          const data = docSnap.data();

          return {
            id: docSnap.id,
            name: data.displayName || "Farmer",
            location: data.address || data.location || "India",
            points: Number(data[sortField] ?? data.reputation ?? 0),
            badges: Array.isArray(data.badges) ? data.badges : [],
            avatar: "👨‍🌾",
            updatedAt:
              data.updatedAt?.seconds ??
              data.lastUpdated?.seconds ??
              0,
          };
        });

        /*
         * Deterministic ranking:
         * 1. Higher score first
         * 2. Most recently updated first
         * 3. Stable id comparison as final tie-breaker
         */
        const sortedResults = [...results].sort((a, b) => {
          if (b.points !== a.points) {
            return b.points - a.points;
          }

          if (b.updatedAt !== a.updatedAt) {
            return b.updatedAt - a.updatedAt;
          }

          return a.id.localeCompare(b.id);
        });

        const rankedResults = sortedResults.map((farmer, index) => ({
          ...farmer,
          rank: index + 1,
        }));

        console.info(
          `[LEADERBOARD] filter=${timeFilter} entries=${rankedResults.length}`
        );

        setFarmers(rankedResults);
      } catch (err) {
        if (!cancelled) {
          console.error("Leaderboard fetch error:", err);
          setError("Could not load leaderboard. Please try again.");
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };

    fetchLeaderboard();

    return () => {
      cancelled = true;
    };
  }, [timeFilter]);

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
          {loading && (
            <div style={{ textAlign: "center", padding: "40px", color: "#6b7280" }}>
              Loading leaderboard…
            </div>
          )}

          {!loading && error && (
            <div style={{ textAlign: "center", padding: "40px", color: "#ef4444" }}>
              {error}
            </div>
          )}

          {!loading && !error && farmers.length === 0 && (
            <div style={{ textAlign: "center", padding: "40px", color: "#6b7280" }}>
              No farmers on the leaderboard yet. Start earning points to be the first!
            </div>
          )}

          {!loading && !error &&
            farmers.map((farmer) => (
              <div
                key={farmer.id}
                className={`leaderboard-card ${
                  farmer.rank <= 3 ? `top-${farmer.rank}` : ""
                }`}
              >
                <div className="card-rank">
                  {getRankIcon(farmer.rank)}
                </div>

                <div className="card-avatar">
                  {farmer.avatar}
                </div>

                <div className="card-info">
                  <h3 className="farmer-name">{farmer.name}</h3>
                  <p className="farmer-location">
                    📍 {farmer.location}
                  </p>

                  {farmer.badges.length > 0 && (
                    <div className="farmer-badges">
                      {farmer.badges.slice(0, 3).map((badge, i) => (
                        <span key={i} className="badge">
                          {badge}
                        </span>
                      ))}
                    </div>
                  )}
                </div>

                <div className="card-points">
                  <span className="points-value">
                    {farmer.points.toLocaleString()}
                  </span>
                  <span className="points-label">
                    points
                  </span>
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
            Earn points by completing farming tasks, sharing knowledge,
            and helping the community
          </p>
          <a href="/advisor" className="cta-button">
            Start Earning Points
          </a>
        </div>
      </div>
    </div>
  );
}