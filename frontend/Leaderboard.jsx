import React, { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import {
  FaAward,
  FaChartLine,
  FaLeaf,
  FaMedal,
  FaSeedling,
  FaShieldAlt,
  FaStar,
  FaTrophy,
  FaUsers,
  FaWater,
  FaRecycle,
  FaArrowUp,
  FaRegGem,
} from "react-icons/fa";
import { collection, getDocs } from "firebase/firestore";
import { auth, db, isFirebaseConfigured } from "./lib/firebase";
import "./Leaderboard.css";

const DEFAULT_FARMERS = [
  {
    id: "demo-1",
    name: "Asha Patil",
    region: "Maharashtra",
    crop: "Cotton",
    yieldImprovement: 28,
    sustainabilityScore: 94,
    communityPoints: 420,
    ecoActions: 16,
    badges: ["Water Saver", "Top Mentor"],
    reward: "Solar irrigation voucher",
  },
  {
    id: "demo-2",
    name: "Ravi Kumar",
    region: "Punjab",
    crop: "Wheat",
    yieldImprovement: 24,
    sustainabilityScore: 88,
    communityPoints: 365,
    ecoActions: 13,
    badges: ["Soil Guardian"],
    reward: "Premium seed pack",
  },
  {
    id: "demo-3",
    name: "Meera Desai",
    region: "Gujarat",
    crop: "Groundnut",
    yieldImprovement: 22,
    sustainabilityScore: 91,
    communityPoints: 334,
    ecoActions: 15,
    badges: ["Regenerative Grower"],
    reward: "Crop insurance discount",
  },
  {
    id: "demo-4",
    name: "Suresh Yadav",
    region: "Madhya Pradesh",
    crop: "Soybean",
    yieldImprovement: 19,
    sustainabilityScore: 84,
    communityPoints: 301,
    ecoActions: 11,
    badges: ["Harvest Hero"],
    reward: "Water audit consultation",
  },
  {
    id: "demo-5",
    name: "Lakshmi Rao",
    region: "Karnataka",
    crop: "Millet",
    yieldImprovement: 17,
    sustainabilityScore: 97,
    communityPoints: 287,
    ecoActions: 18,
    badges: ["Eco Champion", "Community Mentor"],
    reward: "Biocompost starter kit",
  },
  {
    id: "demo-6",
    name: "Imran Sheikh",
    region: "Rajasthan",
    crop: "Mustard",
    yieldImprovement: 15,
    sustainabilityScore: 79,
    communityPoints: 255,
    ecoActions: 10,
    badges: ["Soil Saver"],
    reward: "Weather advisory boost",
  },
];

const RANKING_MODES = [
  { id: "overall", label: "Overall", description: "Balanced farming score" },
  { id: "yield", label: "Yield", description: "Productivity growth" },
  { id: "sustainability", label: "Sustainability", description: "Eco-friendly practice score" },
  { id: "community", label: "Community", description: "Community reputation" },
];

const BADGE_RULES = [
  { min: 90, name: "Regenerative Champion", icon: <FaTrophy /> },
  { min: 80, name: "Harvest Leader", icon: <FaAward /> },
  { min: 70, name: "Soil Guardian", icon: <FaShieldAlt /> },
  { min: 60, name: "Eco Grower", icon: <FaLeaf /> },
  { min: 0, name: "Rising Farmer", icon: <FaSeedling /> },
];

const clamp = (value, min, max) => Math.min(Math.max(value, min), max);

const safeNumber = (value, fallback = 0) => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
};

const getBadge = (overallScore) => BADGE_RULES.find((badge) => overallScore >= badge.min) || BADGE_RULES[BADGE_RULES.length - 1];

const getMetricValue = (farmer, mode) => {
  if (mode === "yield") return farmer.yieldImprovement;
  if (mode === "sustainability") return farmer.sustainabilityScore;
  if (mode === "community") return farmer.communityPoints;
  return farmer.overallScore;
};

const normalizeUser = (docSnap, index, isCurrentUser) => {
  const data = docSnap.data() || {};
  const reputation = safeNumber(data.reputation, 0);
  const sustainabilityScore = clamp(
    safeNumber(data.sustainabilityScore, safeNumber(data.greenScore, 0)),
    0,
    100,
  );
  const yieldImprovement = clamp(
    safeNumber(data.yieldImprovement, safeNumber(data.productivityScore, 0)),
    0,
    100,
  );
  const communityPoints = clamp(reputation, 0, 1000);

  const seededYield = yieldImprovement || clamp(24 + ((reputation + index * 7) % 18), 12, 48);
  const seededSustainability = sustainabilityScore || clamp(68 + ((reputation + index * 5) % 22), 52, 98);

  return {
    id: docSnap.id,
    name: data.displayName || data.farmName || `Farmer ${index + 1}`,
    region: data.address || data.region || "Community member",
    crop: data.cropType || data.primaryCrop || "Mixed crops",
    yieldImprovement: seededYield,
    sustainabilityScore: seededSustainability,
    communityPoints,
    ecoActions: safeNumber(data.ecoActions, Math.max(6, Math.round(seededSustainability / 6))),
    badges: data.badges || [],
    reward: data.reward || "Recognition badge",
    currentUser: isCurrentUser,
  };
};

const Leaderboard = () => {
  const [rankingMode, setRankingMode] = useState("overall");
  const [farmers, setFarmers] = useState(DEFAULT_FARMERS);
  const [loading, setLoading] = useState(true);
  const currentUid = auth?.currentUser?.uid;

  useEffect(() => {
    let isActive = true;

    const loadLeaderboard = async () => {
      if (!isFirebaseConfigured()) {
        if (isActive) setLoading(false);
        return;
      }

      try {
        const snapshot = await getDocs(collection(db, "users"));
        const liveFarmers = snapshot.docs.map((docSnap, index) => normalizeUser(docSnap, index, docSnap.id === currentUid || docSnap.data()?.uid === currentUid));

        const usableFarmers = liveFarmers.length >= 3 ? liveFarmers : DEFAULT_FARMERS;
        if (isActive) {
          setFarmers(usableFarmers);
          setLoading(false);
        }
      } catch (error) {
        console.error("Failed to load leaderboard data:", error);
        if (isActive) {
          setFarmers(DEFAULT_FARMERS);
          setLoading(false);
        }
      }
    };

    loadLeaderboard();
    return () => {
      isActive = false;
    };
  }, []);

  const rankedFarmers = useMemo(() => {
    const sorted = [...farmers].map((farmer) => ({
      ...farmer,
      overallScore: Math.round(
        farmer.yieldImprovement * 0.4 + farmer.sustainabilityScore * 0.35 + clamp(farmer.communityPoints / 5, 0, 100) * 0.25,
      ),
    })).sort((a, b) => {
      const diff = getMetricValue(b, rankingMode) - getMetricValue(a, rankingMode);
      if (diff !== 0) return diff;
      return b.overallScore - a.overallScore;
    });

    return sorted.map((farmer, index) => ({
      ...farmer,
      rank: index + 1,
      badge: getBadge(farmer.overallScore),
    }));
  }, [farmers, rankingMode]);

  const topThree = rankedFarmers.slice(0, 3);
  const spotlight = rankedFarmers[0];
  const currentFarmer = rankedFarmers.find((farmer) => farmer.currentUser);
  const averageYield = Math.round(rankedFarmers.reduce((sum, farmer) => sum + farmer.yieldImprovement, 0) / rankedFarmers.length || 0);
  const averageSustainability = Math.round(rankedFarmers.reduce((sum, farmer) => sum + farmer.sustainabilityScore, 0) / rankedFarmers.length || 0);
  const averageCommunity = Math.round(rankedFarmers.reduce((sum, farmer) => sum + farmer.communityPoints, 0) / rankedFarmers.length || 0);

  return (
    <div className="farmer-leaderboard-page">
      <section className="leaderboard-hero">
        <div className="leaderboard-hero-overlay" />
        <div className="leaderboard-hero-content">
          <div className="leaderboard-hero-copy">
            <div className="leaderboard-eyebrow">
              <FaStar /> Gamified farmer performance
            </div>
            <h1>Farmer Leaderboard</h1>
            <p>
              Rank growers by productivity, sustainability, and community contribution to make better farming practices visible and rewarding.
            </p>
            <div className="leaderboard-actions">
              <Link to="/community" className="leaderboard-primary-action">
                <FaUsers /> Community feed
              </Link>
              <Link to="/advisor" className="leaderboard-secondary-action">
                <FaChartLine /> Improve your score
              </Link>
            </div>
          </div>

          <div className="leaderboard-spotlight-card">
            <div className="spotlight-ribbon">Current leader</div>
            {loading ? (
              <div className="leaderboard-loading">Loading ranking data...</div>
            ) : spotlight ? (
              <>
                <div className="spotlight-rank">#{spotlight.rank}</div>
                <h2>{spotlight.name}</h2>
                <p>{spotlight.region} · {spotlight.crop}</p>
                <div className="spotlight-score-row">
                  <span>{spotlight.overallScore} overall points</span>
                  <span>{spotlight.badge.name}</span>
                </div>
                <div className="spotlight-badge">
                  {spotlight.badge.icon}
                  <span>{spotlight.badge.name}</span>
                </div>
              </>
            ) : null}
          </div>
        </div>
      </section>

      <section className="leaderboard-stats">
        <div className="leaderboard-stat-card">
          <FaArrowUp className="stat-icon" />
          <div>
            <span className="stat-value">{averageYield}%</span>
            <span className="stat-label">Average yield improvement</span>
          </div>
        </div>
        <div className="leaderboard-stat-card">
          <FaRecycle className="stat-icon" />
          <div>
            <span className="stat-value">{averageSustainability}/100</span>
            <span className="stat-label">Average sustainability score</span>
          </div>
        </div>
        <div className="leaderboard-stat-card">
          <FaMedal className="stat-icon" />
          <div>
            <span className="stat-value">{averageCommunity}</span>
            <span className="stat-label">Average community points</span>
          </div>
        </div>
      </section>

      <section className="leaderboard-layout">
        <div className="leaderboard-panel leaderboard-rankings-panel">
          <div className="panel-header">
            <div>
              <h2>Community leaderboard</h2>
              <p>See who is leading across the four gamified farming dimensions.</p>
            </div>
            <div className="ranking-toggle" role="tablist" aria-label="Leaderboard ranking mode">
              {RANKING_MODES.map((mode) => (
                <button
                  key={mode.id}
                  type="button"
                  role="tab"
                  aria-selected={rankingMode === mode.id}
                  className={`ranking-toggle-btn ${rankingMode === mode.id ? "active" : ""}`}
                  onClick={() => setRankingMode(mode.id)}
                >
                  <span>{mode.label}</span>
                  <small>{mode.description}</small>
                </button>
              ))}
            </div>
          </div>

          <div className="leaderboard-list">
            {rankedFarmers.map((farmer) => (
              <article key={farmer.id} className={`leaderboard-row ${farmer.currentUser ? "current-user" : ""}`}>
                <div className="leaderboard-rank-chip">#{farmer.rank}</div>
                <div className="leaderboard-row-main">
                  <div className="leaderboard-row-title">
                    <h3>{farmer.name}</h3>
                    {farmer.currentUser && <span className="you-badge">You</span>}
                  </div>
                  <p>{farmer.region} · {farmer.crop}</p>
                  <div className="badge-pills">
                    {farmer.badges.length > 0 ? farmer.badges.map((badge) => (
                      <span key={badge} className="badge-pill">{badge}</span>
                    )) : (
                      <span className="badge-pill muted">No rewards unlocked yet</span>
                    )}
                  </div>
                </div>

                <div className="leaderboard-metrics">
                  <div>
                    <span className="metric-value">{farmer.yieldImprovement}%</span>
                    <span className="metric-label">Yield improvement</span>
                  </div>
                  <div>
                    <span className="metric-value">{farmer.sustainabilityScore}</span>
                    <span className="metric-label">Sustainability</span>
                  </div>
                  <div>
                    <span className="metric-value">{farmer.communityPoints}</span>
                    <span className="metric-label">Community points</span>
                  </div>
                </div>

                <div className="leaderboard-progress">
                  <div className="progress-track">
                    <span style={{ width: `${clamp(farmer.overallScore, 0, 100)}%` }} />
                  </div>
                  <div className="leaderboard-row-footer">
                    <span>{farmer.badge.name}</span>
                    <span>{farmer.ecoActions} eco actions</span>
                  </div>
                </div>
              </article>
            ))}
          </div>
        </div>

        <aside className="leaderboard-panel leaderboard-side-panel">
          <div className="sidebar-card">
            <div className="sidebar-title">
              <FaShieldAlt /> Score breakdown
            </div>
            <div className="breakdown-grid">
              <div>
                <span>40%</span>
                <p>Yield improvement</p>
              </div>
              <div>
                <span>35%</span>
                <p>Sustainability</p>
              </div>
              <div>
                <span>25%</span>
                <p>Community reputation</p>
              </div>
            </div>
          </div>

          <div className="sidebar-card">
            <div className="sidebar-title">
              <FaTrophy /> Top three podium
            </div>
            <div className="podium-list">
              {topThree.map((farmer) => (
                <div key={farmer.id} className="podium-item">
                  <div className="podium-rank">#{farmer.rank}</div>
                  <div>
                    <strong>{farmer.name}</strong>
                    <p>{farmer.overallScore} pts · {farmer.badge.name}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="sidebar-card reward-card">
            <div className="sidebar-title">
              <FaRegGem /> Reward ladder
            </div>
            <div className="reward-list">
              <div>
                <FaSeedling /> 50 pts - Rising farmer badge
              </div>
              <div>
                <FaLeaf /> 70 pts - Eco grower reward
              </div>
              <div>
                <FaAward /> 80 pts - Harvest leader voucher
              </div>
              <div>
                <FaTrophy /> 90+ pts - Regenerative champion tier
              </div>
            </div>
            <p className="reward-note">
              Rewards can be connected to coupons, advisory boosts, seed discounts, or local recognition programs.
            </p>
          </div>

          <div className="sidebar-card spotlight-summary">
            <div className="sidebar-title">
              <FaWater /> Your standing
            </div>
            {currentFarmer ? (
              <>
                <h3>#{currentFarmer.rank} overall</h3>
                <p>{currentFarmer.name} is currently positioned in the live ranking with a {currentFarmer.overallScore}-point score.</p>
              </>
            ) : (
              <>
                <h3>Not ranked yet</h3>
                <p>Create a profile and start contributing in the community to unlock your place on the board.</p>
              </>
            )}
          </div>
        </aside>
      </section>
    </div>
  );
};

export default Leaderboard;
