import React, { useState, useEffect, useMemo, useCallback, memo, lazy, Suspense } from "react";
import { Link } from "react-router-dom";
import {
  FaUser,
  FaSeedling,
  FaCloudSun,
  FaChartLine,
  FaTractor,
  FaCalendarAlt,
  FaMapMarkerAlt,
  FaArrowRight,
  FaLeaf,
  FaBell,
  FaWater,
  FaBug,
  FaComments,
  FaWhatsapp,
  FaCheckCircle,
  FaBook,
  FaPhoneAlt,
  FaShieldAlt,
  FaFileInvoiceDollar,
  FaChartBar,
  FaTrophy,
  FaRobot,
  FaRecycle,
  FaUserPlus,
  FaExclamationTriangle,
} from "react-icons/fa";
import "./Dashboard.css";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  BarChart, Bar, CartesianGrid
} from "recharts";
import { getHistoricalWeatherData } from "./weather/weatherService";
import ErrorBoundary from "./ErrorBoundary";
import AsyncErrorBoundary from "./AsyncErrorBoundary";
import Loader from "./Loader";
import apiClient from "./lib/apiClient";
import { getBookmarks } from "./utils/bookmarkStorage";
import AdvisoryPanel from "./AdvisoryPanel";
import { fetchData } from "../apiClient";
import ErrorToast from "./ErrorToast";

// ============================================
// Performance Utilities
// ============================================

/**
 * Memoized selector cache to prevent unnecessary recalculations
 */
class SelectorCache {
  constructor() {
    this.cache = new Map();
  }

  memoize(key, fn, deps = []) {
    const cacheKey = JSON.stringify({ key, deps });

    if (!this.cache.has(cacheKey)) {
      this.cache.set(cacheKey, fn());
    }

    return this.cache.get(cacheKey);
  }

  clear() {
    this.cache.clear();
  }
}

const selectorCache = new SelectorCache();

/**
 * Lazy loading image component with Intersection Observer
 */
const LazyImage = memo(({ src, alt, width, height, placeholder }) => {
  const [isLoaded, setIsLoaded] = useState(false);
  const [imageSrc, setImageSrc] = useState(placeholder);
  const imgRef = React.useRef();

  useEffect(() => {
    if (!imgRef.current) return;

    const observer = new IntersectionObserver(
      entries => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            setImageSrc(src);
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.1 }
    );

    observer.observe(imgRef.current);

    return () => observer.disconnect();
  }, [src]);

  return (
    <img
      ref={imgRef}
      src={imageSrc}
      alt={alt}
      width={width}
      height={height}
      className={isLoaded ? "loaded" : "loading"}
      onLoad={() => setIsLoaded(true)}
      style={{ transition: "opacity 0.3s" }}
    />
  );
});

/**
 * Dashboard Card Component (Memoized)
 */
const DashboardCard = memo(({ title, icon: Icon, children, link, className }) => {
  return (
    <div className={`card ${className}`}>
      <div className="card-header">
        <h3>{title}</h3>
        {Icon && <Icon className="card-icon" />}
      </div>
      <div className="card-content">{children}</div>
      {link && (
        <Link to={link} className="card-link">
          Learn more <FaArrowRight />
        </Link>
      )}
    </div>
  );
});

/**
 * Virtualized list component for large datasets
 */
const VirtualizedList = memo(({ items, itemHeight, renderItem, maxHeight = 400 }) => {
  const [scrollTop, setScrollTop] = useState(0);

  const visibleRange = useMemo(() => {
    const startIdx = Math.floor(scrollTop / itemHeight);
    const visibleCount = Math.ceil(maxHeight / itemHeight);
    const endIdx = Math.min(startIdx + visibleCount + 1, items.length);

    return {
      start: Math.max(0, startIdx),
      end: endIdx,
      offset: startIdx * itemHeight
    };
  }, [scrollTop, itemHeight, maxHeight, items.length]);

  const visibleItems = useMemo(
    () => items.slice(visibleRange.start, visibleRange.end),
    [items, visibleRange]
  );

  return (
    <div
      style={{
        height: maxHeight,
        overflow: "auto",
        position: "relative"
      }}
      onScroll={(e) => setScrollTop(e.currentTarget.scrollTop)}
    >
      <div style={{ height: items.length * itemHeight, position: "relative" }}>
        <div style={{ transform: `translateY(${visibleRange.offset}px)` }}>
          {visibleItems.map((item, idx) => (
            <div key={visibleRange.start + idx} style={{ height: itemHeight }}>
              {renderItem(item, visibleRange.start + idx)}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
});

/**
 * Data cache with TTL (Time To Live)
 */
class DataCache {
  constructor(ttl = 5 * 60 * 1000) {
    this.ttl = ttl;
    this.data = new Map();
  }

  set(key, value) {
    this.data.set(key, {
      value,
      timestamp: Date.now()
    });
  }

  get(key) {
    const item = this.data.get(key);
    if (!item) return null;

    if (Date.now() - item.timestamp > this.ttl) {
      this.data.delete(key);
      return null;
    }

    return item.value;
  }

  clear() {
    this.data.clear();
  }
}

const dataCache = new DataCache();

// ============================================
// Dashboard Component
// ============================================

const formatFarmArea = (value) => {
  if (value === undefined || value === null || value === "") return "";

  const areaText = String(value).trim();
  if (/acres?|hectares?|ha\b/i.test(areaText)) return areaText;
  if (/^\d+(?:\.\d+)?$/.test(areaText)) return `${areaText} Acres`;
  return areaText;
};

export default function Dashboard({ userData }) {
  const name = userData?.displayName || "Farmer";
  const preferredLang = userData?.language || "en";
  const normalizedFarmArea = useMemo(
    () => formatFarmArea(userData?.farmArea || userData?.farmSize),
    [userData?.farmArea, userData?.farmSize]
  );
  const normalizedIrrigation = userData?.irrigationType || userData?.irrigationMethod || "";
  const nextHarvestValue = userData?.nextHarvest || userData?.harvestDate || userData?.expectedHarvest || (userData?.season ? `${userData.season} season` : "Plan with Crop Planner");
  const yieldScoreValue = userData?.yieldScore ?? userData?.yieldPredictionScore ?? userData?.estimatedYieldScore ?? (userData?.cropType ? "Use Yield Predictor" : "—");

  const [currentTime, setCurrentTime] = useState(new Date());
  const [historicalWeather, setHistoricalWeather] = useState([]);
  const [phoneNumber, setPhoneNumber] = useState(userData?.phoneNumber || "");
  const [whatsappAlerts, setWhatsappAlerts] = useState(!!userData?.whatsappAlerts);
  const [isUpdating, setIsUpdating] = useState(false);
  const [updateMsg, setUpdateMsg] = useState("");
  const [yieldData, setYieldData] = useState([]);
  const [selectedCrop, setSelectedCrop] = useState("");
  const [selectedRegion, setSelectedRegion] = useState("");
  const [selectedSeason, setSelectedSeason] = useState("");
  const [savedCrops, setSavedCrops] = useState([]);
  const [savedArticles, setSavedArticles] = useState([]);
  const mountedRef = React.useRef(true);
  const dashboardRequestRef = React.useRef(0);

  const [error, setError] = useState(null);
  const [data, setData] = useState(null);

  useEffect(() => {
    fetchData("/api/weather")
      .then(setData)
      .catch((err) => setError(err.message));
  }, []);
  
  // Memoize callback functions to prevent unnecessary re-renders
  const handlePhoneChange = useCallback((e) => {
    setPhoneNumber(e.target.value);
  }, []);

  const handleWhatsappToggle = useCallback(() => {
    setWhatsappAlerts(prev => !prev);
  }, []);

  useEffect(() => {
    if (userData) {
      setPhoneNumber(userData.phoneNumber || "");
      setWhatsappAlerts(!!userData.whatsappAlerts);
    }
  }, [userData]);

  useEffect(() => {
    // Use cached bookmarks data
    const cachedCrops = dataCache.get("bookmarks:crops");
    const cachedArticles = dataCache.get("bookmarks:articles");

    if (cachedCrops && cachedArticles) {
      setSavedCrops(cachedCrops);
      setSavedArticles(cachedArticles);
    } else {
      const crops = getBookmarks("crops");
      const articles = getBookmarks("articles");
      dataCache.set("bookmarks:crops", crops);
      dataCache.set("bookmarks:articles", articles);
      setSavedCrops(crops);
      setSavedArticles(articles);
    }
  }, []);

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 60000);
    return () => clearInterval(timer);
  }, []);

  // Memoize yield data calculation
  const processedYieldData = useMemo(() => {
    return [
      { year: "2019", crop: "Wheat", yield: 30, region: "North", season: "Rabi" },
      { year: "2020", crop: "Rice", yield: 45, region: "South", season: "Kharif" },
      { year: "2021", crop: "Wheat", yield: 50, region: "North", season: "Rabi" },
      { year: "2022", crop: "Rice", yield: 60, region: "South", season: "Kharif" },
    ];
  }, []);

  useEffect(() => {
    if (!mountedRef.current) return;

    setYieldData(processedYieldData);
  }, [processedYieldData]);

  useEffect(() => {
    const requestId = ++dashboardRequestRef.current;

    const fetchData = async () => {
      try {
        const cachedData = dataCache.get("weather:historical");

        if (cachedData) {
          if (
            mountedRef.current &&
            requestId === dashboardRequestRef.current
          ) {
            setHistoricalWeather(cachedData);
          }
          return;
        }

        const data = await getHistoricalWeatherData();

        if (
          !mountedRef.current ||
          requestId !== dashboardRequestRef.current
        ) {
          return;
        }

        dataCache.set("weather:historical", data);
        setHistoricalWeather(data);
      } catch (error) {
        console.error(error);
      }
    };

    fetchData();
  }, []);
  const handleUpdateWhatsApp = async () => {
    setIsUpdating(true);
    setUpdateMsg("");
    try {
      // Use apiClient instead of raw fetch() so the Firebase auth token is
      // automatically injected via the Axios request interceptor in
      // services/api.js.  The backend now derives user identity from the
      // verified token — we no longer send user_id from localStorage, which
      // could be spoofed to overwrite another user's subscription.
      const response = await apiClient.post("/api/whatsapp/subscribe", {
        phone_number: phoneNumber,
        name: name,
      });
      if (response.data?.success) {
      if (mountedRef.current) {
        setUpdateMsg("Settings saved successfully!");
      }
        setTimeout(() => setUpdateMsg(""), 3000);
      }
    } catch {
  if (mountedRef.current) {
    if (mountedRef.current) {
      setUpdateMsg("Error saving settings.");
    }
  }
    }
    finally {
      if (mountedRef.current) {
        setIsUpdating(false);
      }
    }
  };

  const getGreeting = () => {
    const hour = currentTime.getHours();
    if (hour < 12) return "Good Morning";
    if (hour < 17) return "Good Afternoon";
    return "Good Evening";
  };

  const getFormattedDate = () => {
    return currentTime.toLocaleDateString("en-IN", {
      weekday: "long",
      year: "numeric",
      month: "long",
      day: "numeric",
    });
  };

  const quickStats = [
    {
      label: "Primary Crop",
      value: userData?.cropType || "—",
      icon: <FaSeedling />,
      trend: userData?.season ? `${userData.season} season` : "Set up your profile",
    },
    {
      label: "Farm Area",
      value: normalizedFarmArea || "—",
      icon: <FaMapMarkerAlt />,
      trend: userData?.address || userData?.location || "Location not set",
    },
    {
      label: "Yield Score",
      value: yieldScoreValue,
      icon: <FaChartLine />,
      trend: userData?.cropType ? "Use Yield Predictor for estimate" : "Set up your profile",
    },
    {
      label: "Next Harvest",
      value: nextHarvestValue,
      icon: <FaCalendarAlt />,
      trend: userData?.season || normalizedIrrigation ? `${userData.season || normalizedIrrigation} planning` : "Set up your profile",
    },
  ];

  // Recent activity is derived from real user actions where available.
  // Static entries are clearly labelled as tips/reminders, not fabricated events.
  const cropLabel = userData?.cropType || "your crop";
  const recentActivity = [
    {
      icon: <FaCloudSun />,
      title: "Weather Alerts",
      description: "Check the Weather Alerts section for real-time conditions at your location.",
      time: "Live",
      type: "info",
    },
    {
      icon: <FaSeedling />,
      title: "Crop Health",
      description: `Use the Disease Detection tool to check ${cropLabel} health.`,
      time: "Tip",
      type: "default",
    },
    {
      icon: <FaChartLine />,
      title: "Yield Prediction",
      description: "Run the Yield Predictor to get an estimate for your current season.",
      time: "Tip",
      type: "info",
    },
    {
      icon: <FaWater />,
      title: "Irrigation Planner",
      description: "Use the Crop Planner to schedule irrigation based on your crop and soil.",
      time: "Tip",
      type: "default",
    },
    {
      icon: <FaBug />,
      title: "Pest Monitoring",
      description: "Use the Pest Detection tool to identify and manage pest risks early.",
      time: "Tip",
      type: "default",
    },
    {
      icon: <FaTractor />,
      title: "Soil Analysis",
      description: "Run a Soil Analysis to check nutrient levels and get fertilizer advice.",
      time: "Tip",
      type: "info",
    },
  ];

  // Recommendations are derived from the user's actual profile data.
  // Generic fallbacks are shown only when profile fields are missing,
  // and are clearly framed as general tips rather than personalised AI output.
  const userCrop = userData?.cropType?.toLowerCase() || "";
  const userIrrigation = normalizedIrrigation?.toLowerCase() || "";
  const recommendations = [
    {
      icon: <FaLeaf />,
      title: userIrrigation && userIrrigation !== "drip"
        ? "Consider Drip Irrigation"
        : "Optimise Your Irrigation",
      description: userIrrigation && userIrrigation !== "drip"
        ? `Switching from ${userData.irrigationType} to drip irrigation can reduce water use by up to 40% for ${userData.cropType || "most crops"}.`
        : "Review your irrigation schedule with the Crop Planner to match soil moisture needs.",
      tag: "Water Management",
    },
    {
      icon: <FaSeedling />,
      title: "Improve Soil Health",
      description: userCrop
        ? `Leguminous cover crops between ${userData.cropType} seasons can reduce fertilizer costs and improve soil nitrogen.`
        : "Adding cover crops between seasons improves soil nitrogen and reduces fertilizer costs.",
      tag: "Soil Health",
    },
    {
      icon: <FaBell />,
      title: "Plan Your Sowing Window",
      description: userData?.season
        ? `Check the Seasonal Crop Planner for the optimal sowing window for ${userData.season} crops in your region.`
        : "Use the Seasonal Crop Planner to find the best sowing window for your region and season.",
      tag: "Planning",
    },
    {
      icon: <FaChartLine />,
      title: "Track Market Prices",
      description: userCrop
        ? `Monitor live mandi prices for ${userData.cropType} in the Market Prices section to time your sale.`
        : "Check the Market Prices section for live mandi rates before deciding when to sell.",
      tag: "Market",
    },
  ];

  const quickActions = [
    { label: "AI Advisor", icon: <FaSeedling />, link: "/advisor" },
    { label: "Yield Predictor", icon: <FaChartBar />, link: "/yield-predictor" },
    { label: "Farm Autopilot", icon: <FaRobot />, link: "/smart-farm-autopilot" },
    { label: "Sustainability", icon: <FaRecycle />, link: "/sustainability-analytics" },
    { label: "Crop Planner", icon: <FaCalendarAlt />, link: "/crop-planner" },
    { label: "Community", icon: <FaComments />, link: "/community" },
    { label: "Referrals", icon: <FaUserPlus />, link: "/referrals" },
    { label: "Leaderboard", icon: <FaTrophy />, link: "/leaderboard" },
    { label: "Diseases", icon: <FaBug />, link: "/disease-awareness" },
    { label: "Helpline", icon: <FaPhoneAlt />, link: "/helpline" },
    { label: "Glossary", icon: <FaBook />, link: "/glossary" },
    { label: "Risk Index", icon: <FaShieldAlt />, link: "/risk-index" },
  ];
  const filteredData = useMemo(() => {
    return yieldData.filter((item) => {
      return (
        (selectedCrop === "" || item.crop === selectedCrop) &&
        (selectedRegion === "" || item.region === selectedRegion) &&
        (selectedSeason === "" || item.season === selectedSeason)
      );
    });
  }, [
    yieldData,
    selectedCrop,
    selectedRegion,
    selectedSeason,
  ]);

  return (
    <div className="dashboard">
      <section className="dashboard-hero">
        <div className="dashboard-hero-bg"></div>
        <div className="dashboard-hero-content">
          <div className="welcome-block">
            <div className="user-avatar">
              <FaUser />
            </div>
            <div className="welcome-text">
              <h1>{getGreeting()}, {name}</h1>
              <p className="welcome-date">{getFormattedDate()}</p>
              <p className="welcome-sub">Here is an overview of your farm activity and insights</p>
            </div>
          </div>
           <div className="quick-actions-row">
             {quickActions.map((action, idx) => (
               <Link 
                 to={action.link} 
                 key={idx} 
                 className="quick-action-btn"
                 aria-label={`Navigate to ${action.label}`}
               >
                 {action.icon}
                 <span className="notranslate" aria-hidden="true">{action.label}</span>
               </Link>
             ))}
           </div>
        </div>
      </section>

      <section className="dashboard-stats">
        {!userData?.cropType && (
          <div
            className="profile-incomplete-banner"
            style={{
              gridColumn: "1 / -1",
              display: "flex",
              alignItems: "center",
              gap: "10px",
              padding: "12px 16px",
              background: "#fffbeb",
              border: "1px solid #fde68a",
              borderRadius: "10px",
              marginBottom: "4px",
              fontSize: "0.9rem",
              color: "#92400e",
            }}
          >
            <FaExclamationTriangle />
            <span>
              Your farm profile is incomplete — some stats show "—".{" "}
              <Link to="/profile-setup" style={{ color: "#b45309", fontWeight: 600 }}>
                Complete your profile
              </Link>{" "}
              to see personalised data here.
            </span>
          </div>
        )}
        {quickStats.map((stat, idx) => (
          <div className="stat-card" key={idx}>
            <div className="stat-card-icon">{stat.icon}</div>
            <div className="stat-card-info">
              <span className="stat-card-value">{stat.value}</span>
              <span className="stat-card-label notranslate">{stat.label}</span>
              <span className="stat-card-trend">{stat.trend}</span>
            </div>
          </div>
        ))}
      </section>

      <section className="dashboard-grid">
        <AdvisoryPanel userData={userData} />

        <div className="dashboard-column">
          <div className="dashboard-section-card">
            <div className="section-card-header">
              <h2>Recent Activity</h2>
              <span className="section-badge">{recentActivity.length} updates</span>
            </div>
            <div className="activity-list">
              {recentActivity.map((item, idx) => (
                <div className="activity-item" key={idx}>
                  <div className={`activity-icon activity-${item.type}`}>
                    {item.icon}
                  </div>
                  <div className="activity-content">
                    <span className="activity-title">{item.title}</span>
                    <span className="activity-desc">{item.description}</span>
                    <span className="activity-time">{item.time}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="dashboard-column">
          <div className="dashboard-section-card">
            <div className="section-card-header">
              <h2>Recommendations</h2>
              <span className="section-badge">AI Powered</span>
            </div>
            <div className="recommendations-list">
              {recommendations.map((rec, idx) => (
                 <div 
                   className="recommendation-card" 
                   key={idx}
                   role="button"
                   tabIndex={0}
                   aria-label={`Recommendation: ${rec.title}. ${rec.description}`}
                   onKeyDown={(e) => {
                     if (e.key === 'Enter' || e.key === ' ') {
                       e.preventDefault();
                     }
                   }}
                 >
                   <div className="rec-icon" aria-hidden="true">{rec.icon}</div>
                   <div className="rec-content">
                     <div className="rec-header-row">
                       <span className="rec-title">{rec.title}</span>
                       <span className="rec-tag">{rec.tag}</span>
                     </div>
                     <p className="rec-desc">{rec.description}</p>
                   </div>
                   <FaArrowRight className="rec-arrow" aria-hidden="true" />
                 </div>
              ))}
            </div>
          </div>

          <div className="dashboard-section-card saved-items-card">
            <div className="section-card-header">
              <h2>Saved Items</h2>
              <span className="section-badge">{savedCrops.length + savedArticles.length} saved</span>
            </div>
            <div className="saved-items-grid">
              <div className="saved-items-block">
                <h3>Bookmarked Crops</h3>
                {savedCrops.length > 0 ? (
                  <ul className="saved-items-list">
                    {savedCrops.slice(0, 4).map((crop) => (
                      <li key={crop.id}>{crop.name}</li>
                    ))}
                  </ul>
                ) : (
                  <p className="saved-empty">Save crops from the Crop Guide to see them here.</p>
                )}
              </div>
              <div className="saved-items-block">
                <h3>Bookmarked Articles</h3>
                {savedArticles.length > 0 ? (
                  <ul className="saved-items-list">
                    {savedArticles.slice(0, 4).map((article) => (
                      <li key={article.id}>{article.title}</li>
                    ))}
                  </ul>
                ) : (
                  <p className="saved-empty">Save articles from the Knowledge Hub to see them here.</p>
                )}
              </div>
            </div>
          </div>

          <div className="dashboard-section-card farm-summary-card">
            <div className="section-card-header">
              <h2>Farm Overview</h2>
            </div>
            <div className="farm-summary-grid">
              <div className="farm-summary-item">
                <span className="farm-summary-label">Primary Crop</span>
                <span className="farm-summary-value">{userData?.cropType || "—"}</span>
              </div>
              <div className="farm-summary-item">
                <span className="farm-summary-label">Season</span>
                <span className="farm-summary-value">{userData?.season || "—"}</span>
              </div>
              <div className="farm-summary-item">
                <span className="farm-summary-label">Soil Type</span>
                <span className="farm-summary-value">{userData?.soilType || "—"}</span>
              </div>
              <div className="farm-summary-item">
                <span className="farm-summary-label">Irrigation</span>
                <span className="farm-summary-value">{userData?.irrigationType || "—"}</span>
              </div>
              <div className="farm-summary-item">
                <span className="farm-summary-label">Region</span>
                <span className="farm-summary-value">{userData?.address || userData?.location || "—"}</span>
              </div>
              <div className="farm-summary-item">
                <span className="farm-summary-label">Language</span>
                <span className="farm-summary-value">{preferredLang.toUpperCase()}</span>
              </div>
            </div>
            <Link to="/advisor" className="farm-cta-btn">
              Get AI Advice <FaArrowRight />
            </Link>
          </div>


          <div className="dashboard-section-card whatsapp-settings-card">
            <div className="section-card-header">
              <h2><FaWhatsapp /> WhatsApp Alerts</h2>
              <span className={`status-dot ${whatsappAlerts ? "status-active" : ""}`}></span>
            </div>
            <div className="whatsapp-settings-body">
              <p className="settings-intro">Receive real-time weather and pest alerts on your phone.</p>
              <div className="input-group">
                <label>Phone Number (with code)</label>
                <input 
                  type="text" 
                  placeholder="+91 9876543210" 
                  value={phoneNumber} 
                  onChange={(e) => setPhoneNumber(e.target.value)}
                  aria-label="Phone number with country code"
                />
              </div>
              <div className="checkbox-group">
                <input 
                  type="checkbox" 
                  id="wa-toggle" 
                  checked={whatsappAlerts} 
                  onChange={(e) => setWhatsappAlerts(e.target.checked)}
                />
                <label htmlFor="wa-toggle">Enable Real-time Alerts</label>
              </div>
              <button 
                className={`save-wa-btn ${isUpdating ? "loading" : ""}`} 
                onClick={handleUpdateWhatsApp}
                disabled={isUpdating}
              >
                {isUpdating ? "Saving..." : "Save Settings"}
              </button>
              {updateMsg && (
                 <p 
                   className={`update-msg ${updateMsg.includes("Error") ? "error" : "success"}`}
                   role="status"
                   aria-live="polite"
                 >
                   {updateMsg.includes("success") && <FaCheckCircle aria-hidden="true" />} {updateMsg}
                 </p>
              )}
            </div>
          </div>
        </div>
      </section>
      <section className="dashboard-section-card" style={{ marginTop: "30px" }}>
        <div className="section-card-header">
          <h2><FaChartBar /> Crop Yield Insights</h2>
          <span className="section-badge">Analytics</span>
        </div>

        <p style={{ color: "#6b7280", marginBottom: "20px" }}>
          Visual trends and comparison of crop yield over time
        </p>

        {/* 🔽 FILTERS HERE */}
        <div style={{ display: "flex", gap: "12px", marginBottom: "20px" }}>
          <select 
            value={selectedCrop}
            onChange={(e) => setSelectedCrop(e.target.value)}
            style={{ padding: "8px", borderRadius: "6px" }}
            aria-label="Filter by crop"
          >
            <option value="">All Crops</option>
            <option value="Wheat">Wheat</option>
            <option value="Rice">Rice</option>
          </select>

          <select 
            value={selectedRegion}
            onChange={(e) => setSelectedRegion(e.target.value)}
            style={{ padding: "8px", borderRadius: "6px" }}
            aria-label="Filter by region"
          >
            <option value="">All Regions</option>
            <option value="North">North</option>
            <option value="South">South</option>
          </select>

          <select 
            value={selectedSeason}
            onChange={(e) => setSelectedSeason(e.target.value)}
            style={{ padding: "8px", borderRadius: "6px" }}
            aria-label="Filter by season"
          >
            <option value="">All Seasons</option>
            <option value="Kharif">Kharif</option>
            <option value="Rabi">Rabi</option>
          </select>
          <button
            onClick={() => {
              setSelectedCrop("");
              setSelectedRegion("");
              setSelectedSeason("");
            }}
            style={{
              padding: "8px 14px",
              borderRadius: "8px",
              border: "none",
              background: "#22c55e",
              color: "#fff",
              fontWeight: "500",
              cursor: "pointer",
            }}
          >
            Reset
          </button>
        </div>

        {/* CONDITION START */}
        {yieldData.length === 0 ? (
          <Loader message="Loading chart data..." />
        ) : (
          /* GRID */
          <div
            style={{
              display: "grid",
              gridTemplateColumns: window.innerWidth > 768 ? "1fr 1fr" : "1fr",
              gap: "24px",
            }}
          >
            {/* 📈 Line Chart */}
            <ErrorBoundary>
              <div
                style={{
                  background: "#ffffff",
                  borderRadius: "12px",
                  padding: "16px",
                  boxShadow: "0 2px 8px rgba(0,0,0,0.08)",
                }}
              >
                <h4 style={{ marginBottom: "10px" }}>Yield Trend</h4>
                <div 
                   style={{ width: "100%", height: 350 }}
                   role="img"
                   aria-label="Line chart showing crop yield trend over years. The trend shows a steady increase from 30 in 2019 to 60 in 2022."
                 >
                  {filteredData.length === 0 ? (
                    <div style={{ textAlign: "center", padding: "40px" }}>
                      No data found. Try changing filters.
                    </div>
                  ) : (
                    <ResponsiveContainer>
                      <LineChart data={filteredData}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
                        <XAxis dataKey="year" axisLine={false} tickLine={false} />
                        <YAxis axisLine={false} tickLine={false} />
                        <Tooltip isAnimationActive={false} />
                        <Line
                          type="monotone"
                          dataKey="yield"
                          stroke="#22c55e"
                          strokeWidth={3}
                          dot={false}
                          isAnimationActive={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  )}
                </div>
              </div>
            </ErrorBoundary>

            {/* 📊 Bar Chart */}
            <ErrorBoundary>
              <div
                style={{
                  background: "#ffffff",
                  borderRadius: "12px",
                  padding: "16px",
                  boxShadow: "0 2px 8px rgba(0,0,0,0.08)",
                }}
              >
                <h4 style={{ marginBottom: "10px" }}>Crop Comparison</h4>
                <div 
                   style={{ width: "100%", height: 350 }}
                   role="img"
                   aria-label="Bar chart comparing yields across different crops."
                 >
                  <ResponsiveContainer>
                    <BarChart data={yieldData}>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
                      <XAxis dataKey="crop" axisLine={false} tickLine={false} />
                      <YAxis axisLine={false} tickLine={false} />
                      <Tooltip isAnimationActive={false} />
                      <Bar dataKey="yield" fill="#10b981" isAnimationActive={false} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </ErrorBoundary>
          </div>
        )}
        {/* CONDITION END */}
      </section>
      <section className="dashboard-section-card" style={{ marginTop: "30px" }}>
        <div className="section-card-header">
          <h2><FaCloudSun /> Historical Weather Trends</h2>
          <span className="section-badge">Weather</span>
        </div>

        <p style={{ color: "#6b7280", marginBottom: "20px" }}>
          Temperature trends based on past years to improve crop decisions
        </p>

        {/* Weather Chart */}
        <AsyncErrorBoundary>
          <div style={{ width: "100%", height: 350 }}>
            {historicalWeather.length === 0 ? (
              <Loader message="Loading weather data..." />
            ) : (
              <ResponsiveContainer>
                <LineChart data={historicalWeather}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
                  <XAxis dataKey="year" axisLine={false} tickLine={false} />
                  <YAxis axisLine={false} tickLine={false} />
                  <Tooltip isAnimationActive={false} />
                  <Line
                    type="monotone"
                    dataKey="temp"
                    stroke="#f59e0b"
                    strokeWidth={3}
                    dot={false}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
        </AsyncErrorBoundary>

        {/* Insight */}
        <div style={{ marginTop: "15px", fontWeight: "500", color: "#374151" }}>
          <FaSeedling /> Insight: {
            historicalWeather.length > 0
              ? (
                  historicalWeather.reduce((sum, d) => sum + d.rainfall, 0) /
                  historicalWeather.length
                ) > 140
                ? "Rice is suitable based on historical rainfall trends"
                : "Wheat is more suitable based on climate trends"
              : "Analyzing data..."
          }
        </div>
      </section>

      <div>
      <h2>Dashboard</h2>
      {data && <pre>{JSON.stringify(data, null, 2)}</pre>}
      <ErrorToast message={error} onClose={() => setError(null)} />
    </div>
    </div>
  );
}
// Optimized Dashboard.jsx for performance
