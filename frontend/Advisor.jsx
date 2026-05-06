import React, { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import "./Advisor.css";
import WeatherCard from "./weather/WeatherCard";
import Forecast from "./Forecast";
import SoilChatbot from "./SoilChatbot";
import SoilAnalysis from "./SoilAnalysis";
import SoilGuide from "./SoilGuide";
import IrrigationGuidance from "./IrrigationGuidance";
import CropProfitCalculator from "./CropProfitCalculator";
import FarmingMap from "./FarmingMap";
import FertilizerRecommendation from "./FertilizerRecommendation";
import LastUpdated from "./LastUpdated";
import AgriMarketplace from "./AgriMarketplace";
import AgriLMS from "./AgriLMS";
import BankReports from "./BankReports";
import QRTraceability from "./QRTraceability";
import FarmPlanner3D from "./FarmPlanner3D";
import FarmDiary from "./FarmDiary";
import CropDiseaseDetection from "./CropDiseaseDetection";
import PestManagement from "./PestManagement";
import SeedVerifier from "./SeedVerifier";
import ClimateSimulator from "./ClimateSimulator";
import RAGAdvisor from "./RAGAdvisor";
import GreenPractices from "./GreenPractices";
import { Leaf } from "lucide-react";

import CropRotation from "./CropRotation";
import P2PChat from "./P2PChat";
import GeoAlertMesh from "./GeoAlertMesh";
import SmartCropRecommendation from "./SmartCropRecommendation";
import PersonalizedAdvisory from "./PersonalizedAdvisory";
import {
  Sun,
  Droplets,
  IndianRupee,
  Sprout,
  Languages,
  WifiOff,
  Landmark,
  Calendar,
  MessageSquare,
  Info,
  Map,
  FlaskConical,
  Layers,
  ShoppingCart,
  Book,
  CloudSun,
  QrCode,
  Award,
  Star,
  ThumbsUp,
  X,
  AlertTriangle,
  TrendingDown
} from "lucide-react";
import { FaSync } from "react-icons/fa";
import { useAdvisorStore } from "./stores/advisorStore";
import { usePerformanceStore } from "./stores/performanceStore";
import { useYieldPrediction } from "./hooks/useYieldPrediction";
import { auth, db } from "./lib/firebase";
import { generateBankPDF, generateCSV } from "./utils/exportService";
import { doc, onSnapshot } from "firebase/firestore";

export default function Advisor({ userData }) {
  const navigate = useNavigate();
  const WEATHER_API_KEY = import.meta.env.VITE_OPENWEATHER_API_KEY;
  const WEATHER_CACHE_KEY = "advisorWeatherCache";
  
  const {
    farmers,
    setFarmers,
    crops,
    setCrops,
    languages,
    setLanguages,
    showWeather,
    setShowWeather,
    showSoilChatbot,
    setShowSoilChatbot,
    showSoilAnalysis,
    setShowSoilAnalysis,
    showSoilGuide,
    setShowSoilGuide,
    showFertilizerPopup,
    setShowFertilizerPopup,
    showComingSoon,
    setShowComingSoon,
    showIrrigation,
    setShowIrrigation,
    showProfitCalculator,
    setShowProfitCalculator,
    showFarmingMap,
    setShowFarmingMap,
    showCropDiseaseDetection,
    setShowCropDiseaseDetection,
    showPestManagement,
    setShowPestManagement,
    showAgriMarketplace,
    setShowAgriMarketplace,
    showAgriLMS,
    setShowAgriLMS,
    showQRTraceability,
    setShowQRTraceability,
    showFarmPlanner3D,
    setShowFarmPlanner3D,
    showFarmDiary,
    setShowFarmDiary,
    showCropRotation,
    setShowCropRotation,
    showForecast,
    setShowForecast,
    showExpertStatus,
    setShowExpertStatus,
    showBankReport,
    setShowBankReport,
    showP2PChat,
    setShowP2PChat,
    showSmartCropRecommendation,
    setShowSmartCropRecommendation,
    showSeedVerifier,
    setShowSeedVerifier,
    showGeoAlerts,
    setShowGeoAlerts,
    showClimateSimulator,
    setShowClimateSimulator,
    showRAGAdvisor,
    setShowRAGAdvisor,
    showGreenPractices,
    setShowGreenPractices,
  } = useAdvisorStore();

  const { liteMode } = usePerformanceStore();

  const {
    yieldForm,
    updateYieldFormField,
    yieldPrediction,
    yieldLastUpdated,
    yieldError,
    yieldLoading,
    showYieldPopup,
    setShowYieldPopup,
    fetchYield,
    closeYieldPopup,
  } = useYieldPrediction();

  const [weatherStatus, setWeatherStatus] = useState("idle");
  const [weatherError, setWeatherError] = useState("");
  const [weatherData, setWeatherData] = useState(null);
  const [weatherLocation, setWeatherLocation] = useState("");
  const [weatherLastUpdated, setWeatherLastUpdated] = useState(null);
  const [locationQuery, setLocationQuery] = useState("");
  const [coords, setCoords] = useState(null);
  const [userProfile, setUserProfile] = useState(null);

  useEffect(() => {
    // Priority: auth.currentUser, then fallback to localStorage
    const uid = auth?.currentUser?.uid || localStorage.getItem("userId");
    
    if (uid) {
      const unsubscribe = onSnapshot(doc(db, "users", uid), (doc) => {
        if (doc.exists()) {
          setUserProfile(doc.data());
        }
      });
      return () => unsubscribe();
    }
  }, [auth?.currentUser]);

  /* Animate stats on mount */
  useEffect(() => {
    const interval = setInterval(() => {
      const state = useAdvisorStore.getState();
      if (state.farmers < 50000) setFarmers(state.farmers + 500);
      if (state.crops < 120) setCrops(state.crops + 2);
      if (state.languages < 12) setLanguages(state.languages + 1);
    }, 50);
    return () => clearInterval(interval);
  }, [setFarmers, setCrops, setLanguages]);

  useEffect(() => {
    try {
      const cached = localStorage.getItem(WEATHER_CACHE_KEY);
      if (!cached) return;
      const parsed = JSON.parse(cached);
      if (!parsed?.timestamp || !parsed?.data) return;
      const ageMinutes = (Date.now() - parsed.timestamp) / 60000;
      if (ageMinutes <= 30) {
        setWeatherData(parsed.data);
        setWeatherLocation(parsed.location || "");
        setWeatherLastUpdated(parsed.timestamp);
        setWeatherStatus("ready");
      }
    } catch {
      localStorage.removeItem(WEATHER_CACHE_KEY);
    }
  }, []);

  const advisories = useMemo(() => {
    if (!weatherData?.daily?.length) return [];
    const daily = weatherData.daily.slice(0, 7);
    const advisoriesList = [];

    const heatDays = daily.filter((day) => day?.temp?.max >= 38);
    if (heatDays.length >= 2) {
      advisoriesList.push({
        type: "heat",
        title: "Heatwave risk",
        message: "Plan irrigation during early hours and protect seedlings with shade nets.",
      });
    }

    const frostDays = daily.filter((day) => day?.temp?.min <= 4);
    if (frostDays.length > 0) {
      advisoriesList.push({
        type: "frost",
        title: "Frost risk",
        message: "Cover sensitive crops at night and avoid late evening irrigation.",
      });
    }

    const heavyRainDays = daily.filter((day) =>
      day?.pop >= 0.7 && ["Rain", "Thunderstorm"].includes(day?.weather?.[0]?.main)
    );
    if (heavyRainDays.length > 0) {
      advisoriesList.push({
        type: "rain",
        title: "Heavy rain alert",
        message: "Delay fertilizer application and ensure proper field drainage.",
      });
    }

    const dryStretch = daily.filter((day) => day?.pop <= 0.2).length >= 3;
    if (dryStretch) {
      advisoriesList.push({
        type: "dry",
        title: "Dry spell likely",
        message: "Consider light irrigation cycles and mulch to retain soil moisture.",
      });
    }

    return advisoriesList;
  }, [weatherData]);

  const fetchWeather = async ({ latitude, longitude, label }) => {
    if (!WEATHER_API_KEY) {
      setWeatherStatus("error");
      setWeatherError("Weather API key is missing. Add VITE_OPENWEATHER_API_KEY to your env.");
      return;
    }

    setWeatherStatus("loading");
    setWeatherError("");
    const controller = new AbortController();
    const { signal } = controller;

    try {
      const url = new URL("https://api.openweathermap.org/data/2.5/onecall");
      url.searchParams.set("lat", latitude);
      url.searchParams.set("lon", longitude);
      url.searchParams.set("exclude", "minutely,hourly,alerts");
      url.searchParams.set("units", "metric");
      url.searchParams.set("appid", WEATHER_API_KEY);

      const response = await fetch(url.toString(), { signal });
      if (!response.ok) {
        throw new Error(`Weather API error (${response.status})`);
      }

      const data = await response.json();
      const timestamp = Date.now();
      setWeatherData(data);
      setWeatherLocation(label || weatherLocation);
      setWeatherLastUpdated(timestamp);
      setWeatherStatus("ready");

      localStorage.setItem(
        WEATHER_CACHE_KEY,
        JSON.stringify({
          timestamp,
          data,
          location: label || weatherLocation,
        })
      );
    } catch (error) {
      if (error?.name === "AbortError") return;
      setWeatherStatus("error");
      setWeatherError(error?.message || "Failed to load weather data.");
    }
  };

  const handleUseMyLocation = () => {
    if (!navigator.geolocation) {
      setWeatherStatus("error");
      setWeatherError("Geolocation is not supported in this browser.");
      return;
    }

    setWeatherStatus("loading");
    navigator.geolocation.getCurrentPosition(
      (position) => {
        const { latitude, longitude } = position.coords;
        setCoords({ latitude, longitude });
        fetchWeather({
          latitude,
          longitude,
          label: "Current location",
        });
      },
      () => {
        setWeatherStatus("error");
        setWeatherError("Unable to access your location. Please search manually.");
      },
      { enableHighAccuracy: true, timeout: 10000 }
    );
  };

  const handleLocationSearch = async (event) => {
    event.preventDefault();
    if (!locationQuery.trim()) return;
    if (!WEATHER_API_KEY) {
      setWeatherStatus("error");
      setWeatherError("Weather API key is missing. Add VITE_OPENWEATHER_API_KEY to your env.");
      return;
    }

    setWeatherStatus("loading");
    setWeatherError("");
    try {
      const geoUrl = new URL("https://api.openweathermap.org/geo/1.0/direct");
      geoUrl.searchParams.set("q", locationQuery);
      geoUrl.searchParams.set("limit", "1");
      geoUrl.searchParams.set("appid", WEATHER_API_KEY);

      const response = await fetch(geoUrl.toString());
      if (!response.ok) {
        throw new Error(`Location lookup failed (${response.status})`);
      }

      const results = await response.json();
      if (!results?.length) {
        throw new Error("Location not found. Try a nearby city or district.");
      }

      const match = results[0];
      const label = [match.name, match.state, match.country].filter(Boolean).join(", ");
      setCoords({ latitude: match.lat, longitude: match.lon });
      fetchWeather({ latitude: match.lat, longitude: match.lon, label });
    } catch (error) {
      setWeatherStatus("error");
      setWeatherError(error?.message || "Failed to search location.");
    }
  };

  const formatTemp = (value) => `${Math.round(value)}°C`;
  const formatDay = (timestamp) =>
    new Date(timestamp * 1000).toLocaleDateString(undefined, {
      weekday: "short",
      day: "numeric",
      month: "short",
    });

  const getNextBadgeThreshold = (points) => {
    if (points < 50) return { threshold: 50, name: "Active Contributor", icon: "🥉" };
    if (points < 200) return { threshold: 200, name: "Farming Expert", icon: "🥈" };
    if (points < 500) return { threshold: 500, name: "Master Agriculturist", icon: "🥇" };
    return { threshold: points, name: "Maximum Rank", icon: "💎" };
  };

  const currentReputation = userProfile?.reputation || 0;
  const nextBadge = getNextBadgeThreshold(currentReputation);
  const progressPercent = Math.min((currentReputation / nextBadge.threshold) * 100, 100);

  return (
    <section className="advisor">
      <div className="floating-icons">
        <span>🌱</span>
        <span>☀️</span>
        <span>💧</span>
        <span>₹</span>
      </div>

      <div className="advisor-hero">
        <h1 className="fade-in">🌱 <span className="notranslate">AI-Powered Agricultural Advisor</span></h1>
        <p className="fade-in">
          Personalized guidance for <span className="highlight">weather</span>,{" "}
          <span className="highlight">markets</span>, and{" "}
          <span className="highlight">soil health</span>.
        </p>
        <button
          className="get-started shine"
          onClick={() => setShowSoilChatbot(true)}
          aria-label="Get Started with AI Soil Advisor"
        >
          🚀 <span className="notranslate" aria-hidden="true">Get Started</span>
        </button>
      </div>

      <div className="advisor-stats">
        <div className="stat">
          <h2><span className="stat-number">{farmers.toLocaleString()}</span>{farmers >= 50000 && <span className="stat-plus">+</span>}</h2>
          <p><span className="notranslate">Farmers Connected</span></p>
        </div>
        <div className="stat">
          <h2><span className="stat-number">{crops}</span>{crops >= 120 && <span className="stat-plus">+</span>}</h2>
          <p><span className="notranslate">Crops Analyzed</span></p>
        </div>
        <div className="stat">
          <h2><span className="stat-number">{languages}</span>{languages >= 12 && <span className="stat-plus">+</span>}</h2>
          <p><span className="notranslate">Languages Available</span></p>
        </div>
      </div>

      <PersonalizedAdvisory
        userProfile={userProfile}
        weatherData={weatherData}
      />

      <br />
      <br />

      <div className="advisor-highlights">
        <h2 className="slide-in">✨ <span className="notranslate">Features</span></h2>
        <br />
        <br />
        <div className="cards">
          <div
            className="card reveal"
            style={{ cursor: "pointer" }}
            role="button"
            tabIndex={0}
            onClick={() => navigate("/crop-planner")}
            onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') navigate("/crop-planner"); }}
            aria-label="Seasonal Crop Planner: Plan your crops throughout the year"
          >
            <div className="icon" aria-hidden="true">
              <Calendar size={32} strokeWidth={2} />
            </div>
            <h3><span className="notranslate">Seasonal Crop Planner</span></h3>
            <p>Plan your crops throughout the year with seasonal recommendations and crop rotation cycles.</p>
          </div>

          <div
            className="card reveal"
            role="button"
            tabIndex={0}
            onClick={() => setShowWeather(true)}
            onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setShowWeather(true); }}
            aria-label="Weather Intelligence: Hyperlocal weather forecasts and alerts"
          >
            <div className="icon" aria-hidden="true">
              <Sun size={32} strokeWidth={2} />
            </div>
            <h3><span className="notranslate">Weather Intelligence</span></h3>
            <p>Get hyperlocal weather forecasts, alerts, and crop-specific advisories.</p>
          </div>
          
          <div
            className="card reveal"
            role="button"
            tabIndex={0}
            onClick={() => setShowForecast(true)}
            onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setShowForecast(true); }}
            aria-label="7-Day Forecast: Detailed weekly weather outlook"
          >
            <div className="icon" aria-hidden="true">
              <CloudSun size={32} strokeWidth={2} />
            </div>
            <h3><span className="notranslate">7-Day Forecast</span></h3>
            <p>Detailed weekly weather outlook to plan your farming activities.</p>
          </div>

          <div className="card reveal" role="button" tabIndex={0} onClick={() => navigate("/community")} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') navigate("/community"); }} aria-label="Farmer Community: Connect and share tips">
            <div className="icon" aria-hidden="true">
              <MessageSquare size={32} strokeWidth={2} />
            </div>
            <h3><span className="notranslate">Farmer Community</span></h3>
            <p>
              Connect, share tips, and learn from other farmers in your region.
            </p>
          </div>

          <div className="card reveal" role="button" tabIndex={0} onClick={() => navigate("/helpline")} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') navigate("/helpline"); }} aria-label="Emergency Helpline: Get support">
            <div className="icon" aria-hidden="true">
              <Landmark size={32} strokeWidth={2} />
            </div>
            <h3><span className="notranslate">Emergency Helpline</span></h3>
            <p>
              Quick access to emergency farming support and expert advice.
            </p>
          </div>

          <div className="card reveal" role="button" tabIndex={0} onClick={() => navigate("/blog")} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') navigate("/blog"); }} aria-label="Knowledge Blog: Farming articles">
            <div className="icon" aria-hidden="true">
              <Book size={32} strokeWidth={2} />
            </div>
            <h3><span className="notranslate">Knowledge Blog</span></h3>
            <p>
              Read articles on crop management, weather, and farming best practices.
            </p>
          </div>

          <div className="card reveal" role="button" tabIndex={0} onClick={() => navigate("/disease-awareness")} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') navigate("/disease-awareness"); }} aria-label="Crop Disease Awareness: Learn remedies">
            <div className="icon" aria-hidden="true">
              <Info size={32} strokeWidth={2} />
            </div>
            <h3><span className="notranslate">Crop Disease Awareness</span></h3>
            <p>
              Learn about crop diseases and remedies for better farming.
            </p>
          </div>

          <div className="card reveal" role="button" tabIndex={0} onClick={() => setShowIrrigation(true)} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setShowIrrigation(true); }} aria-label="Irrigation Guidance: Water-saving tips">
            <div className="icon" aria-hidden="true">
              <Droplets size={32} strokeWidth={2} />
            </div>
            <h3><span className="notranslate">Irrigation Guidance</span></h3>
            <p>
              Water-saving tips and irrigation schedules tailored to your crops.
            </p>
          </div>

          <div className="card reveal" role="button" tabIndex={0} onClick={() => navigate("/market-prices")} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') navigate("/market-prices"); }} aria-label="Market Price Guidance: Price trends">
            <div className="icon" aria-hidden="true">
              <IndianRupee size={32} strokeWidth={2} />
            </div>
            <h3><span className="notranslate">Market Price Guidance</span></h3>
            <p>
              Market trends and price alerts to help you sell at the best time.
            </p>
          </div>

          <div className="card reveal" role="button" tabIndex={0} onClick={() => setShowSoilChatbot(true)} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setShowSoilChatbot(true); }} aria-label="Soil Health: AI Chatbot analysis">
            <div className="icon" aria-hidden="true">
              <Sprout size={32} strokeWidth={2} />
            </div>
            <h3><span className="notranslate">Soil Health</span></h3>
            <p>Get soil analysis & recommendations via AI chatbot.</p>
          </div>

          <div
            className="card reveal"
            style={{ cursor: "pointer" }}
            role="button"
            tabIndex={0}
            onClick={() => setShowSoilAnalysis(true)}
            onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setShowSoilAnalysis(true); }}
            aria-label="Soil Analysis: NPK nutrient analysis"
          >
            <div className="icon" aria-hidden="true">
              <FlaskConical size={32} strokeWidth={2} />
            </div>
            <h3><span className="notranslate">Soil Analysis</span></h3>
            <p>Analyze NPK nutrients and get personalized crop & fertilizer recommendations.</p>
          </div>

          <div
            className="card reveal"
            style={{ cursor: "pointer" }}
            role="button"
            tabIndex={0}
            onClick={() => setShowSoilGuide(true)}
            onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setShowSoilGuide(true); }}
            aria-label="Soil Type Guide: Explore soil types"
          >
            <div className="icon" aria-hidden="true">
              <Layers size={32} strokeWidth={2} />
            </div>
            <h3>Soil Type Guide</h3>
            <p>Explore major soil types in India and find the most suitable crops for your land.</p>
          </div>

          <div className="card reveal" role="button" tabIndex={0} onClick={() => setShowCropDiseaseDetection(true)} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setShowCropDiseaseDetection(true); }} aria-label="Crop Disease Detection: Upload images">
            <div className="icon" aria-hidden="true">🌿</div>
            <h3><span className="notranslate">Crop Disease Detection</span></h3>
            <p>Upload plant images to detect diseases and get remedies.</p>
          </div>

          <div className="card reveal" role="button" tabIndex={0} onClick={() => setShowFertilizerPopup(true)} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setShowFertilizerPopup(true); }} aria-label="Fertilizer Recommendations: Plan your nutrition">
            <div className="icon" aria-hidden="true">🌾</div>
            <h3><span className="notranslate">Fertilizer Recommendations</span></h3>
            <p>Get a crop-aware fertilizer plan based on soil pH and nutrient status.</p>
          </div>

          <div className="card reveal" role="button" tabIndex={0} onClick={() => setShowComingSoon(true)} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setShowComingSoon(true); }} aria-label="Offline Access: Use anytime">
            <div className="icon" aria-hidden="true">
              <WifiOff size={32} strokeWidth={2} />
            </div>
            <h3><span className="notranslate">Offline Access</span></h3>
            <p>Use the app anytime, even without internet connectivity.</p>
          </div>

          <div className="card reveal" role="button" tabIndex={0} onClick={() => setShowPestManagement(true)} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setShowPestManagement(true); }} aria-label="Pest Management: Early warnings">
            <div className="icon" aria-hidden="true">🐛</div>
            <h3><span className="notranslate">Pest Management</span></h3>
            <p>Early warnings & organic pest control tips.</p>
          </div>

          <div className="card reveal" role="button" tabIndex={0} onClick={() => setShowYieldPopup(true)} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setShowYieldPopup(true); }} aria-label="Yield Prediction: AI-based forecast">
            <div className="icon" aria-hidden="true">📊</div>
            <h3><span className="notranslate">Yield Prediction</span></h3>
            <p>AI predicts crop yield based on soil & weather data.</p>
          </div>

          <div className="card reveal" role="button" tabIndex={0} onClick={() => navigate("/schemes")} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') navigate("/schemes"); }} aria-label="Govt Schemes: Financial support">
            <div className="icon" aria-hidden="true">
              <Landmark size={32} strokeWidth={2} />
            </div>
            <h3><span className="notranslate">Govt Schemes</span></h3>
            <p>Direct subsidies, insurance, and financial benefits for farmers.</p>
          </div>

          {(userData?.role === "vendor" || userData?.role === "admin") && (
            <div className="card reveal" role="button" tabIndex={0} onClick={() => setShowAgriMarketplace(true)} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setShowAgriMarketplace(true); }} aria-label="Agri Marketplace: Equipment rental">
              <div className="icon" aria-hidden="true">🚜</div>
              <h3><span className="notranslate">Agri Marketplace</span></h3>
              <p>Rent or list farm equipment locally. Save costs and earn extra.</p>
            </div>
          )}

          <div className="card reveal" role="button" tabIndex={0} onClick={() => setShowAgriLMS(true)} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setShowAgriLMS(true); }} aria-label="Agri-LMS Academy: Online courses">
            <div className="icon" aria-hidden="true">🎓</div>
            <h3><span className="notranslate">Agri-LMS Academy</span></h3>
            <p>Access video tutorials on modern farming and earn completion certificates.</p>
          </div>

          <div className="card reveal" role="button" tabIndex={0} onClick={() => setShowQRTraceability(true)} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setShowQRTraceability(true); }} aria-label="QR-Farm Traceability: Trace your produce">
            <div className="icon" aria-hidden="true">🔍</div>
            <h3><span className="notranslate">QR-Farm Traceability</span></h3>
            <p>Generate QR codes for your produce. Let customers trace their food from farm to table.</p>
          </div>

          {(userData?.role === "vendor" || userData?.role === "admin") && (
            <div 
              className="card reveal" 
              role="button" 
              tabIndex={0} 
              onClick={() => setShowSeedVerifier(true)} 
              onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setShowSeedVerifier(true); }} 
              aria-label="Vision-Lite: Seed Authenticity Verifier"
            >
              <div className="icon" aria-hidden="true">
                <QrCode size={32} strokeWidth={2} />
              </div>
              <h3><span className="notranslate">Vision-Lite: Seed Verifier</span></h3>
              <p>Scan seed packets to verify authenticity and prevent counterfeit usage.</p>
            </div>
          )}

          <div className="card reveal" role="button" tabIndex={0} onClick={() => setShowFarmPlanner3D(true)} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setShowFarmPlanner3D(true); }} aria-label="3D Farm Planner: Interactive design">
            <div className="icon" aria-hidden="true">🗺️</div>
            <h3><span className="notranslate">3D Farm Planner</span></h3>
            <p>Design your farm layout in interactive 3D. Optimize land usage and irrigation.</p>
          </div>

          <div className="card reveal" role="button" tabIndex={0} onClick={() => setShowProfitCalculator(true)} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setShowProfitCalculator(true); }} aria-label="Profit Calculator: ROI analysis">
            <div className="icon" aria-hidden="true">💰</div>
            <h3><span className="notranslate">Profit Calculator</span></h3>
            <p>Calculate your crop profits and ROI before planting.</p>
          </div>

          <div
            className="card reveal"
            style={{ cursor: "pointer" }}
            role="button"
            tabIndex={0}
            onClick={() => setShowFarmingMap(true)}
            onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setShowFarmingMap(true); }}
            aria-label="Farming Map: Interactive farm viewer"
          >
            <div className="icon" aria-hidden="true">
              <Map size={32} strokeWidth={2} />
            </div>
            <h3><span className="notranslate">Farming Map</span></h3>
            <p>View your fields, weather data, and crop locations on an interactive map.</p>
          </div>

          <div className="card reveal" role="button" tabIndex={0} onClick={() => navigate("/calendar")} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') navigate("/calendar"); }} aria-label="Activity Calendar: Task reminders">
            <div className="icon" aria-hidden="true">
              <Calendar size={32} strokeWidth={2} />
            </div>
            <h3><span className="notranslate">Activity Calendar</span></h3>
            <p>Schedule sowing, watering, and harvesting with reminders.</p>
          </div>

          <div className="card reveal" role="button" tabIndex={0} onClick={() => navigate("/share-feedback")} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') navigate("/share-feedback"); }} aria-label="Share Feedback: Help us improve">
            <div className="icon" aria-hidden="true">
              <MessageSquare size={32} strokeWidth={2} />
            </div>
            <h3><span className="notranslate">Share Feedback</span></h3>
            <p>Help us improve <span className="notranslate" translate="no">Fasal Saathi</span> with your valuable suggestions.</p>
          </div>

          <div className="card reveal" role="button" tabIndex={0} onClick={() => setShowFarmDiary(true)} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setShowFarmDiary(true); }} aria-label="Digital Farm Diary: Log activity">
            <div className="icon" aria-hidden="true">
              <Book size={32} strokeWidth={2} />
            </div>
            <h3><span className="notranslate">Digital Farm Diary</span></h3>
            <p>Log daily farming activities, set task reminders, and export records as PDF reports.</p>
          </div>

          <div className="card reveal" role="button" tabIndex={0} onClick={() => setShowCropRotation(true)} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setShowCropRotation(true); }} aria-label="Crop Rotation: Soil health optimization">
            <div className="icon" aria-hidden="true">
              <Layers size={32} strokeWidth={2} />
            </div>
            <h3><span className="notranslate">Crop Rotation</span></h3>
            <p>Optimize your soil health with intelligent crop rotation planning.</p>
          </div>

          <div className="card reveal" role="button" tabIndex={0} onClick={() => setShowP2PChat(true)} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setShowP2PChat(true); }} aria-label="P2P Farmer Chat: Connect with others">
            <div className="icon" aria-hidden="true">
              <MessageSquare size={32} strokeWidth={2} />
            </div>
            <h3><span className="notranslate">P2P Farmer Chat</span></h3>
            <p>Connect directly with fellow farmers for real-time advice and support.</p>
          </div>

          <div className="card reveal" role="button" tabIndex={0} onClick={() => setShowSmartCropRecommendation(true)} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setShowSmartCropRecommendation(true); }} aria-label="Smart Crop Recommendation: AI-powered suggestions">
            <div className="icon" aria-hidden="true">
              <Sprout size={32} strokeWidth={2} />
            </div>
            <h3><span className="notranslate">Smart Crop Recommendation</span></h3>
            <p>Get AI-powered crop suggestions based on your soil and climate.</p>
          </div>

          {(userData?.role === "expert" || userData?.role === "admin") && (
            <div className="card reveal expert-card" role="button" tabIndex={0} onClick={() => setShowExpertStatus(true)} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setShowExpertStatus(true); }} aria-label="Expert Reputation: View badges">
              <div className="icon" aria-hidden="true">
                <Award size={32} strokeWidth={2} />
              </div>
              <h3><span className="notranslate">Expert Reputation</span></h3>
              <p>Track your community points and earn expert badges for your contributions.</p>
              <div className="mini-badge-info">
                {currentReputation} pts · {currentReputation >= 500 ? "🥇" : currentReputation >= 200 ? "🥈" : currentReputation >= 50 ? "🥉" : "🌱"}
              </div>
            </div>
          )}

          <div className="card reveal" role="button" tabIndex={0} onClick={() => setShowGeoAlerts(true)} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setShowGeoAlerts(true); }} aria-label="Geo-Hashed Disaster Mesh: View nearby alerts">
            <div className="icon" aria-hidden="true" style={{background: 'rgba(239, 68, 68, 0.1)', color: '#ef4444'}}>
              <AlertTriangle size={32} strokeWidth={2} />
            </div>
            <h3><span className="notranslate">Disaster Mesh Alerts</span></h3>
            <p>Report and receive highly localized (5km radius) real-time disaster alerts.</p>
          </div>

          {(userData?.role === "expert" || userData?.role === "admin") && (
            <div className="card reveal bank-report-card" role="button" tabIndex={0} onClick={() => setShowBankReport(true)} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setShowBankReport(true); }} aria-label="Bank Reports: Export financial data">
              <div className="icon" aria-hidden="true">
                <Landmark size={32} strokeWidth={2} />
              </div>
              <h3><span className="notranslate">Bank Reports & Export</span></h3>
              <p>Generate professional PDF/CSV reports for bank loans and financial records.</p>
            </div>
          )}

          <div 
            className="card reveal" 
            role="button" 
            tabIndex={0} 
            onClick={() => setShowClimateSimulator(true)} 
            onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setShowClimateSimulator(true); }} 
            aria-label="Climate Risk Simulator: Scenario analysis"
          >
            <div className="icon" aria-hidden="true">
              <TrendingDown size={32} strokeWidth={2} />
            </div>
            <h3><span className="notranslate">Climate Risk Simulator</span></h3>
            <p>Evaluate crop performance under different long-term climate scenarios.</p>
          </div>

          <div 
            className="card reveal" 
            role="button" 
            tabIndex={0} 
            onClick={() => setShowRAGAdvisor(true)} 
            onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setShowRAGAdvisor(true); }} 
            aria-label="AI Research Advisor: Citation-backed answers"
          >
            <div className="icon" aria-hidden="true">
              <Book size={32} strokeWidth={2} />
            </div>
            <h3><span className="notranslate">AI Research Advisor</span></h3>
            <p>Get research-backed agricultural advice with verified citations from ICAR, FAO, and more.</p>
          </div>

          <div 
            className="card reveal" 
            style={{ border: '2px solid #10b981', background: 'rgba(16, 185, 129, 0.02)' }}
            role="button" 
            tabIndex={0} 
            onClick={() => setShowGreenPractices(true)} 
            onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setShowGreenPractices(true); }} 
            aria-label="Green Practices: Track carbon credits"
          >
            <div className="icon" aria-hidden="true" style={{ background: 'rgba(16, 185, 129, 0.1)', color: '#10b981' }}>
              <Leaf size={32} strokeWidth={2} />
            </div>
            <div style={{ position: 'absolute', top: '12px', right: '12px', background: '#10b981', color: 'white', fontSize: '10px', padding: '2px 8px', borderRadius: '10px', fontWeight: 'bold' }}>EARN</div>
            <h3><span className="notranslate">Green Practices & Carbon</span></h3>
            <p>Track eco-friendly practices, calculate carbon impact, and monetize sustainability.</p>
          </div>
        </div>

        <div
          className="weather-dashboard"
          style={{
            marginTop: "36px",
            padding: "24px",
            borderRadius: "18px",
            background: "linear-gradient(135deg, rgba(255,255,255,0.96), rgba(239,253,245,0.98))",
            boxShadow: "0 18px 45px rgba(15, 23, 42, 0.08)",
          }}
        >
          <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "12px" }}>
            <h2 style={{ margin: 0 }}>@ 🌦️ Live Weather & Advisories</h2>
            {weatherLastUpdated && (
              <LastUpdated timestamp={weatherLastUpdated} />
            )}
          </div>

          <p style={{ marginTop: "8px", color: "#0f172a" }}>
            Get real-time conditions, 7-day forecasts, and actionable crop guidance directly in the advisor view.
          </p>

          <div
            style={{
              display: "flex",
              flexWrap: "wrap",
              gap: "12px",
              marginTop: "16px",
            }}
          >
            <button
              className="action-btn"
              type="button"
              onClick={handleUseMyLocation}
            >
              Use My Location
            </button>
            <form
              onSubmit={handleLocationSearch}
              style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}
            >
              <input
                type="text"
                value={locationQuery}
                onChange={(event) => setLocationQuery(event.target.value)}
                placeholder="Search by city or district"
                aria-label="Search weather by city or district"
                style={{
                  minWidth: "240px",
                  padding: "10px 12px",
                  borderRadius: "10px",
                  border: "1px solid #cbd5f5",
                }}
              />
              <button className="action-btn secondary" type="submit">
                Search
              </button>
            </form>
            <button
              className="action-btn secondary"
              type="button"
              onClick={() => {
                if (coords) {
                  fetchWeather({
                    latitude: coords.latitude,
                    longitude: coords.longitude,
                    label: weatherLocation,
                  });
                }
              }}
            >
              Refresh
            </button>
          </div>
          
          {weatherLocation && (
            <p style={{ marginTop: "12px" }}>
              <strong>Location:</strong> {weatherLocation}
            </p>
          )}

          {weatherStatus === "loading" && (
            <p style={{ marginTop: "12px" }}>Loading weather data...</p>
          )}

          {weatherStatus === "error" && (
            <div
              style={{
                marginTop: "12px",
                padding: "12px",
                borderRadius: "10px",
                background: "#fef2f2",
                color: "#b91c1c",
              }}
            >
              {weatherError}
            </div>
          )}

          {weatherStatus === "ready" && weatherData?.current && (
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
                gap: "16px",
                marginTop: "16px",
              }}
            >
              <div
                style={{
                  padding: "16px",
                  borderRadius: "14px",
                  background: "white",
                  boxShadow: "0 12px 24px rgba(15, 23, 42, 0.08)",
                }}
              >
                <h3 style={{ marginTop: 0 }}>Now</h3>
                <p style={{ fontSize: "28px", margin: "8px 0" }}>
                  {formatTemp(weatherData.current.temp)}
                </p>
                <p style={{ margin: 0 }}>
                  {weatherData.current.weather?.[0]?.description}
                </p>
                <p style={{ margin: "8px 0 0" }}>
                  Humidity: {weatherData.current.humidity}%
                </p>
                <p style={{ margin: 0 }}>
                  Wind: {Math.round(weatherData.current.wind_speed)} m/s
                </p>
              </div>

              <div
                style={{
                  padding: "16px",
                  borderRadius: "14px",
                  background: "white",
                  boxShadow: "0 12px 24px rgba(15, 23, 42, 0.08)",
                }}
              >
                <h3 style={{ marginTop: 0 }}>Alerts</h3>
                {advisories.length === 0 ? (
                  <p style={{ margin: 0 }}>No severe alerts expected this week.</p>
                ) : (
                  advisories.map((item) => (
                    <p key={item.title} style={{ margin: "8px 0" }}>
                      <strong>{item.title}:</strong> {item.message}
                    </p>
                  ))
                )}
              </div>
            </div>
          )}

          {weatherStatus === "ready" && weatherData?.daily?.length > 0 && (
            <div
              style={{
                marginTop: "18px",
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))",
                gap: "12px",
              }}
            >
              {weatherData.daily.slice(0, 7).map((day) => (
                <div
                  key={day.dt}
                  style={{
                    background: "white",
                    borderRadius: "14px",
                    padding: "12px",
                    textAlign: "center",
                    boxShadow: "0 10px 20px rgba(15, 23, 42, 0.06)",
                  }}
                >
                  <p style={{ margin: "0 0 6px" }}>{formatDay(day.dt)}</p>
                  <p style={{ margin: "0 0 6px", fontSize: "18px" }}>
                    {formatTemp(day.temp.max)} / {formatTemp(day.temp.min)}
                  </p>
                  <p style={{ margin: 0, fontSize: "12px", color: "#475569" }}>
                    {day.weather?.[0]?.main} · {Math.round(day.pop * 100)}% rain
                  </p>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      

      {/* Modals */}
      {showWeather && (
        <div className="weather-overlay" onClick={() => setShowWeather(false)}>
          <div className="weather-popup" onClick={(e)=>e.stopPropagation()}>
            <WeatherCard onClose={() => setShowWeather(false)} />
          </div>
        </div>
      )}

      {showSoilChatbot && (
        <div className="weather-overlay" onClick={() => setShowSoilChatbot(false)}>
          <div className="chatbot-popup" onClick={(e)=>e.stopPropagation()}>
            <SoilChatbot onClose={() => setShowSoilChatbot(false)} />
          </div>
        </div>
      )}

      {showForecast && (
        <div className="weather-overlay" onClick={() => setShowForecast(false)}>
          <div className="weather-popup" onClick={(e)=>e.stopPropagation()}>
            <Forecast onClose={() => setShowForecast(false)} />
          </div>
        </div>
      )}

      {showExpertStatus && (
        <div className="weather-overlay" onClick={() => setShowExpertStatus(false)}>
          <div className="expert-status-modal" onClick={(e)=>e.stopPropagation()}>
            <div className="modal-header">
              <h2><Award className="header-icon" /> Expert Status</h2>
              <button className="close-btn" onClick={() => setShowExpertStatus(false)}><X /></button>
            </div>
            
            <div className="expert-status-content">
              <div className="reputation-hero">
                <div className="rep-main">
                  <span className="rep-value">{currentReputation}</span>
                  <span className="rep-label">Reputation Points</span>
                </div>
                <div className="badge-display">
                  <span className="badge-icon-large">
                    {currentReputation >= 500 ? "🥇" : currentReputation >= 200 ? "🥈" : currentReputation >= 50 ? "🥉" : "🌱"}
                  </span>
                  <span className="badge-title">
                    {currentReputation >= 500 ? "Master Agriculturist" : 
                     currentReputation >= 200 ? "Farming Expert" : 
                     currentReputation >= 50 ? "Active Contributor" : "Rising Star"}
                  </span>
                </div>
              </div>

              <div className="progress-section">
                <div className="progress-labels">
                  <span>Next: {nextBadge.name}</span>
                  <span>{currentReputation} / {nextBadge.threshold}</span>
                </div>
                <div className="progress-track">
                  <div className="progress-fill" style={{ width: `${progressPercent}%` }}></div>
                </div>
                <p className="progress-note">
                  Earn {nextBadge.threshold - currentReputation} more points to reach {nextBadge.icon} {nextBadge.name}
                </p>
              </div>

              <div className="earning-guide">
                <h3>How to earn points:</h3>
                <div className="guide-grid">
                  <div className="guide-card">
                    <Star className="guide-icon" />
                    <div>
                      <h4>Start Discussions</h4>
                      <p>+10 points for starting a new topic in the community.</p>
                    </div>
                  </div>
                  <div className="guide-card">
                    <Star className="guide-icon" />
                    <div>
                      <h4>Post Answers</h4>
                      <p>+5 points for every helpful comment in the community.</p>
                    </div>
                  </div>
                  <div className="guide-card">
                    <ThumbsUp className="guide-icon" />
                    <div>
                      <h4>Get Upvotes</h4>
                      <p>+10 points when other farmers upvote your contributions.</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {showBankReport && (
        <div className="weather-overlay" onClick={() => setShowBankReport(false)}>
          <div className="bank-report-modal" onClick={(e)=>e.stopPropagation()}>
            <div className="modal-header">
              <h2><Landmark className="header-icon" /> Bank Reporting & Export</h2>
              <button className="close-btn" onClick={() => setShowBankReport(false)}><X /></button>
            </div>
            
            <div className="report-preview-box">
              <div className="preview-header">
                <Sprout className="preview-logo" />
                <h3>Fasal Saathi AI Advisor</h3>
              </div>
              <div className="preview-body">
                <div className="preview-row">
                  <span>Farmer:</span>
                  <strong>{userProfile?.displayName || "Farmer"}</strong>
                </div>
                <div className="preview-row">
                  <span>Primary Crop:</span>
                  <strong>{userProfile?.cropType || "Not set"}</strong>
                </div>
                <div className="preview-row">
                  <span>Location:</span>
                  <strong>{userProfile?.address || userProfile?.location || "India"}</strong>
                </div>
                <div className="preview-divider"></div>
                <div className="preview-row">
                  <span>Reputation Points:</span>
                  <span className="risk-badge">{currentReputation} pts</span>
                </div>
              </div>
            </div>

            <div className="export-actions-grid">
              <button className="export-btn pdf" onClick={() => generateBankPDF({
                farmerName: userProfile?.displayName || "Farmer",
                cropType: userProfile?.cropType || "N/A",
                landArea: userProfile?.landArea || "N/A",
                season: userProfile?.season || "N/A",
                location: userProfile?.address || userProfile?.location || "India",
                estimatedRevenue: userProfile?.estimatedRevenue || 0,
                estimatedCost: userProfile?.estimatedCost || 0,
                netProfit: userProfile?.netProfit || 0,
                riskLevel: userProfile?.riskLevel || "Moderate",
                date: new Date().toLocaleDateString("en-IN"),
              })}>
                <div className="btn-icon">📄</div>
                <div className="btn-text">
                  <strong>Export as PDF</strong>
                  <span>Bank-friendly format</span>
                </div>
              </button>

              <button className="export-btn csv" onClick={() => generateCSV({
                farmerName: userProfile?.displayName || "Farmer",
                cropType: userProfile?.cropType || "N/A",
                landArea: userProfile?.landArea || "N/A",
                season: userProfile?.season || "N/A",
                location: userProfile?.address || userProfile?.location || "India",
                estimatedRevenue: userProfile?.estimatedRevenue || 0,
                estimatedCost: userProfile?.estimatedCost || 0,
                netProfit: userProfile?.netProfit || 0,
                riskLevel: userProfile?.riskLevel || "Moderate",
                date: new Date().toLocaleDateString("en-IN"),
              })}>
                <div className="btn-icon">📊</div>
                <div className="btn-text">
                  <strong>Export as CSV</strong>
                  <span>Spreadsheet format</span>
                </div>
              </button>
            </div>

            <div className="certified-report-section" style={{ marginTop: '2rem', borderTop: '2px dashed #e2e8f0', paddingTop: '2rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '1rem', color: '#2e7d32' }}>
                <Award size={24} />
                <h3 style={{ margin: 0 }}>Certified Digital Signature Report</h3>
              </div>
              <p style={{ fontSize: '0.9rem', color: '#64748b', marginBottom: '1.5rem' }}>
                Generate a cryptographically signed, tamper-proof report for official bank applications.
              </p>
              <BankReports userData={userProfile} />
            </div>

            <p className="report-disclaimer">
              * Reports are generated using your latest soil analysis, profit calculations, and risk index data.
            </p>
          </div>
        </div>
      )}

      {showSoilAnalysis && (
        <div className="weather-overlay" onClick={() => setShowSoilAnalysis(false)}>
          <div className="soil-analysis-popup" onClick={(e)=>e.stopPropagation()}>
            <button className="close-btn" onClick={() => setShowSoilAnalysis(false)} style={{ position: 'absolute', top: '12px', right: '12px', zIndex: 10 }}>✕</button>
            <SoilAnalysis />
          </div>
        </div>
      )}

      {showSoilGuide && (
        <div className="weather-overlay" onClick={() => setShowSoilGuide(false)}>
          <div className="soil-analysis-popup" onClick={(e) => e.stopPropagation()}>
            <button className="close-btn" onClick={() => setShowSoilGuide(false)} style={{ position: 'absolute', top: '12px', right: '12px', zIndex: 10 }}>✕</button>
            <SoilGuide />
          </div>
        </div>
      )}

      {showIrrigation && (
        <div className="weather-overlay" onClick={()=>setShowIrrigation(false)}>
          <div onClick={(e)=>e.stopPropagation()}>
            <IrrigationGuidance onClose={() => setShowIrrigation(false)} />
          </div>
        </div>
      )}

      {showYieldPopup && (
        <div className="weather-overlay" onClick={closeYieldPopup}>
          <div className="yield-popup" onClick={(e)=>e.stopPropagation()}>
            <button className="close-btn" onClick={closeYieldPopup}>✕</button>
            <h2>📊 Yield Prediction</h2>
            {yieldError && (
              <div style={{ color: '#dc2626', marginBottom: '16px', padding: '12px', background: '#fef2f2', borderRadius: '8px' }}>
                Error: {yieldError}
              </div>
            )}
            {yieldPrediction === null ? (
              <form onSubmit={fetchYield} className="yield-form">
                <div className="form-group">
                  <label>Crop</label>
                  <select value={yieldForm.Crop} onChange={(e) => updateYieldFormField("Crop", e.target.value)}>
                    <option value="Paddy">Paddy</option>
                    <option value="Cotton">Cotton</option>
                    <option value="Maize">Maize</option>
                    <option value="Bengal Gram">Bengal Gram</option>
                    <option value="Groundnut">Groundnut</option>
                    <option value="Chillies">Chillies</option>
                    <option value="Red Gram">Red Gram</option>
                  </select>
                </div>
                <div className="form-group">
                  <label>Season</label>
                  <select value={yieldForm.Season} onChange={(e) => updateYieldFormField("Season", e.target.value)}>
                    <option value="Rabi">Rabi</option>
                    <option value="Kharif">Kharif</option>
                  </select>
                </div>
                <div className="form-group">
                  <label>Covered Area (acres)</label>
                  <input type="number" value={yieldForm.CropCoveredArea} onChange={(e) => updateYieldFormField("CropCoveredArea", parseFloat(e.target.value))} />
                </div>
                <div className="form-group">
                  <label>Crop Height (cm)</label>
                  <input type="number" value={yieldForm.CHeight} onChange={(e) => updateYieldFormField("CHeight", parseInt(e.target.value))} />
                </div>
                <div className="form-group">
                  <label>Next Crop</label>
                  <select value={yieldForm.CNext} onChange={(e) => updateYieldFormField("CNext", e.target.value)}>
                    <option value="Pea">Pea</option>
                    <option value="Lentil">Lentil</option>
                    <option value="Maize">Maize</option>
                    <option value="Sorghum">Sorghum</option>
                    <option value="Wheat">Wheat</option>
                    <option value="Soybean">Soybean</option>
                    <option value="Mustard">Mustard</option>
                    <option value="Rice">Rice</option>
                    <option value="Tomato">Tomato</option>
                    <option value="Onion">Onion</option>
                  </select>
                </div>
                <div className="form-group">
                  <label>Last Crop</label>
                  <select value={yieldForm.CLast} onChange={(e) => updateYieldFormField("CLast", e.target.value)}>
                    <option value="Lentil">Lentil</option>
                    <option value="Pea">Pea</option>
                    <option value="Maize">Maize</option>
                    <option value="Sorghum">Sorghum</option>
                    <option value="Soybean">Soybean</option>
                    <option value="Wheat">Wheat</option>
                    <option value="Mustard">Mustard</option>
                    <option value="Rice">Rice</option>
                    <option value="Tomato">Tomato</option>
                    <option value="Onion">Onion</option>
                  </select>
                </div>
                <div className="form-group">
                  <label>Transplanting Method</label>
                  <select value={yieldForm.CTransp} onChange={(e) => updateYieldFormField("CTransp", e.target.value)}>
                    <option value="Transplanting">Transplanting</option>
                    <option value="Drilling">Drilling</option>
                    <option value="Broadcasting">Broadcasting</option>
                    <option value="Seed Drilling">Seed Drilling</option>
                  </select>
                </div>
                <div className="form-group">
                  <label>Irrigation Type</label>
                  <select value={yieldForm.IrriType} onChange={(e) => updateYieldFormField("IrriType", e.target.value)}>
                    <option value="Flood">Flood</option>
                    <option value="Sprinkler">Sprinkler</option>
                    <option value="Drip">Drip</option>
                    <option value="Surface">Surface</option>
                  </select>
                </div>
                <div className="form-group">
                  <label>Irrigation Source</label>
                  <select value={yieldForm.IrriSource} onChange={(e) => updateYieldFormField("IrriSource", e.target.value)}>
                    <option value="Groundwater">Groundwater</option>
                    <option value="Canal">Canal</option>
                    <option value="Rainfed">Rainfed</option>
                    <option value="Well">Well</option>
                    <option value="Tubewell">Tubewell</option>
                  </select>
                </div>
                <div className="form-group">
                  <label>Irrigation Count</label>
                  <input type="number" value={yieldForm.IrriCount} onChange={(e) => updateYieldFormField("IrriCount", parseInt(e.target.value))} />
                </div>
                <div className="form-group">
                  <label>Water Coverage (%)</label>
                  <input type="number" max="100" value={yieldForm.WaterCov} onChange={(e) => updateYieldFormField("WaterCov", parseInt(e.target.value))} />
                </div>
                <div className="form-group full-width form-actions">
                  <button type="submit" className="action-btn" disabled={yieldLoading}>
                    {yieldLoading ? "Predicting..." : "Predict Yield"}
                  </button>
                  <button type="button" className="action-btn secondary" onClick={closeYieldPopup}>Cancel</button>
                </div>
              </form>
            ) : (
              <>
                <p className="yield-result">Predicted Yield: <strong>{yieldPrediction.toFixed(2)}</strong> quintals/acre</p>
                {yieldLastUpdated && <div style={{ marginBottom: '16px', display: 'flex', justifyContent: 'center' }}><LastUpdated timestamp={yieldLastUpdated} /></div>}
                <button className="action-btn" onClick={closeYieldPopup}>Predict Another</button>
              </>
            )}
          </div>
        </div>
      )}

      {showProfitCalculator && (
        <div className="weather-overlay" onClick={()=>setShowProfitCalculator(false)}>
          <div className="weather-popup profit-popup" onClick={(e)=>e.stopPropagation()}>
            <CropProfitCalculator />
            <button className="close-btn" onClick={() => setShowProfitCalculator(false)}>Close</button>
          </div>
        </div>
      )}

      {showFertilizerPopup && (
        <div className="weather-overlay" onClick={() => setShowFertilizerPopup(false)}>
          <div className="weather-popup fertilizer-popup-shell" onClick={(e) => e.stopPropagation()}>
            <FertilizerRecommendation onClose={() => setShowFertilizerPopup(false)} />
          </div>
        </div>
      )}

      {showFarmingMap && (
        <div className="farming-map-overlay" onClick={() => setShowFarmingMap(false)}>
          <div className="farming-map-popup" onClick={(e) => e.stopPropagation()}>
            <button className="close-btn" onClick={() => setShowFarmingMap(false)}>Close</button>
            <FarmingMap />
          </div>
        </div>
      )}

      {showCropDiseaseDetection && (
        <div className="weather-overlay" onClick={() => setShowCropDiseaseDetection(false)}>
          <div className="weather-popup" onClick={(e) => e.stopPropagation()}>
            <CropDiseaseDetection onClose={() => setShowCropDiseaseDetection(false)} />
          </div>
        </div>
      )}

      {showPestManagement && (
        <div className="weather-overlay" onClick={() => setShowPestManagement(false)}>
          <div className="weather-popup" onClick={(e) => e.stopPropagation()} style={{ padding: 0, background: 'transparent', boxShadow: 'none' }}>
            <PestManagement onClose={() => setShowPestManagement(false)} />
          </div>
        </div>
      )}

      {showAgriMarketplace && (
        <div className="weather-overlay" onClick={() => setShowAgriMarketplace(false)}>
          <div className="agri-modal-wrapper" onClick={(e) => e.stopPropagation()}>
            <button className="close-btn agri-close-btn" onClick={() => setShowAgriMarketplace(false)}>✕</button>
            <AgriMarketplace onClose={() => setShowAgriMarketplace(false)} />
          </div>
        </div>
      )}

      {showAgriLMS && (
        <div className="weather-overlay" onClick={() => setShowAgriLMS(false)}>
          <div className="agri-modal-wrapper" style={{ maxWidth: '1200px' }} onClick={(e) => e.stopPropagation()}>
            <button className="close-btn agri-close-btn" onClick={() => setShowAgriLMS(false)}>✕</button>
            <AgriLMS />
          </div>
        </div>
      )}

      {showQRTraceability && (
        <div className="weather-overlay" onClick={() => setShowQRTraceability(false)}>
          <div className="agri-modal-wrapper" style={{ maxWidth: '1200px' }} onClick={(e) => e.stopPropagation()}>
            <button className="close-btn agri-close-btn" onClick={() => setShowQRTraceability(false)}>✕</button>
            <QRTraceability />
          </div>
        </div>
      )}

      {showFarmPlanner3D && (
        <div className="weather-overlay" onClick={() => setShowFarmPlanner3D(false)}>
          <div className="agri-modal-wrapper" style={{ maxWidth: '1200px' }} onClick={(e) => e.stopPropagation()}>
            <button className="close-btn agri-close-btn" onClick={() => setShowFarmPlanner3D(false)}>✕</button>
            <FarmPlanner3D />
          </div>
        </div>
      )}

      {showComingSoon && (
        <div className="weather-overlay" onClick={()=>setShowComingSoon(false)}>
          <div className="weather-popup coming-soon" onClick={(e)=>e.stopPropagation()}>
            <h2>🚧 Coming Soon</h2>
            <p>This feature is under development. Stay tuned!</p>
            <button className="close-btn" onClick={() => setShowComingSoon(false)}>Close</button>
          </div>
        </div>
      )}

      {showFarmDiary && (
        <div className="weather-overlay" onClick={() => setShowFarmDiary(false)}>
          <div className="agri-modal-wrapper" style={{ maxWidth: '1200px' }} onClick={(e) => e.stopPropagation()}>
            <button className="close-btn agri-close-btn" onClick={() => setShowFarmDiary(false)}>✕</button>
            <FarmDiary onClose={() => setShowFarmDiary(false)} />
          </div>
        </div>
      )}

      {showCropRotation && (
        <div className="weather-overlay" onClick={() => setShowCropRotation(false)}>
          <div className="agri-modal-wrapper" style={{ maxWidth: '1200px' }} onClick={(e) => e.stopPropagation()}>
            <button className="close-btn agri-close-btn" onClick={() => setShowCropRotation(false)}>✕</button>
            <CropRotation />
          </div>
        </div>
      )}

      {showP2PChat && (
        <div className="weather-overlay" onClick={() => setShowP2PChat(false)}>
          <div className="weather-popup" onClick={(e) => e.stopPropagation()}>
            <P2PChat 
              recipient={{ userId: "advisor", userName: "AI Farming Advisor" }} 
              onClose={() => setShowP2PChat(false)} 
            />
          </div>
        </div>
      )}

      {showGeoAlerts && (
        <div className="weather-overlay" onClick={() => setShowGeoAlerts(false)}>
          <GeoAlertMesh onClose={() => setShowGeoAlerts(false)} />
        </div>
      )}

      {showSmartCropRecommendation && (
        <div className="weather-overlay" onClick={() => setShowSmartCropRecommendation(false)}>
          <div className="weather-popup" onClick={(e) => e.stopPropagation()}>
            <SmartCropRecommendation />
            <button
              className="close-btn"
              onClick={() => setShowSmartCropRecommendation(false)}
            >
              Close
            </button>
          </div>
        </div>
      )}

      {showSeedVerifier && (
        <div className="weather-overlay" onClick={() => setShowSeedVerifier(false)}>
          <div className="weather-popup" style={{ width: '90%', maxWidth: '450px', padding: 0, overflowY: 'auto', maxHeight: '90vh' }} onClick={(e) => e.stopPropagation()}>
            <SeedVerifier onClose={() => setShowSeedVerifier(false)} />
          </div>
        </div>
      )}

      <br />
      <br />
      <ClimateSimulator 
        isOpen={showClimateSimulator} 
        onClose={() => setShowClimateSimulator(false)} 
        userData={userData}
      />
      <RAGAdvisor
        isOpen={showRAGAdvisor}
        onClose={() => setShowRAGAdvisor(false)}
      />

      {showGreenPractices && (
        <div className="weather-overlay" onClick={() => setShowGreenPractices(false)}>
          <div onClick={(e) => e.stopPropagation()} style={{ width: '100%', display: 'flex', justifyContent: 'center' }}>
            <GreenPractices 
              userProfile={userProfile} 
              onClose={() => setShowGreenPractices(false)} 
            />
          </div>
        </div>
      )}
    </section>
  );
}
