import React, { useEffect, useMemo, useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import "./Advisor.css";
// Components - Static imports (lazy loading removed for faster feature access)
import WeatherCard from "./weather/WeatherCard";
import Forecast from "./Forecast";
import SoilChatbot from "./SoilChatbot";
import SoilAnalysis from "./SoilAnalysis";
import SoilGuide from "./SoilGuide";
import IrrigationGuidance from "./IrrigationGuidance";
import CropProfitCalculator from "./CropProfitCalculator";
import FarmingMap from "./FarmingMap";
import FertilizerRecommendation from "./FertilizerRecommendation";
import AgriMarketplace from "./AgriMarketplace";
import AgriLMS from "./AgriLMS";
import BankReports from "./BankReports";
import QRTraceability from "./QRTraceability";
import FarmPlanner3D from "./FarmPlanner3D";
import FarmDiary from "./FarmDiary";
import CropDiseaseDetection from "./CropDiseaseDetection";
import PestDetection from "./PestDetection";
import PestManagement from "./PestManagement";
import SeedVerifier from "./SeedVerifier";
import ClimateSimulator from "./ClimateSimulator";
import RAGAdvisor from "./RAGAdvisor";
import GreenPractices from "./GreenPractices";
import YieldPredictorForm from "./YieldPredictorForm";
import CropRotation from "./CropRotation";
import P2PChat from "./P2PChat";
import GeoAlertMesh from "./GeoAlertMesh";
import SmartCropRecommendation from "./SmartCropRecommendation";
import CropRecommendationAdvisor from "./CropRecommendationAdvisor";
import PersonalizedAdvisory from "./PersonalizedAdvisory";
import YieldHistory from "./YieldHistory";
import CropRecommendationAdvisor from "./CropRecommendationAdvisor";
import EquipmentManagement from "./EquipmentManagement";
import LastUpdated from "./LastUpdated";
import { Leaf } from "lucide-react";
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
  TrendingDown,
  Bug,
  BarChart3,
  Rocket,
  Trophy,
  Medal,
   Gem,
   FileText,
   Construction,
   CloudRain,
   Settings,
 } from "lucide-react";
import { FaSync } from "react-icons/fa";
import { useAdvisorStore } from "./stores/advisorStore";

import { useYieldPrediction } from "./hooks/useYieldPrediction";
import { auth, db } from "./lib/firebase";
import { generateBankPDF, generateCSV } from "./utils/exportService";
import { doc, onSnapshot } from "firebase/firestore";
import {
  WEATHER_SNAPSHOT_EVENT,
  getStoredWeatherSnapshot,
  fetchWeatherByLocation,
  getCurrentPosition,
  fetchWeatherByIP,
  searchLocationByName,
} from "./weather/weatherService";

export default function Advisor({ userData }) {
  const navigate = useNavigate();
  
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
    showCropRecommendationAdvisor,
    setShowCropRecommendationAdvisor,
    showEquipmentManagement,
    setShowEquipmentManagement,
} = useAdvisorStore();



  const {
    showYieldPopup,
    setShowYieldPopup,
    closeYieldPopup,
  } = useYieldPrediction();

  const [weatherStatus, setWeatherStatus] = useState("idle");
  const [weatherError, setWeatherError] = useState("");
  const [userProfile, setUserProfile] = useState(null);
  const [weatherSnapshot, setWeatherSnapshot] = useState(() => getStoredWeatherSnapshot());
  const [showYieldHistory, setShowYieldHistory] = useState(false);
  const [locationQuery, setLocationQuery] = useState("");

  // ── Shared weather snapshot integration ──────────────────────────────────
  // Subscribe to the global WEATHER_SNAPSHOT_EVENT so any fetch by
  // WeatherAlertBar or WeatherQuickWidget is immediately reflected here —
  // no duplicate API call needed.
  useEffect(() => {
    const handleSnapshot = (event) => {
      const snap = event.detail;
      if (snap?.location) {
        setWeatherSnapshot(snap);
        setWeatherStatus("ready");
        setWeatherError("");
      }
    };
    window.addEventListener(WEATHER_SNAPSHOT_EVENT, handleSnapshot);
    return () => window.removeEventListener(WEATHER_SNAPSHOT_EVENT, handleSnapshot);
  }, []);

  // On mount: if a valid cached snapshot already exists (written by
  // WeatherAlertBar on the Home page), use it immediately — no fetch needed.
  useEffect(() => {
    const cached = getStoredWeatherSnapshot();
    if (cached?.location) {
      setWeatherSnapshot(cached);
      setWeatherStatus("ready");
    }
  }, []);

  // Derive advisories from the open-meteo snapshot alerts array.
  // weatherService.js already computes these via deriveAlerts() — we just
  // map them to the shape the existing JSX expects.
  const advisories = useMemo(() => {
    if (!weatherSnapshot?.alerts?.length) return [];
    return weatherSnapshot.alerts
      .filter(a => a.type !== "stable")
      .map(a => ({ type: a.type, title: a.title, message: a.message }));
  }, [weatherSnapshot]);

  // Fetch weather via the shared service (writes to the shared cache and
  // broadcasts WEATHER_SNAPSHOT_EVENT so all components stay in sync).
  const fetchWeather = async ({ latitude, longitude, label }) => {
    setWeatherStatus("loading");
    setWeatherError("");
    try {
      const snap = await fetchWeatherByLocation({
        latitude, longitude,
        city: label || "Your area",
        name: label || "Your area",
        source: "manual",
      });
      setWeatherSnapshot(snap);
      setWeatherStatus("ready");
    } catch (err) {
      setWeatherStatus("error");
      setWeatherError(err?.message || "Failed to load weather data.");
    }
  };

  const handleUseMyLocation = async () => {
    setWeatherStatus("loading");
    setWeatherError("");
    try {
      const location = await getCurrentPosition();
      const snap = await fetchWeatherByLocation(location);
      setWeatherSnapshot(snap);
      setWeatherStatus("ready");
    } catch {
      // GPS failed — fall back to IP-based location
      try {
        const snap = await fetchWeatherByIP();
        setWeatherSnapshot(snap);
        setWeatherStatus("ready");
      } catch (err) {
        setWeatherStatus("error");
        setWeatherError(err?.message || "Unable to access your location. Please search manually.");
      }
    }
  };

  const handleLocationSearch = async (event) => {
    event.preventDefault();
    if (!locationQuery.trim()) return;
    setWeatherStatus("loading");
    setWeatherError("");
    try {
      const location = await searchLocationByName(locationQuery.trim());
      const snap = await fetchWeatherByLocation(location);
      setWeatherSnapshot(snap);
      setWeatherStatus("ready");
    } catch (err) {
      setWeatherStatus("error");
      setWeatherError(err?.message || "Location not found. Try a nearby city or district.");
    }
  };

  // Helpers to format open-meteo data for the weather dashboard JSX.
  // open-meteo daily arrays are indexed by day (0 = today).
  const formatTemp = (value) => `${Math.round(value ?? 0)}°C`;
  const formatDay = (isoDate) =>
    new Date(isoDate).toLocaleDateString(undefined, {
      weekday: "short", day: "numeric", month: "short",
    });

  // Build a normalised daily array from the open-meteo snapshot so the
  // existing JSX can iterate it without changes to the template.
  const dailyForecast = useMemo(() => {
    const d = weatherSnapshot?.daily;
    if (!d?.time?.length) return [];
    return d.time.slice(0, 7).map((date, i) => ({
      date,
      maxTemp: d.temperature_2m_max?.[i] ?? null,
      minTemp: d.temperature_2m_min?.[i] ?? null,
      rain:    d.precipitation_sum?.[i] ?? 0,
      code:    d.weather_code?.[i] ?? 0,
    }));
  }, [weatherSnapshot]);

  const weatherLocation = weatherSnapshot?.location?.name || weatherSnapshot?.location?.city || "";
  const weatherLastUpdated = weatherSnapshot?.fetchedAt ? new Date(weatherSnapshot.fetchedAt).getTime() : null;

   useEffect(() => {
     // Check if Firebase is configured
     if (!auth || !db) {
       console.warn("Firebase not configured - skipping user profile subscription");
       return;
     }
     
     // Priority: auth.currentUser, then fallback to localStorage
     const uid = auth.currentUser?.uid || localStorage.getItem("userId");
     
     if (uid) {
       const unsubscribe = onSnapshot(doc(db, "users", uid), (doc) => {
         if (doc.exists()) {
           setUserProfile(doc.data());
         }
       });
       return () => unsubscribe();
     }
   }, []); // Run once on mount — rAF loop manages its own lifecycle internally.

  /**
   * Architecture
   * ------------
   * The animation runs entirely in local component state using
   * requestAnimationFrame (rAF) rather than setInterval + global store writes.
   *
   * Why this is better than the previous setInterval approach:
   *
   * 1. No global store thrashing — the previous implementation wrote to the
   *    Zustand store on every tick (every 50 ms = ~200 writes to reach the
   *    targets), causing the entire component tree subscribed to those values
   *    to re-render on each write.  Local state confines re-renders to this
   *    component only.
   *
   * 2. No stale-closure / rapid create-destroy cycle — the previous fix put
   *    [farmers, crops, languages] in the dependency array, which caused the
   *    effect to tear down and recreate the interval on every store update,
   *    resulting in a rapid create/destroy cycle that was worse than the
   *    original bug.
   *
   * 3. rAF is frame-rate aware — it fires at most once per display frame
   *    (~16 ms at 60 fps) and is automatically paused by the browser when
   *    the tab is hidden, saving CPU on background tabs.
   *
   * 4. Single global store write — the store is updated exactly once when
   *    all counters reach their targets, so the final values are persisted
   *    for when the user navigates back to this page.
   *
   * 5. Clean unmount — cancelling the rAF handle in the cleanup function
   *    guarantees the animation stops immediately when the component unmounts,
   *    with no background updates.
   */
  const TARGETS = { farmers: 50000, crops: 120, languages: 12 };
  const STEPS   = { farmers: 500,   crops: 2,   languages: 1  };

  // Local display counters — drive the rendered numbers without touching
  // the global store on every frame.
  const [displayFarmers,   setDisplayFarmers]   = useState(farmers);
  const [displayCrops,     setDisplayCrops]     = useState(crops);
  const [displayLanguages, setDisplayLanguages] = useState(languages);

  // Stable refs so the rAF callback always reads the latest values without
  // being listed as effect dependencies (avoids the stale-closure trap).
  const displayRef = useRef({ farmers, crops, languages });
  const rafRef     = useRef(null);

  useEffect(() => {
    // If the store already holds the final values (e.g. user navigated back),
    // sync local display state and skip the animation entirely.
    if (
      farmers  >= TARGETS.farmers  &&
      crops    >= TARGETS.crops    &&
      languages >= TARGETS.languages
    ) {
      setDisplayFarmers(TARGETS.farmers);
      setDisplayCrops(TARGETS.crops);
      setDisplayLanguages(TARGETS.languages);
      return;
    }

    // Reset local counters to 0 so the animation always plays from the start
    // when the component mounts fresh.
    displayRef.current = { farmers: 0, crops: 0, languages: 0 };
    setDisplayFarmers(0);
    setDisplayCrops(0);
    setDisplayLanguages(0);

    const tick = () => {
      const cur = displayRef.current;
      const nextFarmers   = Math.min(cur.farmers   + STEPS.farmers,   TARGETS.farmers);
      const nextCrops     = Math.min(cur.crops     + STEPS.crops,     TARGETS.crops);
      const nextLanguages = Math.min(cur.languages + STEPS.languages, TARGETS.languages);

      displayRef.current = {
        farmers:   nextFarmers,
        crops:     nextCrops,
        languages: nextLanguages,
      };

      setDisplayFarmers(nextFarmers);
      setDisplayCrops(nextCrops);
      setDisplayLanguages(nextLanguages);

      const done =
        nextFarmers   >= TARGETS.farmers  &&
        nextCrops     >= TARGETS.crops    &&
        nextLanguages >= TARGETS.languages;

      if (done) {
        // Write final values to the global store exactly once so they are
        // persisted if the user navigates away and returns.
        setFarmers(TARGETS.farmers);
        setCrops(TARGETS.crops);
        setLanguages(TARGETS.languages);
      } else {
        rafRef.current = requestAnimationFrame(tick);
      }
    };

    rafRef.current = requestAnimationFrame(tick);

    // Cancel the animation immediately on unmount — no background updates.
    return () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Run once on mount — rAF loop manages its own lifecycle internally.

  const getNextBadgeThreshold = (points) => {
    if (points < 50) return { threshold: 50, name: "Active Contributor", icon: <Medal size={16} style={{ color: '#cd7f32' }} /> };
    if (points < 200) return { threshold: 200, name: "Farming Expert", icon: <Medal size={16} style={{ color: '#c0c0c0' }} /> };
    if (points < 500) return { threshold: 500, name: "Master Agriculturist", icon: <Trophy size={16} style={{ color: '#ffd700' }} /> };
    return { threshold: points, name: "Maximum Rank", icon: <Gem size={16} style={{ color: '#4facfe' }} /> };
  };

  const currentReputation = userProfile?.reputation || 0;
  const nextBadge = getNextBadgeThreshold(currentReputation);
  const progressPercent = Math.min((currentReputation / nextBadge.threshold) * 100, 100);

  return (
    <section className="advisor">
      <div className="floating-icons">
        <span><Sprout /></span>
        <span><Sun /></span>
        <span><Droplets /></span>
        <span><IndianRupee /></span>
      </div>

      <div className="advisor-hero">
        <button 
          className="back-btn" 
          onClick={() => navigate(-1)}
          aria-label="Go back"
        >
          <X size={20} />
        </button>
        <h1 className="fade-in"><Sprout className="inline-icon" /> <span className="notranslate">AI-Powered Agricultural Advisor</span></h1>
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
          <Rocket className="inline-icon" /> <span className="notranslate" aria-hidden="true">Get Started</span>
        </button>
      </div>

      <div className="advisor-stats">
        <div className="stat">
          <h2><span className="stat-number">{displayFarmers.toLocaleString()}</span>{displayFarmers >= 50000 && <span className="stat-plus">+</span>}</h2>
          <p><span className="notranslate">Farmers Connected</span></p>
        </div>
        <div className="stat">
          <h2><span className="stat-number">{displayCrops}</span>{displayCrops >= 120 && <span className="stat-plus">+</span>}</h2>
          <p><span className="notranslate">Crops Analyzed</span></p>
        </div>
        <div className="stat">
          <h2><span className="stat-number">{displayLanguages}</span>{displayLanguages >= 12 && <span className="stat-plus">+</span>}</h2>
          <p><span className="notranslate">Languages Available</span></p>
        </div>
      </div>

<PersonalizedAdvisory
         userProfile={userProfile}
         weatherData={weatherSnapshot}
       />

      <br />
      <br />

      <div className="advisor-highlights">
        <h2 className="slide-in"><Layers className="inline-icon" /> <span className="notranslate">Features</span></h2>
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

          <div className="card reveal" role="button" tabIndex={0} onClick={() => navigate("/pest-detection")} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') navigate("/pest-detection"); }} aria-label="Pest Detection: Identify pests and get treatment">
            <div className="icon" aria-hidden="true">
              <Bug size={32} strokeWidth={2} />
            </div>
            <h3><span className="notranslate">Pest Detection</span></h3>
            <p>
              AI-powered pest identification with real-time alerts and treatment recommendations.
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
            <div className="icon" aria-hidden="true"><Sprout size={32} /></div>
            <h3><span className="notranslate">Crop Disease Detection</span></h3>
            <p>Upload plant images to detect diseases and get remedies.</p>
          </div>

          <div className="card reveal" role="button" tabIndex={0} onClick={() => setShowFertilizerPopup(true)} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setShowFertilizerPopup(true); }} aria-label="Fertilizer Recommendations: Plan your nutrition">
            <div className="icon" aria-hidden="true"><FlaskConical size={32} /></div>
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
            <div className="icon" aria-hidden="true"><Bug size={32} /></div>
            <h3><span className="notranslate">Pest Management</span></h3>
            <p>Early warnings & organic pest control tips.</p>
          </div>

          <div className="card reveal" role="button" tabIndex={0} onClick={() => setShowYieldPopup(true)} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setShowYieldPopup(true); }} aria-label="Yield Prediction: AI-based forecast">
            <div className="icon" aria-hidden="true"><BarChart3 size={32} /></div>
            <h3><span className="notranslate">Yield Prediction</span></h3>
            <p>AI predicts crop yield based on soil &amp; weather data.</p>
            <button
              className="card-link-btn"
              onClick={(e) => { e.stopPropagation(); navigate("/yield-predictor"); }}
              aria-label="Open Yield Predictor as full page"
            >
              Open full page →
            </button>
          </div>

          <div className="card reveal" role="button" tabIndex={0} onClick={() => setShowYieldHistory(true)} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setShowYieldHistory(true); }} aria-label="Yield History: Track past predictions and accuracy">
            <div className="icon" aria-hidden="true"><BarChart3 size={32} /></div>
            <h3><span className="notranslate">Yield History</span></h3>
            <p>Track past yield predictions, record actual harvests, and monitor model accuracy.</p>
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
              <div className="icon" aria-hidden="true"><ShoppingCart size={32} /></div>
              <h3><span className="notranslate">Agri Marketplace</span></h3>
              <p>Rent or list farm equipment locally. Save costs and earn extra.</p>
            </div>
          )}

          {(userData?.role === "farmer" || userData?.role === "admin") && (
            <div className="card reveal" role="button" tabIndex={0} onClick={() => navigate("/equipment-management")} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') navigate("/equipment-management"); }} aria-label="Equipment Management: Monitor and maintain farm equipment">
              <div className="icon" aria-hidden="true"><Settings size={32} /></div>
              <h3><span className="notranslate">Equipment Management</span></h3>
              <p>Real-time monitoring and predictive maintenance for all farm equipment.</p>
            </div>
          )}

          <div className="card reveal" role="button" tabIndex={0} onClick={() => setShowAgriLMS(true)} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setShowAgriLMS(true); }} aria-label="Agri-LMS Academy: Online courses">
            <div className="icon" aria-hidden="true"><Award size={32} /></div>
            <h3><span className="notranslate">Agri-LMS Academy</span></h3>
            <p>Access video tutorials on modern farming and earn completion certificates.</p>
          </div>

           <div className="card reveal" role="button" tabIndex={0} onClick={() => setShowQRTraceability(true)} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setShowQRTraceability(true); }} aria-label="QR-Farm Traceability: Trace your produce">
             <div className="icon" aria-hidden="true"><QrCode size={32} /></div>
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
            <div className="icon" aria-hidden="true"><Map size={32} /></div>
            <h3><span className="notranslate">3D Farm Planner</span></h3>
            <p>Design your farm layout in interactive 3D. Optimize land usage and irrigation.</p>
          </div>

          <div className="card reveal" role="button" tabIndex={0} onClick={() => navigate("/farm-finance")} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') navigate("/farm-finance"); }} aria-label="Farm Finance: Seasonal P&L tracking">
            <div className="icon" aria-hidden="true">
              <IndianRupee size={32} strokeWidth={2} />
            </div>
            <h3><span className="notranslate">Farm Finance</span></h3>
            <p>Track seasonal income, expenses, and overall profitability with visual analytics.</p>
          </div>

          <div className="card reveal" role="button" tabIndex={0} onClick={() => setShowProfitCalculator(true)} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setShowProfitCalculator(true); }} aria-label="Profit Calculator: ROI analysis">
            <div className="icon" aria-hidden="true"><IndianRupee size={32} /></div>
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

          <div className="card reveal" role="button" tabIndex={0} onClick={() => setShowCropRecommendationAdvisor(true)} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setShowCropRecommendationAdvisor(true); }} aria-label="Crop Advisor: Detailed soil analysis and recommendations">
            <div className="icon" aria-hidden="true" style={{background: 'rgba(16, 185, 129, 0.1)', color: '#10b981'}}>
              <FlaskConical size={32} strokeWidth={2} />
            </div>
            <h3><span className="notranslate">Crop Advisor (Soil Analysis)</span></h3>
            <p>Enter soil parameters for detailed crop compatibility analysis and recommendations.</p>
          </div>

          {(userData?.role === "expert" || userData?.role === "admin") && (
            <div className="card reveal expert-card" role="button" tabIndex={0} onClick={() => setShowExpertStatus(true)} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setShowExpertStatus(true); }} aria-label="Expert Reputation: View badges">
              <div className="icon" aria-hidden="true">
                <Award size={32} strokeWidth={2} />
              </div>
              <h3><span className="notranslate">Expert Reputation</span></h3>
              <p>Track your community points and earn expert badges for your contributions.</p>
              <div className="mini-badge-info">
                {currentReputation} pts · {currentReputation >= 500 ? <Trophy size={14} style={{ color: '#ffd700' }} /> : currentReputation >= 200 ? <Medal size={14} style={{ color: '#c0c0c0' }} /> : currentReputation >= 50 ? <Medal size={14} style={{ color: '#cd7f32' }} /> : <Sprout size={14} />}
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

<div className="card reveal bank-report-card" role="button" tabIndex={0} onClick={() => setShowBankReport(true)} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setShowBankReport(true); }} aria-label="Bank Reports: Export financial data">
  <div className="icon" aria-hidden="true">
    <Landmark size={32} strokeWidth={2} />
  </div>
  <h3><span className="notranslate">Bank Reports & Export</span></h3>
  <p>Generate professional PDF/CSV reports for bank loans and financial records.</p>
</div>

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
<div className="card reveal" role="button" tabIndex={0} onClick={() => navigate("/farming-news")} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') navigate("/farming-news"); }} aria-label="Farming News: Latest agricultural updates">
              <div className="icon" aria-hidden="true">
                <Book size={32} strokeWidth={2} />
              </div>
              <h3><span className="notranslate">Farming News</span></h3>
              <p>
                Stay updated with the latest agricultural news, weather alerts, and policy changes.
              </p>
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
            <h2 style={{ margin: 0 }}><CloudRain className="inline-icon" /> Live Weather & Advisories</h2>
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
                if (weatherSnapshot?.location) {
                  fetchWeather({
                    latitude: weatherSnapshot.location.latitude,
                    longitude: weatherSnapshot.location.longitude,
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

          {weatherStatus === "ready" && weatherSnapshot?.current && (
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
                  {formatTemp(weatherSnapshot.current.temperature_2m)}
                </p>
                <p style={{ margin: 0 }}>
                  {weatherSnapshot.summary || "Current conditions"}
                </p>
                <p style={{ margin: "8px 0 0" }}>
                  Humidity: {weatherSnapshot.current.relative_humidity_2m}%
                </p>
                <p style={{ margin: 0 }}>
                  Wind: {Math.round(weatherSnapshot.current.wind_speed_10m)} m/s
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

          {weatherStatus === "ready" && dailyForecast.length > 0 && (
            <div
              style={{
                marginTop: "18px",
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))",
                gap: "12px",
              }}
            >
              {dailyForecast.map((day) => (
                <div
                  key={day.date}
                  style={{
                    background: "white",
                    borderRadius: "14px",
                    padding: "12px",
                    textAlign: "center",
                    boxShadow: "0 10px 20px rgba(15, 23, 42, 0.06)",
                  }}
                >
                  <p style={{ margin: "0 0 6px" }}>{formatDay(day.date)}</p>
                  <p style={{ margin: "0 0 6px", fontSize: "18px" }}>
                    {formatTemp(day.maxTemp)} / {formatTemp(day.minTemp)}
                  </p>
                  <p style={{ margin: 0, fontSize: "12px", color: "#475569" }}>
                    Rain: {Math.round(day.rain)} mm
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
                    {currentReputation >= 500 ? <Trophy size={24} style={{ color: '#ffd700' }} /> : currentReputation >= 200 ? <Medal size={24} style={{ color: '#c0c0c0' }} /> : currentReputation >= 50 ? <Medal size={24} style={{ color: '#cd7f32' }} /> : <Sprout size={24} />}
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
                  Earn {nextBadge.threshold - currentReputation} more points to reach <span className="inline-icon-wrap">{nextBadge.icon}</span> {nextBadge.name}
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
                <div className="btn-icon"><FileText size={20} /></div>
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
                <div className="btn-icon"><BarChart3 size={20} /></div>
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
            <button className="close-btn" onClick={() => setShowSoilAnalysis(false)} style={{ position: 'absolute', top: '12px', right: '12px', zIndex: 10 }}><X /></button>
            <SoilAnalysis />
          </div>
        </div>
      )}

      {showSoilGuide && (
        <div className="weather-overlay" onClick={() => setShowSoilGuide(false)}>
          <div className="soil-analysis-popup" onClick={(e) => e.stopPropagation()}>
            <button className="close-btn" onClick={() => setShowSoilGuide(false)} style={{ position: 'absolute', top: '12px', right: '12px', zIndex: 10 }}><X /></button>
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
          <div className="yield-popup" onClick={(e) => e.stopPropagation()}>
            <button className="close-btn" onClick={closeYieldPopup} aria-label="Close yield prediction">
              <X />
            </button>
            <YieldPredictorForm onClose={closeYieldPopup} />
          </div>
        </div>
      )}

      {showYieldHistory && (
        <div className="weather-overlay" onClick={() => setShowYieldHistory(false)}>
          <div className="weather-popup" style={{ maxWidth: "900px", width: "95vw", maxHeight: "90vh", overflowY: "auto" }} onClick={(e) => e.stopPropagation()}>
            <button className="close-btn" onClick={() => setShowYieldHistory(false)} aria-label="Close yield history">
              <X />
            </button>
            <YieldHistory />
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
            <button className="close-btn agri-close-btn" onClick={() => setShowAgriMarketplace(false)}><X /></button>
            <AgriMarketplace onClose={() => setShowAgriMarketplace(false)} />
          </div>
        </div>
      )}

      {showAgriLMS && (
        <div className="weather-overlay" onClick={() => setShowAgriLMS(false)}>
          <div className="agri-modal-wrapper" style={{ maxWidth: '1200px' }} onClick={(e) => e.stopPropagation()}>
            <button className="close-btn agri-close-btn" onClick={() => setShowAgriLMS(false)}><X /></button>
            <AgriLMS />
          </div>
        </div>
      )}

      {showQRTraceability && (
        <div className="weather-overlay" onClick={() => setShowQRTraceability(false)}>
          <div className="agri-modal-wrapper" style={{ maxWidth: '1200px' }} onClick={(e) => e.stopPropagation()}>
            <button className="close-btn agri-close-btn" onClick={() => setShowQRTraceability(false)}><X /></button>
            <QRTraceability />
          </div>
        </div>
      )}

      {showFarmPlanner3D && (
        <div className="weather-overlay" onClick={() => setShowFarmPlanner3D(false)}>
          <div className="agri-modal-wrapper" style={{ maxWidth: '1200px' }} onClick={(e) => e.stopPropagation()}>
            <button className="close-btn agri-close-btn" onClick={() => setShowFarmPlanner3D(false)}><X /></button>
            <FarmPlanner3D />
          </div>
        </div>
      )}

      {showComingSoon && (
        <div className="weather-overlay" onClick={()=>setShowComingSoon(false)}>
          <div className="weather-popup coming-soon" onClick={(e)=>e.stopPropagation()}>
            <h2><Construction className="inline-icon" /> Coming Soon</h2>
            <p>This feature is under development. Stay tuned!</p>
            <button className="close-btn" onClick={() => setShowComingSoon(false)}>Close</button>
          </div>
        </div>
      )}

      {showFarmDiary && (
        <div className="weather-overlay" onClick={() => setShowFarmDiary(false)}>
          <div className="agri-modal-wrapper" style={{ maxWidth: '1200px' }} onClick={(e) => e.stopPropagation()}>
            <button className="close-btn agri-close-btn" onClick={() => setShowFarmDiary(false)}><X /></button>
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

      {showCropRecommendationAdvisor && (
        <div className="weather-overlay" onClick={() => setShowCropRecommendationAdvisor(false)}>
          <div className="weather-popup crop-advisor-popup" onClick={(e) => e.stopPropagation()}>
            <CropRecommendationAdvisor onClose={() => setShowCropRecommendationAdvisor(false)} />
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
