import React, { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import "./Advisor.css";
import WeatherCard from "./weather/WeatherCard";
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
import QRTraceability from "./QRTraceability";
import FarmPlanner3D from "./FarmPlanner3D";
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
} from "lucide-react";
import FarmDiary from "./FarmDiary";
import { useAdvisorStore } from "./stores/advisorStore";
import { useYieldPrediction } from "./hooks/useYieldPrediction";
import CropDiseaseDetection from "./CropDiseaseDetection";
import PestManagement from "./PestManagement";

export default function Advisor() {
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
  } = useAdvisorStore();

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
        >
          🚀 <span className="notranslate">Get Started</span>
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
             onClick={() => navigate("/crop-planner")}
           >
             <div className="icon">
               <Calendar size={32} strokeWidth={2} />
             </div>
             <h3><span className="notranslate">Seasonal Crop Planner</span></h3>
             <p>Plan your crops throughout the year with seasonal recommendations and crop rotation cycles.</p>
           </div>
           <div
             className="card reveal"
             style={{ cursor: "pointer" }}
             onClick={() => setShowWeather(true)}
           >
             <div className="icon">
               <Sun size={32} strokeWidth={2} />
             </div>
             <h3><span className="notranslate">Weather Forecasts</span></h3>
             <p>
               Accurate daily & weekly weather insights for farming decisions.
             </p>
           </div>

            <div className="card reveal" onClick={() => navigate("/community")}>
              <div className="icon">
                <MessageSquare size={32} strokeWidth={2} />
              </div>
              <h3><span className="notranslate">Farmer Community</span></h3>
              <p>
                Connect, share tips, and learn from other farmers in your region.
              </p>
            </div>
             <div className="card reveal" onClick={() => navigate("/helpline")}>
               <div className="icon">
                 <Landmark size={32} strokeWidth={2} />
               </div>
               <h3><span className="notranslate">Emergency Helpline</span></h3>
               <p>
                 Quick access to emergency farming support and expert advice.
               </p>
             </div>
             <div className="card reveal" onClick={() => navigate("/blog")}>
               <div className="icon">
                 <Book size={32} strokeWidth={2} />
               </div>
               <h3><span className="notranslate">Knowledge Blog</span></h3>
               <p>
                 Read articles on crop management, weather, and farming best practices.
               </p>
             </div>
            <div className="card reveal" onClick={() => navigate("/disease-awareness")}>
              <div className="icon">
                <Info size={32} strokeWidth={2} />
              </div>
              <h3><span className="notranslate">Crop Disease Awareness</span></h3>
              <p>
                Learn about crop diseases and remedies for better farming.
              </p>
            </div>
            <div className="card reveal" onClick={() => setShowIrrigation(true)}>
            <div className="icon">
              <Droplets size={32} strokeWidth={2} />
            </div>
            <h3><span className="notranslate">Irrigation Guidance</span></h3>
            <p>
              Water-saving tips and irrigation schedules tailored to your crops.
            </p>
          </div>

          <div className="card reveal" onClick={() => navigate("/market-prices")}>
            <div className="icon">
              <IndianRupee size={32} strokeWidth={2} />
            </div>
            <h3><span className="notranslate">Market Price Guidance</span></h3>
            <p>
              Market trends and price alerts to help you sell at the best time.
            </p>
          </div>

          <div className="card reveal" onClick={() => setShowSoilChatbot(true)}>
            <div className="icon">
              <Sprout size={32} strokeWidth={2} />
            </div>
            <h3><span className="notranslate">Soil Health</span></h3>
            <p>Get soil analysis & recommendations via AI chatbot.</p>
          </div>

          <div
            className="card reveal"
            style={{ cursor: "pointer" }}
            onClick={() => setShowSoilAnalysis(true)}
          >
            <div className="icon">
              <FlaskConical size={32} strokeWidth={2} />
            </div>
            <h3><span className="notranslate">Soil Analysis</span></h3>
            <p>Analyze NPK nutrients and get personalized crop & fertilizer recommendations.</p>
          </div>

          <div
            className="card reveal"
            style={{ cursor: "pointer" }}
            onClick={() => setShowSoilGuide(true)}
          >
            <div className="icon">
              <Layers size={32} strokeWidth={2} />
            </div>
            <h3>Soil Type Guide</h3>
            <p>Explore major soil types in India and find the most suitable crops for your land.</p>
          </div>

          {/* Crop Disease Detection */}
          <div className="card reveal" onClick={() => setShowCropDiseaseDetection(true)}>
            <div className="icon">🌿</div>
            <h3><span className="notranslate">Crop Disease Detection</span></h3>
            <p>Upload plant images to detect diseases and get remedies.</p>
          </div>

          <div className="card reveal" onClick={() => setShowFertilizerPopup(true)}>
            <div className="icon">🌾</div>
            <h3><span className="notranslate">Fertilizer Recommendations</span></h3>
            <p>Get a crop-aware fertilizer plan based on soil pH and nutrient status.</p>
          </div>
          <div className="card reveal" onClick={() => setShowComingSoon(true)}>
            <div className="icon">
              <WifiOff size={32} strokeWidth={2} />
            </div>
            <h3><span className="notranslate">Offline Access</span></h3>
            <p>Use the app anytime, even without internet connectivity.</p>
          </div>
          <div className="card reveal" onClick={() => setShowPestManagement(true)}>
            <div className="icon">🐛</div>
            <h3><span className="notranslate">Pest Management</span></h3>
            <p>Early warnings & organic pest control tips.</p>
          </div>

          <div className="card reveal" onClick={() => setShowYieldPopup(true)}>
            <div className="icon">📊</div>
            <h3><span className="notranslate">Yield Prediction</span></h3>
            <p>AI predicts crop yield based on soil & weather data.</p>
          </div>

          <div className="card reveal" onClick={() => navigate("/schemes")}>
            <div className="icon">
              <Landmark size={32} strokeWidth={2} />
            </div>
            <h3><span className="notranslate">Govt Schemes</span></h3>
            <p>Direct subsidies, insurance, and financial benefits for farmers.</p>
          </div>

          <div className="card reveal" onClick={() => setShowAgriMarketplace(true)}>
            <div className="icon">🚜</div>
            <h3><span className="notranslate">Agri Marketplace</span></h3>
            <p>Rent or list farm equipment locally. Save costs and earn extra.</p>
          </div>

          <div className="card reveal" onClick={() => setShowAgriLMS(true)}>
            <div className="icon">🎓</div>
            <h3><span className="notranslate">Agri-LMS Academy</span></h3>
            <p>Access video tutorials on modern farming and earn completion certificates.</p>
          </div>

          <div className="card reveal" onClick={() => setShowQRTraceability(true)}>
            <div className="icon">🔍</div>
            <h3><span className="notranslate">QR-Farm Traceability</span></h3>
            <p>Generate QR codes for your produce. Let customers trace their food from farm to table.</p>
          </div>

          <div className="card reveal" onClick={() => setShowFarmPlanner3D(true)}>
            <div className="icon">🗺️</div>
            <h3><span className="notranslate">3D Farm Planner</span></h3>
            <p>Design your farm layout in interactive 3D. Optimize land usage and irrigation.</p>
          </div>

          <div className="card reveal" onClick={() => setShowProfitCalculator(true)}>
            <div className="icon">💰</div>
            <h3><span className="notranslate">Profit Calculator</span></h3>
            <p>Calculate your crop profits and ROI before planting.</p>
          </div>

          <div
            className="card reveal"
            style={{ cursor: "pointer" }}
            onClick={() => setShowFarmingMap(true)}
          >
            <div className="icon">
              <Map size={32} strokeWidth={2} />
            </div>
            <h3><span className="notranslate">Farming Map</span></h3>
            <p>View your fields, weather data, and crop locations on an interactive map.</p>
          </div>

           <div className="card reveal" onClick={() => navigate("/calendar")}>
            <div className="icon">
              <Calendar size={32} strokeWidth={2} />
            </div>
            <h3><span className="notranslate">Activity Calendar</span></h3>
            <p>Schedule sowing, watering, and harvesting with reminders.</p>
          </div>
          <div className="card reveal" onClick={() => navigate("/share-feedback")}>
            <div className="icon">
              <MessageSquare size={32} strokeWidth={2} />
            </div>
            <h3><span className="notranslate">Share Feedback</span></h3>
             <p>Help us improve <span className="notranslate" translate="no">Fasal Saathi</span> with your valuable suggestions.</p>
          </div>

          <div className="card reveal" onClick={() => setShowFarmDiary(true)}>
            <div className="icon">
              <Book size={32} strokeWidth={2} />
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
              <h2 style={{ margin: 0 }}>🌦️ Live Weather & Advisories</h2>
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
            <h3><span className="notranslate">Digital Farm Diary</span></h3>
            <p>Log daily farming activities, set task reminders, and export records as PDF reports.</p>
          </div>
        </div>
      </div>
          {showWeather && (
        <div className="weather-overlay" onClick={() => setShowWeather(false)}>
          <div className="weather-popup" onClick={(e)=>{e.stopPropagation()}}>
            <WeatherCard onClose={() => setShowWeather(false)} />
          </div>



        </div>

      )}

      {showSoilChatbot && (
        <div className="weather-overlay" onClick={() => setShowSoilChatbot(false)}>
          <div className="chatbot-popup" onClick={(e)=>{e.stopPropagation()}}>
            <SoilChatbot onClose={() => setShowSoilChatbot(false)} />
          </div>
        </div>
      )}

      {showSoilAnalysis && (
        <div className="weather-overlay" onClick={() => setShowSoilAnalysis(false)}>
          <div className="soil-analysis-popup" onClick={(e)=>e.stopPropagation()}>
            <button
              className="close-btn"
              onClick={() => setShowSoilAnalysis(false)}
              style={{ position: 'absolute', top: '12px', right: '12px', zIndex: 10 }}
            >
              ✕
            </button>
            <SoilAnalysis />
          </div>
        </div>
      )}

      {showSoilGuide && (
        <div className="weather-overlay" onClick={() => setShowSoilGuide(false)}>
          <div className="soil-analysis-popup" onClick={(e) => e.stopPropagation()}>
            <button
              className="close-btn"
              onClick={() => setShowSoilGuide(false)}
              style={{ position: 'absolute', top: '12px', right: '12px', zIndex: 10 }}
            >
              ✕
            </button>
            <SoilGuide />
          </div>
        </div>
      )}

      {showIrrigation && (
        <div className="weather-overlay" onClick={()=>setShowIrrigation(false)}
         style={{ alignItems: 'center', justifyContent: 'center' }}>
          <div onClick={(e)=>{e.stopPropagation()}}>
          <IrrigationGuidance onClose={() => setShowIrrigation(false)} />
            </div>
        </div>
      )}

      {showYieldPopup && (
        <div className="weather-overlay" onClick={()=>{closeYieldPopup()}}>
          <div className="yield-popup" onClick={(e)=>{e.stopPropagation()}}>
            <button
              className="close-btn"
              onClick={closeYieldPopup}
            >
              ✕
            </button>
            <h2>📊 Yield Prediction</h2>
            {yieldError && (
              <div style={{ color: '#dc2626', marginBottom: '16px', padding: '12px', background: '#fef2f2', borderRadius: '8px' }}>
                Error: {yieldError}
              </div>
            )}
            {yieldPrediction === null ? (
              <form onSubmit={fetchYield} className="yield-form">
                <div className="form-group">
                  <label>
                    Crop
                    <span className="tooltip-container">
                      <Info className="tooltip-icon" size={14} />
                      <span className="tooltip-text">The crop you want to predict yield for.</span>
                    </span>
                  </label>
                  <select
                    value={yieldForm.Crop}
                    onChange={(e) =>
                      updateYieldFormField("Crop", e.target.value)
                    }
                  >
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
                  <label>
                    Season
                    <span className="tooltip-container">
                      <Info className="tooltip-icon" size={14} />
                      <span className="tooltip-text">The growing season for the crop.</span>
                    </span>
                  </label>
                  <select
                    value={yieldForm.Season}
                    onChange={(e) =>
                      updateYieldFormField("Season", e.target.value)
                    }
                  >
                    <option value="Rabi">Rabi</option>
                    <option value="Kharif">Kharif</option>
                  </select>
                </div>
                <div className="form-group">
                  <label>
                    Covered Area (acres)
                    <span className="tooltip-container">
                      <Info className="tooltip-icon" size={14} />
                      <span className="tooltip-text">Total area planted in acres to gauge production volume.</span>
                    </span>
                  </label>
                  <input
                    type="number"
                    value={yieldForm.CropCoveredArea}
                    onChange={(e) =>
                      updateYieldFormField("CropCoveredArea", parseFloat(e.target.value))
                    }
                  />
                </div>
                <div className="form-group">
                  <label>
                    Crop Height (cm)
                    <span className="tooltip-container">
                      <Info className="tooltip-icon" size={14} />
                      <span className="tooltip-text">Estimated average height of the mature crop in centimeters.</span>
                    </span>
                  </label>
                  <input
                    type="number"
                    value={yieldForm.CHeight}
                    onChange={(e) =>
                      updateYieldFormField("CHeight", parseInt(e.target.value))
                    }
                  />
                </div>
                <div className="form-group">
                  <label>
                    Next Crop
                    <span className="tooltip-container">
                      <Info className="tooltip-icon" size={14} />
                      <span className="tooltip-text">The expected crop to be planted in the following season.</span>
                    </span>
                  </label>
                  <select
                    value={yieldForm.CNext}
                    onChange={(e) =>
                      updateYieldFormField("CNext", e.target.value)
                    }
                  >
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
                  <label>
                    Last Crop
                    <span className="tooltip-container">
                      <Info className="tooltip-icon" size={14} />
                      <span className="tooltip-text">The crop that was planted in the previous season.</span>
                    </span>
                  </label>
                  <select
                    value={yieldForm.CLast}
                    onChange={(e) =>
                      updateYieldFormField("CLast", e.target.value)
                    }
                  >
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
                  <label>
                    Transplanting Method
                    <span className="tooltip-container">
                      <Info className="tooltip-icon" size={14} />
                      <span className="tooltip-text">The method used to plant the crop (e.g. Drilling).</span>
                    </span>
                  </label>
                  <select
                    value={yieldForm.CTransp}
                    onChange={(e) =>
                      updateYieldFormField("CTransp", e.target.value)
                    }
                  >
                    <option value="Transplanting">Transplanting</option>
                    <option value="Drilling">Drilling</option>
                    <option value="Broadcasting">Broadcasting</option>
                    <option value="Seed Drilling">Seed Drilling</option>
                  </select>
                </div>
                <div className="form-group">
                  <label>
                    Irrigation Type
                    <span className="tooltip-container">
                      <Info className="tooltip-icon" size={14} />
                      <span className="tooltip-text">The technique for distributing water in the field.</span>
                    </span>
                  </label>
                  <select
                    value={yieldForm.IrriType}
                    onChange={(e) =>
                      updateYieldFormField("IrriType", e.target.value)
                    }
                  >
                    <option value="Flood">Flood</option>
                    <option value="Sprinkler">Sprinkler</option>
                    <option value="Drip">Drip</option>
                    <option value="Surface">Surface</option>
                  </select>
                </div>
                <div className="form-group">
                  <label>
                    Irrigation Source
                    <span className="tooltip-container">
                      <Info className="tooltip-icon" size={14} />
                      <span className="tooltip-text">The origin of the water used for irrigation.</span>
                    </span>
                  </label>
                  <select
                    value={yieldForm.IrriSource}
                    onChange={(e) =>
                      updateYieldFormField("IrriSource", e.target.value)
                    }
                  >
                    <option value="Groundwater">Groundwater</option>
                    <option value="Canal">Canal</option>
                    <option value="Rainfed">Rainfed</option>
                    <option value="Well">Well</option>
                    <option value="Tubewell">Tubewell</option>
                  </select>
                </div>
                <div className="form-group">
                  <label>
                    Irrigation Count
                    <span className="tooltip-container">
                      <Info className="tooltip-icon" size={14} />
                      <span className="tooltip-text">Number of times the crop is irrigated per season.</span>
                    </span>
                  </label>
                  <input
                    type="number"
                    value={yieldForm.IrriCount}
                    onChange={(e) =>
                      updateYieldFormField("IrriCount", parseInt(e.target.value))
                    }
                  />
                </div>
                <div className="form-group">
                  <label>
                    Water Coverage (%)
                    <span className="tooltip-container">
                      <Info className="tooltip-icon" size={14} />
                      <span className="tooltip-text">Percentage of field area receiving adequate water.</span>
                    </span>
                  </label>
                  <input
                    type="number"
                    max="100"
                    value={yieldForm.WaterCov}
                    onChange={(e) =>
                      updateYieldFormField("WaterCov", parseInt(e.target.value))
                    }
                  />
                </div>
                <div className="form-group full-width form-actions">
                  <button
                    type="submit"
                    className="action-btn"
                    disabled={yieldLoading}
                  >
                    {yieldLoading ? "Predicting..." : "Predict Yield"}
                  </button>
                  <button
                    type="button"
                    className="action-btn secondary"
                    onClick={closeYieldPopup}
                  >
                    Cancel
                  </button>
                </div>
              </form>
            ) : (
              <>
                <p className="yield-result">
                  Predicted Yield: <strong>{yieldPrediction.toFixed(2)}</strong>{" "}
                  quintals/acre
                </p>
                {yieldLastUpdated && (
                  <div style={{ marginBottom: '16px', display: 'flex', justifyContent: 'center' }}>
                    <LastUpdated timestamp={yieldLastUpdated} />
                  </div>
                )}
                <button
                  className="action-btn"
                  onClick={() => {
                    closeYieldPopup();
                  }}
                >
                  Predict Another
                </button>
              </>
            )}
          </div>
        </div>
      )}

      {showProfitCalculator && (
        <div className="weather-overlay" onClick={()=>{setShowProfitCalculator(false)}}>
          <div className="weather-popup profit-popup" onClick={(e)=>e.stopPropagation()}>
            <CropProfitCalculator />
            <button
              className="close-btn"
              onClick={() => setShowProfitCalculator(false)}
            >
              Close
            </button>
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
            <button
              className="close-btn"
              onClick={() => setShowFarmingMap(false)}
            >
              Close
            </button>
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
            <AgriLMS onClose={() => setShowAgriLMS(false)} />
          </div>
        </div>
      )}

      {showComingSoon && (
        <div className="weather-overlay" onClick={()=>{setShowComingSoon(false)}}>
          <div className="weather-popup coming-soon" onClick={(e)=>e.stopPropagation()}>
            <h2>🚧 Coming Soon</h2>
            <p>This feature is under development. Stay tuned!</p>
            <button
              className="close-btn"
              onClick={() => setShowComingSoon(false)}
            >
              Close
            </button>
          </div>
        </div>
      )}

      {/* AgriMarketplace Modal removed duplication */}

      {showFarmDiary && (
        <div className="weather-overlay" onClick={() => setShowFarmDiary(false)} style={{ zIndex: 1000, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <div onClick={(e) => e.stopPropagation()} style={{ width: '100%', display: 'flex', justifyContent: 'center' }}>
            <FarmDiary onClose={() => setShowFarmDiary(false)} />
          </div>
        </div>
      )}

      <br />
      <br />
    </section>
  );
}
