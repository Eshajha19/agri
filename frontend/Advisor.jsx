import React, { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import "./Advisor.css";
import WeatherCard from "./weather/WeatherCard";
import SoilChatbot from "./SoilChatbot";
import GreenPractices from "./GreenPractices";
import { Leaf } from "lucide-react";


import {
  Sun,
  Droplets,
  IndianRupee,
  Sprout,
  Languages,
  WifiOff,
} from "lucide-react";
import { useAdvisorStore } from "./stores/advisorStore";
import { useYieldPrediction } from "./hooks/useYieldPrediction";

export default function Advisor({ userData }) {
  const navigate = useNavigate();
  const userProfile = userData;
  const currentReputation = userData?.reputation || 0;
  const {
    farmers,
    setCarmers,
    crops,
    setCrops,
    languages,
    setLanguages,
    showWeather,
    setShowWeather,
    showSoilChatbot,
    setShowSoilChatbot,
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

  const {
    yieldForm,
    updateYieldFormField,
    yieldPrediction,
    yieldError,
    yieldLoading,
    showYieldPopup,
    fetchYield,
    closeYieldPopup,
  } = useYieldPrediction();

  /* Animate stats on mount */
  useEffect(() => {
    let f = 0,
      c = 0,
      l = 0;
    const interval = setInterval(() => {
      if (f < 5000) setCarmers((f += 50));
      if (c < 120) setCrops((c += 2));
      if (l < 10) setLanguages((l += 1));
    }, 50);
    return () => clearInterval(interval);
  }, []);

  return (
    <section className="advisor">
      <div className="floating-icons">
        <span>🌱</span>
        <span>☀️</span>
        <span>💧</span>
        <span>₹</span>
      </div>

      <div className="advisor-hero">
        <h1 className="fade-in">🌱 AI-Powered Agricultural Advisor</h1>
        <p className="fade-in">
          Personalized guidance for <span className="highlight">weather</span>,{" "}
          <span className="highlight">markets</span>, and{" "}
          <span className="highlight">soil health</span>.
        </p>
        <button
          className="get-started shine"
          onClick={() => setShowSoilChatbot(true)}
        >
          🚀 Get Started
        </button>
      </div>

      <div className="advisor-stats">
        <div className="stat">
          <h2>{farmers}+</h2>
          <p>Farmers Connected</p>
        </div>
        <div className="stat">
          <h2>{crops}+</h2>
          <p>Crops Analyzed</p>
        </div>
        <div className="stat">
          <h2>{languages}+</h2>
          <p>Languages Available</p>
        </div>
      </div>

      <br />
      <br />

      <div className="advisor-highlights">
        <h2 className="slide-in">✨ Features</h2>
        <br />
        <br />
        <div className="cards">
          <div
            className="card reveal"
            style={{ cursor: "pointer" }}
            onClick={() => setShowWeather(true)}
          >
            <div className="icon">
              <Sun size={32} strokeWidth={2} />
            </div>
            <h3>Weather Forecasts</h3>
            <p>
              Accurate daily & weekly weather insights for farming decisions.
            </p>
          </div>

          <div className="card reveal" onClick={() => setShowComingSoon(true)}>
            <div className="icon">👨‍🌾👩‍🌾</div>
            <h3>Farmer Community</h3>
            <p>
              Connect, share tips, and learn from other farmers in your region.
            </p>
          </div>
          <div className="card reveal" onClick={() => setShowComingSoon(true)}>
            <div className="icon">
              <Droplets size={32} strokeWidth={2} />
            </div>
            <h3>Irrigation Guidance</h3>
            <p>
              Water-saving tips and irrigation schedules tailored to your crops.
            </p>
          </div>

          <div className="card reveal" onClick={() => setShowComingSoon(true)}>
            <div className="icon">
              <IndianRupee size={32} strokeWidth={2} />
            </div>
            <h3>Market Price Guidance</h3>
            <p>
              Market trends and price alerts to help you sell at the best time.
            </p>
          </div>

          <div
            className="card reveal"
            style={{ cursor: "pointer" }}
            onClick={() => setShowSoilChatbot(true)}
          >
            <div className="icon">
              <Sprout size={32} strokeWidth={2} />
            </div>
            <h3>Soil Health</h3>
            <p>Get soil analysis & recommendations via AI chatbot.</p>
          </div>

          {/* Crop Disease Detection */}
          <div className="card reveal" onClick={() => setShowComingSoon(true)}>
            <div className="icon">🌿</div>
            <h3>Crop Disease Detection</h3>
            <p>Upload plant images to detect diseases and get remedies.</p>
          </div>

          <div className="card reveal" onClick={() => setShowComingSoon(true)}>
            <div className="icon">🌾</div>
            <h3>Fertilizer Recommendations</h3>
            <p>AI-powered fertilizer advice tailored to your soil & crops.</p>
          </div>
          <div className="card reveal" onClick={() => setShowComingSoon(true)}>
            <div className="icon">
              <WifiOff size={32} strokeWidth={2} />
            </div>
            <h3>Offline Access</h3>
            <p>Use the app anytime, even without internet connectivity.</p>
          </div>
          <div className="card reveal" onClick={() => setShowComingSoon(true)}>
            <div className="icon">🐛</div>
            <h3>Pest Management</h3>
            <p>Early warnings & organic pest control tips.</p>
          </div>

          <div className="card reveal" onClick={() => setShowYieldPopup(true)}>
            <div className="icon">📊</div>
            <h3>Yield Prediction</h3>
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
            borderRadius: "18px",
            background: "linear-gradient(135deg, rgba(255,255,255,0.96), rgba(239,253,245,0.98))",
            boxShadow: "0 18px 45px rgba(15, 23, 42, 0.08)",
          }}
        >
          <WeatherCard embedded={true} />
        </div>
      </div>
          {showWeather && (
        <div className="weather-overlay">
          <div className="weather-popup">
            <WeatherCard onClose={() => setShowWeather(false)} />
          </div>
        </div>
      )}

      {showSoilChatbot && (
        <div className="weather-overlay">
          <div className="chatbot-popup">
            <SoilChatbot onClose={() => setShowSoilChatbot(false)} />
          </div>
        </div>
      )}
      {showYieldPopup && (
        <div className="weather-overlay">
          <div className="yield-popup">
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
                  <label>Crop</label>
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
                  <label>Season</label>
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
                  <label>Covered Area (acres)</label>
                  <input
                    type="number"
                    value={yieldForm.CropCoveredArea}
                    onChange={(e) =>
                      updateYieldFormField("CropCoveredArea", parseFloat(e.target.value))
                    }
                  />
                </div>
                <div className="form-group">
                  <label>Crop Height (cm)</label>
                  <input
                    type="number"
                    value={yieldForm.CHeight}
                    onChange={(e) =>
                      updateYieldFormField("CHeight", parseInt(e.target.value))
                    }
                  />
                </div>
                <div className="form-group">
                  <label>Next Crop</label>
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
                  <label>Last Crop</label>
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
                  <label>Transplanting Method</label>
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
                  <label>Irrigation Type</label>
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
                  <label>Irrigation Source</label>
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
                  <label>Irrigation Count</label>
                  <input
                    type="number"
                    value={yieldForm.IrriCount}
                    onChange={(e) =>
                      updateYieldFormField("IrriCount", parseInt(e.target.value))
                    }
                  />
                </div>
                <div className="form-group">
                  <label>Water Coverage (%)</label>
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

      {showComingSoon && (
        <div className="weather-overlay">
          <div className="weather-popup coming-soon">
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

      <br />
      <br />


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
