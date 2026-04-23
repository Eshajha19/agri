import React, { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import "./Advisor.css";
import WeatherCard from "./weather/WeatherCard";
import SoilChatbot from "./SoilChatbot";
import IrrigationGuidance from "./IrrigationGuidance";
import CropProfitCalculator from "./CropProfitCalculator";
import FarmingMap from "./FarmingMap";
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
} from "lucide-react";
import { useAdvisorStore } from "./stores/advisorStore";
import { useYieldPrediction } from "./hooks/useYieldPrediction";
import CropDiseaseDetection from "./CropDiseaseDetection";

export default function Advisor() {
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
  } = useAdvisorStore();

  const {
    yieldForm,
    updateYieldFormField,
    yieldPrediction,
    yieldError,
    yieldLoading,
    showYieldPopup,
    setShowYieldPopup,
    fetchYield,
    closeYieldPopup,
  } = useYieldPrediction();

  /* Animate stats on mount */
  useEffect(() => {
    let f = 0,
      c = 0,
      l = 0;
    const interval = setInterval(() => {
      if (f < 5000) setFarmers((f += 50));
      if (c < 120) setCrops((c += 2));
      if (l < 10) setLanguages((l += 1));
    }, 50);
    return () => clearInterval(interval);
  }, [setFarmers, setCrops, setLanguages]);

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

          <div className="card reveal" onClick={() => navigate("/community")}>
            <div className="icon">
              <MessageSquare size={32} strokeWidth={2} />
            </div>
            <h3>Farmer Community</h3>
            <p>
              Connect, share tips, and learn from other farmers in your region.
            </p>
          </div>
          <div className="card reveal" onClick={() => setShowIrrigation(true)}>
            <div className="icon">
              <Droplets size={32} strokeWidth={2} />
            </div>
            <h3>Irrigation Guidance</h3>
            <p>
              Water-saving tips and irrigation schedules tailored to your crops.
            </p>
          </div>

          <div className="card reveal" onClick={() => navigate("/market-prices")}>
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
          <div className="card reveal" onClick={() => setShowCropDiseaseDetection(true)}>
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

          <div className="card reveal" onClick={() => navigate("/schemes")}>
            <div className="icon">
              <Landmark size={32} strokeWidth={2} />
            </div>
            <h3>Govt Schemes</h3>
            <p>Direct subsidies, insurance, and financial benefits for farmers.</p>
          </div>

          <div className="card reveal" onClick={() => setShowProfitCalculator(true)}>
            <div className="icon">💰</div>
            <h3>Profit Calculator</h3>
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
            <h3>Farming Map</h3>
            <p>View your fields, weather data, and crop locations on an interactive map.</p>
          </div>

          <div className="card reveal" onClick={() => navigate("/calendar")}>
            <div className="icon">
              <Calendar size={32} strokeWidth={2} />
            </div>
            <h3>Activity Calendar</h3>
            <p>Schedule sowing, watering, and harvesting with reminders.</p>
          </div>

          <div className="card reveal" onClick={() => navigate("/share-feedback")}>
            <div className="icon">
              <MessageSquare size={32} strokeWidth={2} />
            </div>
            <h3>Share Feedback</h3>
            <p>Help us improve Fasal Saathi with your valuable suggestions.</p>
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

      <br />
      <br />
    </section>
  );
}
