import React, { useEffect, useState } from "react";
import "./Advisor.css";
import WeatherCard from "./weather/WeatherCard";
import SoilChatbot from "./SoilChatbot";
import {
  Sun,
  Droplets,
  IndianRupee,
  Sprout,
  Languages,
  WifiOff,
} from "lucide-react";

export default function Advisor() {
  const [farmers, setFarmers] = useState(0);
  const [crops, setCrops] = useState(0);
  const [languages, setLanguages] = useState(0);

  const [showWeather, setShowWeather] = useState(false);
  const [showSoilChatbot, setShowSoilChatbot] = useState(false);
  const [showComingSoon, setShowComingSoon] = useState(false);

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
  }, []);
  const [yieldPrediction, setYieldPrediction] = useState(null);
  const [yieldError, setYieldError] = useState(null);
  const [showYieldPopup, setShowYieldPopup] = useState(false);
  const [yieldLoading, setYieldLoading] = useState(false);
  const [yieldForm, setYieldForm] = useState({
    Crop: "Paddy",
    CropCoveredArea: 50,
    CHeight: 50,
    CNext: "Lentil",
    CLast: "Pea",
    CTransp: "Transplanting",
    IrriType: "Flood",
    IrriSource: "Groundwater",
    IrriCount: 3,
    WaterCov: 50,
    Season: "Rabi",
  });

  const fetchYield = async (e) => {
    e.preventDefault();
    setYieldLoading(true);
    setYieldError(null);
    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(yieldForm),
      });
      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }
      const data = await response.json();
      setYieldPrediction(data.predicted_ExpYield);
      setShowYieldPopup(true);
    } catch (error) {
      console.error("Error fetching yield:", error);
      setYieldError(error.message || "Failed to get prediction");
    } finally {
      setYieldLoading(false);
    }
  };

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
              onClick={() => {
                setShowYieldPopup(false);
                setYieldPrediction(null);
                setYieldError(null);
              }}
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
                      setYieldForm({ ...yieldForm, Crop: e.target.value })
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
                      setYieldForm({ ...yieldForm, Season: e.target.value })
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
                      setYieldForm({
                        ...yieldForm,
                        CropCoveredArea: parseFloat(e.target.value),
                      })
                    }
                  />
                </div>
                <div className="form-group">
                  <label>Crop Height (cm)</label>
                  <input
                    type="number"
                    value={yieldForm.CHeight}
                    onChange={(e) =>
                      setYieldForm({
                        ...yieldForm,
                        CHeight: parseInt(e.target.value),
                      })
                    }
                  />
                </div>
                <div className="form-group">
                  <label>Next Crop</label>
                  <select
                    value={yieldForm.CNext}
                    onChange={(e) =>
                      setYieldForm({ ...yieldForm, CNext: e.target.value })
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
                      setYieldForm({ ...yieldForm, CLast: e.target.value })
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
                      setYieldForm({ ...yieldForm, CTransp: e.target.value })
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
                      setYieldForm({ ...yieldForm, IrriType: e.target.value })
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
                      setYieldForm({ ...yieldForm, IrriSource: e.target.value })
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
                      setYieldForm({
                        ...yieldForm,
                        IrriCount: parseInt(e.target.value),
                      })
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
                      setYieldForm({
                        ...yieldForm,
                        WaterCov: parseInt(e.target.value),
                      })
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
                    onClick={() => setShowYieldPopup(false)}
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
                    setShowYieldPopup(false);
                    setYieldPrediction(null);
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
    </section>
  );
}
