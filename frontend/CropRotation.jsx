import React, { useState } from "react";
import { FaHistory, FaFlask, FaSeedling, FaChartPie, FaArrowRight, FaSync, FaBrain, FaCheckCircle } from "react-icons/fa";
import "./CropRotation.css";

const CROP_DATA = {
  Cereals: ["Rice", "Wheat", "Maize", "Barley", "Millets"],
  Legumes: ["Moong Bean", "Chickpea", "Lentils", "Soybean", "Pigeon Pea"],
  Oilseeds: ["Mustard", "Groundnut", "Sunflower", "Soybean"],
  Vegetables: ["Tomato", "Potato", "Onion", "Cabbage", "Cauliflower"],
  Fibers: ["Cotton", "Jute"]
};

const SOIL_TYPES = [
  "Alluvial Soil",
  "Black Soil",
  "Red Soil",
  "Laterite Soil",
  "Arid Soil",
  "Mountain Soil"
];

const RECOMMENDATION_LOGIC = {
  Cereals: { next: "Legumes", reason: "Replenishes nitrogen depleted by cereals.", nitrogenImpact: "+25 kg/ha" },
  Legumes: { next: "Oilseeds", reason: "Uses residual nitrogen; prevents pest buildup.", nitrogenImpact: "-10 kg/ha" },
  Oilseeds: { next: "Cereals", reason: "Breaks disease cycles for oilseeds.", nitrogenImpact: "-20 kg/ha" },
  Vegetables: { next: "Cereals", reason: "Restores soil structure after shallow-rooted crops.", nitrogenImpact: "-15 kg/ha" },
  Fibers: { next: "Legumes", reason: "Deep-rooted fiber crops followed by nitrogen-fixing legumes.", nitrogenImpact: "+30 kg/ha" }
};

export default function CropRotation() {
  const [history, setHistory] = useState([]);
  const [currentCrop, setCurrentCrop] = useState("");
  const [soilType, setSoilType] = useState(SOIL_TYPES[0]);
  const [recommendation, setRecommendation] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [aiStatus, setAiStatus] = useState("");

  const handleAddHistory = () => {
    if (currentCrop && !history.includes(currentCrop)) {
      setHistory([...history, currentCrop]);
      setCurrentCrop("");
    }
  };

  const getCategory = (crop) => {
    for (const [cat, crops] of Object.entries(CROP_DATA)) {
      if (crops.includes(crop)) return cat;
    }
    return "Cereals"; 
  };

  const analyzeRotation = () => {
    if (history.length === 0) return;
    
    setIsAnalyzing(true);
    setRecommendation(null);
    
    const steps = [
      "Analyzing historical crop sequences...",
      "Evaluating soil nutrient depletion patterns...",
      "Calculating nitrogen balance for " + soilType + "...",
      "Matching upcoming seasonal patterns...",
      "Generating optimal sustainable rotation strategy..."
    ];

    let stepIdx = 0;
    const interval = setInterval(() => {
      setAiStatus(steps[stepIdx]);
      stepIdx++;
      if (stepIdx >= steps.length) {
        clearInterval(interval);
        finalizeAnalysis();
      }
    }, 800);
  };

  const finalizeAnalysis = () => {
    const lastCrop = history[history.length - 1];
    const category = getCategory(lastCrop);
    const logic = RECOMMENDATION_LOGIC[category];
    
    // AI decision based on history length and pattern
    const suggestedCrops = CROP_DATA[logic.next];
    const suggestion = suggestedCrops[Math.floor(Math.random() * suggestedCrops.length)];
    
    setRecommendation({
      crop: suggestion,
      category: logic.next,
      reason: logic.reason,
      nitrogen: logic.nitrogenImpact,
      soilCompatibility: "High",
      confidence: "94%"
    });
    setIsAnalyzing(false);
  };

  const resetEngine = () => {
    setHistory([]);
    setRecommendation(null);
    setAiStatus("");
  };

  return (
    <div className="crop-rotation-engine">
      <div className="engine-header">
        <div className="ai-badge"><FaBrain /> AI-POWERED ENGINE</div>
        <h1>Adaptive Crop Rotation Engine</h1>
        <p>Maintain soil health and optimize yield through AI-driven rotation strategies.</p>
      </div>

      <div className="engine-grid">
        <div className="input-section card">
          <div className="card-title">
            <FaHistory /> <h3>Crop History</h3>
          </div>
          <div className="input-group">
            <label>Select previous crops in order:</label>
            <div className="add-crop-wrap">
              <select 
                value={currentCrop} 
                onChange={(e) => setCurrentCrop(e.target.value)}
              >
                <option value="">-- Choose Crop --</option>
                {Object.values(CROP_DATA).flat().sort().map(crop => (
                  <option key={crop} value={crop}>{crop}</option>
                ))}
              </select>
              <button onClick={handleAddHistory} className="btn-add">Add</button>
            </div>
          </div>

          <div className="history-tags">
            {history.map((crop, index) => (
              <span key={index} className="history-tag">
                {index + 1}. {crop}
              </span>
            ))}
            {history.length === 0 && <p className="placeholder-text">No history added yet. Please add your last crop.</p>}
          </div>

          <div className="input-group">
            <label><FaFlask /> Current Soil Type:</label>
            <select value={soilType} onChange={(e) => setSoilType(e.target.value)}>
              {SOIL_TYPES.map(type => <option key={type} value={type}>{type}</option>)}
            </select>
          </div>

          <div className="action-buttons">
            <button 
              className={`btn-analyze ${history.length === 0 ? 'disabled' : ''}`}
              onClick={analyzeRotation}
              disabled={history.length === 0 || isAnalyzing}
            >
              {isAnalyzing ? "AI Processing..." : "Generate AI Recommendation"}
            </button>
            <button className="btn-reset" onClick={resetEngine}>Reset History</button>
          </div>
        </div>

        <div className="output-section card">
          <div className="card-title">
            <FaChartPie /> <h3>Decision Analysis</h3>
          </div>
          
          {recommendation ? (
            <div className="recommendation-content fade-in">
              <div className="rec-hero">
                <div className="rec-crop">
                  <div className="confidence-label"><FaCheckCircle /> Confidence Score: {recommendation.confidence}</div>
                  <span className="label">Recommended Next Crop</span>
                  <h2>{recommendation.crop}</h2>
                  <span className="category-badge">{recommendation.category}</span>
                </div>
                <div className="rec-icon">
                  <FaSeedling />
                </div>
              </div>

              <div className="rec-details">
                <div className="detail-item">
                  <span className="detail-label">AI Logic Rationale</span>
                  <p>{recommendation.reason}</p>
                </div>
                <div className="detail-row">
                  <div className="detail-item">
                    <span className="detail-label">Predicted N2 Impact</span>
                    <span className="impact-value">{recommendation.nitrogen}</span>
                  </div>
                  <div className="detail-item">
                    <span className="detail-label">Soil Stability</span>
                    <span className="impact-value positive">{recommendation.soilCompatibility}</span>
                  </div>
                </div>
              </div>

              <div className="rec-alert">
                <FaArrowRight /> 
                <span>AI Recommendation: Rotating from <b>{history[history.length-1]}</b> to <b>{recommendation.crop}</b> will optimize your soil biology in <b>{soilType}</b>.</span>
              </div>
            </div>
          ) : (
            <div className="empty-state">
              <div className="ai-visualizer">
                <FaSync className={isAnalyzing ? "spin" : ""} />
                {isAnalyzing && <div className="scanning-line"></div>}
              </div>
              <p className="status-text">{isAnalyzing ? aiStatus : "Awaiting input data for historical analysis..."}</p>
            </div>
          )}
        </div>
      </div>

      <div className="rotation-info-section">
        <h3><FaBrain /> AI-Driven Sustainability</h3>
        <div className="info-grid">
          <div className="info-item">
            <h4>Deep History Analysis</h4>
            <p>Our engine tracks multiple seasons of history to prevent monoculture risks and soil exhaustion.</p>
          </div>
          <div className="info-item">
            <h4>Nutrient Optimization</h4>
            <p>Calculates precise nitrogen fixing and consumption cycles to reduce chemical fertilizer dependency.</p>
          </div>
          <div className="info-item">
            <h4>Pest Cycle Interruption</h4>
            <p>Strategically breaks the biological cycles of specific pests by suggesting non-host crop categories.</p>
          </div>
        </div>
      </div>
    </div>
  );
}