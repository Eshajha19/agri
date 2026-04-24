import React, { useState } from "react";
import { Leaf, FlaskConical, Sprout, AlertCircle, RotateCcw, BarChart3, Wheat, Droplets } from "lucide-react";
import "./SoilAnalysis.css";

const NUTRIENT_RANGES = {
  nitrogen: { low: 140, medium: 280, unit: "kg/ha", label: "Nitrogen (N)" },
  phosphorus: { low: 10, medium: 25, unit: "kg/ha", label: "Phosphorus (P)" },
  potassium: { low: 110, medium: 280, unit: "kg/ha", label: "Potassium (K)" },
};

const CROP_DATABASE = [
  { name: "Rice", icon: "🌾", nitrogen: "high", phosphorus: "medium", potassium: "medium" },
  { name: "Wheat", icon: "🌿", nitrogen: "high", phosphorus: "medium", potassium: "low" },
  { name: "Maize", icon: "🌽", nitrogen: "high", phosphorus: "high", potassium: "medium" },
  { name: "Sugarcane", icon: "🎋", nitrogen: "high", phosphorus: "medium", potassium: "high" },
  { name: "Cotton", icon: "☁️", nitrogen: "medium", phosphorus: "medium", potassium: "high" },
  { name: "Soybean", icon: "🫘", nitrogen: "low", phosphorus: "high", potassium: "medium" },
  { name: "Groundnut", icon: "🥜", nitrogen: "low", phosphorus: "medium", potassium: "medium" },
  { name: "Pulses", icon: "🫛", nitrogen: "low", phosphorus: "medium", potassium: "low" },
  { name: "Millets", icon: "🌱", nitrogen: "low", phosphorus: "low", potassium: "low" },
  { name: "Mustard", icon: "🌼", nitrogen: "medium", phosphorus: "medium", potassium: "low" },
  { name: "Potato", icon: "🥔", nitrogen: "high", phosphorus: "high", potassium: "high" },
  { name: "Tomato", icon: "🍅", nitrogen: "medium", phosphorus: "high", potassium: "high" },
];

const FERTILIZER_MAP = {
  low_nitrogen: [
    { name: "Urea (46-0-0)", dosage: "80–120 kg/ha", priority: "high" },
    { name: "Ammonium Sulphate", dosage: "100–150 kg/ha", priority: "medium" },
    { name: "Vermicompost", dosage: "2–5 tons/ha", priority: "medium" },
  ],
  low_phosphorus: [
    { name: "DAP (18-46-0)", dosage: "50–100 kg/ha", priority: "high" },
    { name: "Single Super Phosphate", dosage: "150–200 kg/ha", priority: "medium" },
    { name: "Bone Meal (Organic)", dosage: "200–400 kg/ha", priority: "medium" },
  ],
  low_potassium: [
    { name: "Muriate of Potash (MOP)", dosage: "50–80 kg/ha", priority: "high" },
    { name: "Sulphate of Potash (SOP)", dosage: "60–100 kg/ha", priority: "medium" },
    { name: "Wood Ash (Organic)", dosage: "500–1000 kg/ha", priority: "low" },
  ],
  balanced: [
    { name: "NPK Complex (10-26-26)", dosage: "100–150 kg/ha", priority: "medium" },
    { name: "Farm Yard Manure (FYM)", dosage: "5–10 tons/ha", priority: "medium" },
    { name: "Neem Cake", dosage: "100–200 kg/ha", priority: "low" },
  ],
};

function getNutrientLevel(nutrient, value) {
  const range = NUTRIENT_RANGES[nutrient];
  if (value < range.low) return "low";
  if (value < range.medium) return "medium";
  return "high";
}

function getSoilQuality(levels) {
  const scores = { low: 1, medium: 2, high: 3 };
  const avg = (scores[levels.nitrogen] + scores[levels.phosphorus] + scores[levels.potassium]) / 3;
  if (avg >= 2.5) return { label: "Excellent", score: 90, color: "#10b981" };
  if (avg >= 2.0) return { label: "Good", score: 72, color: "#22c55e" };
  if (avg >= 1.5) return { label: "Moderate", score: 55, color: "#f59e0b" };
  return { label: "Poor", score: 30, color: "#ef4444" };
}

function getRecommendedCrops(levels) {
  return CROP_DATABASE.filter((crop) => {
    let matchScore = 0;
    if (crop.nitrogen === levels.nitrogen) matchScore++;
    if (crop.phosphorus === levels.phosphorus) matchScore++;
    if (crop.potassium === levels.potassium) matchScore++;
    return matchScore >= 2;
  }).slice(0, 6);
}

function getRecommendedFertilizers(levels) {
  const fertilizers = [];
  if (levels.nitrogen === "low") fertilizers.push(...FERTILIZER_MAP.low_nitrogen);
  if (levels.phosphorus === "low") fertilizers.push(...FERTILIZER_MAP.low_phosphorus);
  if (levels.potassium === "low") fertilizers.push(...FERTILIZER_MAP.low_potassium);
  if (fertilizers.length === 0) fertilizers.push(...FERTILIZER_MAP.balanced);
  return fertilizers;
}

export default function SoilAnalysis() {
  const [formData, setFormData] = useState({ nitrogen: "", phosphorus: "", potassium: "" });
  const [errors, setErrors] = useState({});
  const [results, setResults] = useState(null);
  const [hasAnalyzed, setHasAnalyzed] = useState(false);

  const validateInputs = () => {
    const newErrors = {};
    Object.keys(NUTRIENT_RANGES).forEach((key) => {
      if (!formData[key] || parseFloat(formData[key]) < 0) {
        newErrors[key] = `Please enter a valid ${NUTRIENT_RANGES[key].label} value`;
      }
    });
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
    if (errors[name]) {
      setErrors((prev) => ({ ...prev, [name]: "" }));
    }
  };

  const handleAnalyze = (e) => {
    e.preventDefault();
    if (!validateInputs()) return;

    const n = parseFloat(formData.nitrogen);
    const p = parseFloat(formData.phosphorus);
    const k = parseFloat(formData.potassium);

    const levels = {
      nitrogen: getNutrientLevel("nitrogen", n),
      phosphorus: getNutrientLevel("phosphorus", p),
      potassium: getNutrientLevel("potassium", k),
    };

    const quality = getSoilQuality(levels);
    const crops = getRecommendedCrops(levels);
    const fertilizers = getRecommendedFertilizers(levels);

    setResults({ levels, quality, crops, fertilizers, values: { nitrogen: n, phosphorus: p, potassium: k } });
    setHasAnalyzed(true);
  };

  const handleReset = () => {
    setFormData({ nitrogen: "", phosphorus: "", potassium: "" });
    setResults(null);
    setErrors({});
    setHasAnalyzed(false);
  };

  const getLevelBadgeClass = (level) => {
    if (level === "high") return "badge-high";
    if (level === "medium") return "badge-medium";
    return "badge-low";
  };

  return (
    <div className="soil-analysis-page">
      <div className="soil-analysis-container">
        <div className="soil-analysis-header">
          <div className="sa-header-content">
            <div className="sa-header-icon-wrap">
              <FlaskConical className="sa-header-icon" />
            </div>
            <div>
              <h1>Soil Analysis Tool</h1>
              <p>Analyze your soil nutrients and get personalized crop & fertilizer recommendations</p>
            </div>
          </div>
          <div className="sa-header-badges">
            <span className="sa-badge"><Leaf size={14} /> NPK Analysis</span>
            <span className="sa-badge"><Sprout size={14} /> Crop Matching</span>
            <span className="sa-badge"><Droplets size={14} /> Fertilizer Guide</span>
          </div>
        </div>

        <div className="soil-analysis-content">
          <div className="sa-form-section">
            <div className="sa-section-title">
              <BarChart3 size={22} />
              <h2>Enter Soil Nutrient Values</h2>
            </div>

            <form onSubmit={handleAnalyze} className="sa-form">
              {Object.entries(NUTRIENT_RANGES).map(([key, range]) => (
                <div className="sa-form-group" key={key}>
                  <label htmlFor={`sa-${key}`}>
                    <span className={`sa-nutrient-dot dot-${key}`}></span>
                    {range.label}
                    <span className="sa-unit-hint">{range.unit}</span>
                  </label>
                  <div className="sa-input-wrapper">
                    <input
                      type="number"
                      id={`sa-${key}`}
                      name={key}
                      placeholder={`e.g. ${range.low + Math.floor((range.medium - range.low) / 2)}`}
                      value={formData[key]}
                      onChange={handleInputChange}
                      min="0"
                      step="0.1"
                      className={errors[key] ? "sa-input-error" : ""}
                    />
                    <span className="sa-unit-suffix">{range.unit}</span>
                  </div>
                  {errors[key] && (
                    <div className="sa-error-msg">
                      <AlertCircle size={14} />
                      {errors[key]}
                    </div>
                  )}
                  <div className="sa-range-hint">
                    <span>Low: &lt;{range.low}</span>
                    <span>Medium: {range.low}–{range.medium}</span>
                    <span>High: &gt;{range.medium}</span>
                  </div>
                </div>
              ))}

              <div className="sa-form-buttons">
                <button type="submit" className="sa-btn-analyze">
                  <FlaskConical size={18} />
                  Analyze Soil
                </button>
                {hasAnalyzed && (
                  <button type="button" onClick={handleReset} className="sa-btn-reset">
                    <RotateCcw size={18} />
                    Reset
                  </button>
                )}
              </div>
            </form>
          </div>

          {hasAnalyzed && results && (
            <div className="sa-results-section">
              <div className="sa-quality-card">
                <h2>Soil Quality</h2>
                <div className="sa-gauge-container">
                  <div className="sa-gauge">
                    <svg viewBox="0 0 200 120" className="sa-gauge-svg">
                      <path
                        d="M 20 100 A 80 80 0 0 1 180 100"
                        fill="none"
                        stroke="#e5e7eb"
                        strokeWidth="16"
                        strokeLinecap="round"
                      />
                      <path
                        d="M 20 100 A 80 80 0 0 1 180 100"
                        fill="none"
                        stroke={results.quality.color}
                        strokeWidth="16"
                        strokeLinecap="round"
                        strokeDasharray={`${(results.quality.score / 100) * 251.2} 251.2`}
                        className="sa-gauge-fill"
                      />
                    </svg>
                    <div className="sa-gauge-label">
                      <span className="sa-gauge-score" style={{ color: results.quality.color }}>
                        {results.quality.score}
                      </span>
                      <span className="sa-gauge-text">{results.quality.label}</span>
                    </div>
                  </div>
                </div>

                <div className="sa-nutrient-levels">
                  {Object.entries(results.levels).map(([key, level]) => (
                    <div className="sa-nutrient-bar" key={key}>
                      <div className="sa-nutrient-info">
                        <span className={`sa-nutrient-dot dot-${key}`}></span>
                        <span className="sa-nutrient-name">{NUTRIENT_RANGES[key].label}</span>
                        <span className={`sa-level-badge ${getLevelBadgeClass(level)}`}>{level}</span>
                      </div>
                      <div className="sa-bar-track">
                        <div
                          className={`sa-bar-fill bar-${key}`}
                          style={{
                            width: `${Math.min((results.values[key] / NUTRIENT_RANGES[key].medium) * 100, 100)}%`,
                          }}
                        ></div>
                      </div>
                      <span className="sa-nutrient-value">{results.values[key]} {NUTRIENT_RANGES[key].unit}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="sa-crops-card">
                <div className="sa-section-title">
                  <Wheat size={22} />
                  <h2>Recommended Crops</h2>
                </div>
                {results.crops.length > 0 ? (
                  <div className="sa-crops-grid">
                    {results.crops.map((crop, i) => (
                      <div className="sa-crop-item" key={i}>
                        <span className="sa-crop-icon">{crop.icon}</span>
                        <span className="sa-crop-name">{crop.name}</span>
                        <div className="sa-crop-tags">
                          <span className={`sa-tag tag-n-${crop.nitrogen}`}>N:{crop.nitrogen[0].toUpperCase()}</span>
                          <span className={`sa-tag tag-p-${crop.phosphorus}`}>P:{crop.phosphorus[0].toUpperCase()}</span>
                          <span className={`sa-tag tag-k-${crop.potassium}`}>K:{crop.potassium[0].toUpperCase()}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="sa-no-results">
                    <Sprout size={40} />
                    <p>No strong crop matches found for this specific nutrient combination. Consider soil amendments to broaden your options.</p>
                  </div>
                )}
              </div>

              <div className="sa-fertilizer-card">
                <div className="sa-section-title">
                  <Droplets size={22} />
                  <h2>Fertilizer Recommendations</h2>
                </div>
                <div className="sa-fertilizer-list">
                  {results.fertilizers.map((fert, i) => (
                    <div className="sa-fertilizer-item" key={i}>
                      <div className="sa-fert-header">
                        <span className="sa-fert-name">{fert.name}</span>
                        <span className={`sa-priority-badge priority-${fert.priority}`}>{fert.priority}</span>
                      </div>
                      <div className="sa-fert-dosage">
                        <span className="sa-dosage-label">Recommended Dosage</span>
                        <span className="sa-dosage-value">{fert.dosage}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className={`sa-summary-card ${results.quality.score >= 55 ? "summary-positive" : "summary-warning"}`}>
                {results.quality.score >= 55 ? (
                  <>
                    <h3>✅ Your Soil is in {results.quality.label} Condition</h3>
                    <p>
                      With a soil quality score of {results.quality.score}/100, your land is well-suited for farming.
                      Focus on maintaining nutrient balance with the recommended fertilizers above for best yields.
                    </p>
                  </>
                ) : (
                  <>
                    <h3>⚠️ Your Soil Needs Improvement</h3>
                    <p>
                      With a soil quality score of {results.quality.score}/100, your soil has nutrient deficiencies.
                      Apply the recommended fertilizers and consider adding organic matter like compost or green manure
                      to improve soil health over time.
                    </p>
                  </>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
