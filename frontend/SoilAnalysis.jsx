import React, { useState, useEffect } from "react";
import {
  Leaf,
  FlaskConical,
  Sprout,
  AlertCircle,
  RotateCcw,
  BarChart3,
  Wheat,
  Droplets,
  Brain,
} from "lucide-react";
import "./SoilAnalysis.css";

const NUTRIENT_RANGES = {
  nitrogen: { veryLow: 80, low: 140, medium: 280, high: 360, unit: "kg/ha", label: "Nitrogen (N)" },
  phosphorus: { veryLow: 5, low: 10, medium: 25, high: 40, unit: "kg/ha", label: "Phosphorus (P)" },
  potassium: { veryLow: 70, low: 110, medium: 280, high: 420, unit: "kg/ha", label: "Potassium (K)" },
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
    { name: "Urea (46-0-0)", dosage: "80–120 kg/ha", priority: "high", type: "Fast-release", benefit: "Boosts green leafy growth quickly." },
    { name: "Ammonium Sulphate", dosage: "100–150 kg/ha", priority: "medium", type: "High-Nitrogen", benefit: "Supports strong vegetative growth in acid soils." },
    { name: "Blood Meal", dosage: "20–30 kg/ha", priority: "medium", type: "Organic", benefit: "Provides rapid nitrogen release with better soil microbial support." },
    { name: "Vermicompost", dosage: "2–5 tons/ha", priority: "medium", type: "Organic", benefit: "Builds long-term nitrogen availability and improves soil structure." },
  ],
  low_phosphorus: [
    { name: "DAP (18-46-0)", dosage: "50–100 kg/ha", priority: "high", type: "Fast-release", benefit: "Delivers phosphorus and nitrogen for early crop establishment." },
    { name: "Single Super Phosphate", dosage: "150–200 kg/ha", priority: "medium", type: "Balanced", benefit: "Boosts root development and flowering over time." },
    { name: "Bone Meal", dosage: "200–400 kg/ha", priority: "medium", type: "Organic", benefit: "Releases phosphorus slowly for sustained use." },
    { name: "Rock Phosphate", dosage: "250–400 kg/ha", priority: "low", type: "Slow-release", benefit: "Feeds phosphorus gradually and improves soil fertility long-term." },
  ],
  low_potassium: [
    { name: "Muriate of Potash (MOP)", dosage: "50–80 kg/ha", priority: "high", type: "Fast-release", benefit: "Restores potassium quickly for improved fruit quality." },
    { name: "Sulphate of Potash (SOP)", dosage: "60–100 kg/ha", priority: "medium", type: "Balanced", benefit: "Adds potassium without increasing chloride levels." },
    { name: "Sulphate of Potash Magnesia (SPM)", dosage: "80–120 kg/ha", priority: "medium", type: "Balanced", benefit: "Supports potassium and magnesium for stronger plant health." },
    { name: "Wood Ash", dosage: "500–1000 kg/ha", priority: "low", type: "Organic", benefit: "Provides potassium and calcium while raising soil pH." },
  ],
  balanced: [
    { name: "NPK Complex (10-26-26)", dosage: "100–150 kg/ha", priority: "medium", type: "Balanced", benefit: "Delivers a complete nutrient blend for stable crop growth." },
    { name: "Farm Yard Manure (FYM)", dosage: "5–10 tons/ha", priority: "medium", type: "Organic", benefit: "Improves soil texture while releasing nutrients slowly." },
    { name: "Neem Cake", dosage: "100–200 kg/ha", priority: "low", type: "Organic", benefit: "Enhances soil biology and acts as a mild pest deterrent." },
    { name: "Green Manure", dosage: "2–4 tons/ha", priority: "low", type: "Organic", benefit: "Builds soil organic matter and stabilizes nutrient release." },
  ],
};

const SOIL_INSIGHTS = {
  nitrogen: {
    title: "Nitrogen for leafy vigor",
    description:
      "Nitrogen is the key to strong vegetative growth, darker leaves, and good protein levels in crops.",
    actionLow: "Add nitrogen-rich fertilizers and plant legume cover crops.",
    actionMedium: "Maintain nitrogen with compost and balanced fertilization.",
    actionHigh: "Avoid excess nitrogen and focus on balanced nutrition and soil testing.",
  },
  phosphorus: {
    title: "Phosphorus for roots & blooms",
    description:
      "Phosphorus supports root development, flowering and early fruit set, especially in young plants.",
    actionLow: "Use phosphorus-rich fertilizers and organic residues near the root zone.",
    actionMedium: "Keep phosphorus levels stable with crop rotation and mulch.",
    actionHigh: "Prevent runoff and use slow-release phosphorus sources." ,
  },
  potassium: {
    title: "Potassium for strength",
    description:
      "Potassium improves water use efficiency, disease resistance, and crop quality.",
    actionLow: "Supply potassium through potash fertilizers and organic amendments.",
    actionMedium: "Sustain potassium with residue recycling and proper irrigation.",
    actionHigh: "Limit potassium inputs and watch for nutrient balance issues.",
  },
};

function getSoilInsights(levels) {
  return Object.keys(levels).map((key) => {
    const insight = SOIL_INSIGHTS[key];
    return {
      nutrient: key,
      label: levels[key],
      title: insight.title,
      description: insight.description,
      tip:
        levels[key] === "low" || levels[key] === "verylow"
          ? insight.actionLow
          : levels[key] === "medium"
          ? insight.actionMedium
          : insight.actionHigh,
    };
  });
}

function getNutrientLevel(nutrient, value) {
  const range = NUTRIENT_RANGES[nutrient];
  if (value < range.veryLow) return "verylow";
  if (value < range.low) return "low";
  if (value < range.medium) return "medium";
  if (value < range.high) return "high";
  return "veryhigh";
}

function normalizeLevel(level) {
  if (level === "verylow") return "low";
  if (level === "veryhigh") return "high";
  return level;
}

function formatLevelLabel(level) {
  if (level === "verylow") return "Very Low";
  if (level === "veryhigh") return "Very High";
  return level.charAt(0).toUpperCase() + level.slice(1);
}

function getSoilQuality(levels) {
  const scores = { verylow: 0.5, low: 1, medium: 2, high: 3, veryhigh: 4 };
  const avg = (scores[levels.nitrogen] + scores[levels.phosphorus] + scores[levels.potassium]) / 3;
  if (avg >= 3.0) return { label: "Excellent", score: 95, color: "#10b981" };
  if (avg >= 2.2) return { label: "Good", score: 78, color: "#22c55e" };
  if (avg >= 1.4) return { label: "Moderate", score: 60, color: "#f59e0b" };
  return { label: "Poor", score: 32, color: "#ef4444" };
}

function getRecommendedCrops(levels) {
  return CROP_DATABASE.filter((crop) => {
    let matchScore = 0;
    if (crop.nitrogen === normalizeLevel(levels.nitrogen)) matchScore++;
    if (crop.phosphorus === normalizeLevel(levels.phosphorus)) matchScore++;
    if (crop.potassium === normalizeLevel(levels.potassium)) matchScore++;
    return matchScore >= 2;
  }).slice(0, 6);
}

function getRecommendedFertilizers(levels) {
  const fertilizers = [];
  if (levels.nitrogen === "low" || levels.nitrogen === "verylow") fertilizers.push(...FERTILIZER_MAP.low_nitrogen);
  if (levels.phosphorus === "low" || levels.phosphorus === "verylow") fertilizers.push(...FERTILIZER_MAP.low_phosphorus);
  if (levels.potassium === "low" || levels.potassium === "verylow") fertilizers.push(...FERTILIZER_MAP.low_potassium);
  if (fertilizers.length === 0) fertilizers.push(...FERTILIZER_MAP.balanced);
  return fertilizers;
}

function getSoilUpdateSuggestions(levels) {
  return [
    {
      nutrient: "Nitrogen",
      level: levels.nitrogen,
      advice:
        levels.nitrogen === "low"
          ? "Boost nitrogen with urea, green manure, and legume cover crops."
          : levels.nitrogen === "medium"
          ? "Maintain balance with compost, legumes, and slow-release nitrogen."
          : "Avoid excess nitrogen and monitor nutrient levels regularly.",
    },
    {
      nutrient: "Phosphorus",
      level: levels.phosphorus,
      advice:
        levels.phosphorus === "low"
          ? "Apply phosphorus-rich fertilizers like DAP and bone meal, and use organic residues."
          : levels.phosphorus === "medium"
          ? "Keep phosphorus steady with balanced fertilizer and crop rotation."
          : "Use controlled phosphorus sources and prevent runoff loss.",
    },
    {
      nutrient: "Potassium",
      level: levels.potassium,
      advice:
        levels.potassium === "low"
          ? "Increase potassium with MOP, SOP, and composted plant material."
          : levels.potassium === "medium"
          ? "Support potassium with mulch, organic matter, and regular soil testing."
          : "Maintain potassium balance and avoid over-application to protect root health.",
    },
  ];
}

function detectSoilType(levels) {
  if (
    levels.nitrogen === "high" &&
    levels.potassium === "high"
  ) {
    return "Loamy Soil";
  }

  if (levels.phosphorus === "low") {
    return "Sandy Soil";
  }

  return "Clay Soil";
}

function getUpdateFrequency(score) {
  if (score >= 72) return "Recheck soil nutrients every 3-4 months and maintain existing practices.";
  if (score >= 55) return "Retest each season and adjust fertilizer applications as needed.";
  return "Retest in 6-8 weeks after applying amendments for best recovery tracking.";
}

export default function SoilAnalysis() {
  const [formData, setFormData] = useState({ nitrogen: "", phosphorus: "", potassium: "" });
  const [errors, setErrors] = useState({});
  const [results, setResults] = useState(null);
  const [hasAnalyzed, setHasAnalyzed] = useState(false);
  const [animatedScore, setAnimatedScore] = useState(0);

const SOIL_TIPS = [
  "🌱 Rotate crops regularly to maintain nitrogen balance.",
  "💧 Avoid overwatering to prevent nutrient leaching.",
  "🌿 Add compost yearly to improve soil structure.",
  "☀️ Test soil every season for accurate fertilizer planning.",
];

const [tipIndex, setTipIndex] = useState(0);

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
    const updateSuggestions = getSoilUpdateSuggestions(levels);
    const updateFrequency = getUpdateFrequency(quality.score);
    const soilInsights = getSoilInsights(levels);
    const soilType = detectSoilType(levels);

    setResults({
      levels,
      quality,
      crops,
      fertilizers,
      values: { nitrogen: n, phosphorus: p, potassium: k },
      updateSuggestions,
      updateFrequency,
      soilInsights,
      soilType,
    });
    setHasAnalyzed(true);
  };

  useEffect(() => {
    if (results) {
      let start = 0;
      setAnimatedScore(0);

      const interval = setInterval(() => {
        start += 1;

        if (start >= results.quality.score) {
          start = results.quality.score;
          clearInterval(interval);
        }

        setAnimatedScore(start);
      }, 15);

      return () => clearInterval(interval);
    }
  }, [results]);

useEffect(() => {
  const interval = setInterval(() => {
    setTipIndex((prev) => (prev + 1) % SOIL_TIPS.length);
  }, 4000);

  return () => clearInterval(interval);
}, []);
  const handleReset = () => {
    setFormData({ nitrogen: "", phosphorus: "", potassium: "" });
    setResults(null);
    setErrors({});
    setHasAnalyzed(false);
  };

  const getLevelBadgeClass = (level) => {
    if (level === "veryhigh") return "badge-veryhigh";
    if (level === "high") return "badge-high";
    if (level === "medium") return "badge-medium";
    if (level === "low") return "badge-low";
    return "badge-verylow";
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
              <h1><span className="notranslate">Soil Analysis Tool</span></h1>
              <p>Analyze your soil nutrients and get personalized crop & fertilizer recommendations</p>
            </div>
          </div>
          <div className="sa-header-badges">
            <span className="sa-badge"><Leaf size={14} /> <span className="notranslate">NPK Analysis</span></span>
            <span className="sa-badge"><Sprout size={14} /> <span className="notranslate">Crop Matching</span></span>
            <span className="sa-badge"><Droplets size={14} /> <span className="notranslate">Fertilizer Guide</span></span>
          </div>
        </div>

        <div className="soil-analysis-content">
          <div className="sa-tips-banner">
            {SOIL_TIPS[tipIndex]}
          </div>
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
                  {formData[key] && (
                <div
                className={`sa-live-status ${getNutrientLevel(
                key,
                parseFloat(formData[key] || 0)
              )}`}
  >
              Status:
              { " "}
              {formatLevelLabel(
                getNutrientLevel(
              key,
              parseFloat(formData[key] || 0)
            )
              )}
              </div>
              )}
                  <div className="sa-range-hint">
                    <span>Very Low: &lt;{range.veryLow}</span>
                    <span>Low: {range.veryLow}–{range.low}</span>
                    <span>Medium: {range.low}–{range.medium}</span>
                    <span>High: {range.medium}–{range.high}</span>
                    <span>Very High: &gt;{range.high}</span>
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

                <button
  type="button"
  className="sa-btn-download"
>
  Download Report
</button>
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
                <div className="sa-circle-grid">
  {Object.entries(results.levels).map(([key, level]) => {
    const percentage = Math.min(
      (results.values[key] /
        NUTRIENT_RANGES[key].high) *
        100,
      100
    );

    return (
      <div className="sa-circle-card" key={key}>
        <div
          className="sa-circle"
          style={{
            background: `conic-gradient(
              var(--${key}-color)
              ${percentage * 3.6}deg,
              #e5e7eb 0deg
            )`,
          }}
        >
          <div className="sa-circle-inner">
            <span>{Math.round(percentage)}%</span>
          </div>
        </div>

        <h4>{NUTRIENT_RANGES[key].label}</h4>
        <p>{formatLevelLabel(level)}</p>
      </div>
    );
  })}
</div>

                <div className="sa-nutrient-levels">
                  {Object.entries(results.levels).map(([key, level]) => (
                    <div className="sa-nutrient-bar" key={key}>
                      <div className="sa-nutrient-info">
                        <span className={`sa-nutrient-dot dot-${key}`}></span>
                        <span className="sa-nutrient-name">{NUTRIENT_RANGES[key].label}</span>
                        <span className={`sa-level-badge ${getLevelBadgeClass(level)}`}>{formatLevelLabel(level)}</span>
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

              <div className="sa-insights-card">
                <div className="sa-section-title">
                  <FlaskConical size={22} />
                  <h2>Detailed Soil Insights</h2>
                </div>
                <div className="sa-insight-grid">
                  {results.soilInsights.map((insight, index) => (
                    <div className="sa-insight-item" key={index}>
                      <div className={`sa-insight-tag insight-${insight.label}`}>{insight.label.toUpperCase()}</div>
                      <h3>{insight.title}</h3>
                      <p>{insight.description}</p>
                      <p className="sa-insight-tip">{insight.tip}</p>
                    </div>
                  ))}
                </div>
              </div>
              <div className="sa-soil-type-card">
  <h3>Estimated Soil Type</h3>
  <p>{results.soilType}</p>
</div>

              <div className="sa-update-card">
                <div className="sa-section-title">
                  <Leaf size={22} />
                  <h2>Soil Update Plan</h2>
                </div>
                <p className="sa-update-intro">
                  Follow these focused updates to improve your soil health and strengthen future yields.
                </p>
                <div className="sa-update-grid">
                  {results.updateSuggestions.map((item, index) => (
                    <div className="sa-update-item" key={index}>
                      <span className={`sa-update-chip update-${item.level}`}>{item.level}</span>
                      <h3>{item.nutrient}</h3>
                      <p>{item.advice}</p>
                    </div>
                  ))}
                </div>
                <div className="sa-update-frequency">
                  <h3>Recommended Review</h3>
                  <p>{results.updateFrequency}</p>
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
                      <div className="sa-fert-meta">
                        <span className="sa-fert-type">{fert.type}</span>
                        <span className="sa-fert-benefit">{fert.benefit}</span>
                      </div>
                      <div className="sa-fert-dosage">
                        <span className="sa-dosage-label">Recommended Dosage</span>
                        <span className="sa-dosage-value">{fert.dosage}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="sa-ai-summary">
  <div className="sa-ai-header">
    <Brain size={22} />
    <h2>AI Soil Recommendation</h2>
  </div>

  <p>
    Based on the nutrient profile,
    your soil is best suited for
    {" "}
    <strong>
      {results.crops
        .slice(0, 3)
        .map((c) => c.name)
        .join(", ")}
    </strong>.

    Focus on improving
    {" "}
    {results.levels.nitrogen === "low" &&
      " nitrogen "}
    {results.levels.phosphorus === "low" &&
      " phosphorus "}
    {results.levels.potassium === "low" &&
      " potassium "}
    levels for better crop yield.
  </p>
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
