import React, { useState, useCallback } from 'react';
import './CropRecommendationAdvisor.css';
import {
  Leaf,
  AlertCircle,
  CheckCircle,
  Activity,
  MapPin,
  Droplets,
  Zap,
  TrendingUp,
  Loader,
  X,
  ChevronDown,
  ChevronUp,
  Info,
} from 'lucide-react';
import apiClient from './services/api';

export default function CropRecommendationAdvisor({ onClose }) {
  // Form state
  const [formData, setFormData] = useState({
    soil_ph: 6.5,
    nitrogen: 25,
    phosphorus: 15,
    potassium: 100,
    location: '',
    season: 'kharif',
    area_size: null,
  });

  // UI state
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);
  const [recommendations, setRecommendations] = useState(null);
  const [soilAnalysis, setSoilAnalysis] = useState(null);
  const [warnings, setWarnings] = useState([]);
  const [expandedCrop, setExpandedCrop] = useState(null);

  // Input validation
  const isFormValid = () => {
    return (
      formData.location.trim() !== '' &&
      formData.soil_ph >= 4.0 &&
      formData.soil_ph <= 9.0 &&
      formData.nitrogen >= 0 &&
      formData.phosphorus >= 0 &&
      formData.potassium >= 0
    );
  };

   // Handle input changes
   const handleInputChange = (e) => {
     const { name, value, type } = e.target;
     // Parse float for numeric inputs (including range)
     const numericValue = type === 'number' || type === 'range' ? parseFloat(value) : value;
     setFormData((prev) => ({
       ...prev,
       [name]: numericValue,
     }));
   };

  // Reset form
  const handleReset = () => {
    setFormData({
      soil_ph: 6.5,
      nitrogen: 25,
      phosphorus: 15,
      potassium: 100,
      location: '',
      season: 'kharif',
      area_size: null,
    });
    setRecommendations(null);
    setSoilAnalysis(null);
    setWarnings([]);
    setError('');
    setSuccess(false);
    setExpandedCrop(null);
  };

  // Fetch crop recommendations from API
  const handleSubmit = useCallback(
    async (e) => {
      e.preventDefault();

      if (!isFormValid()) {
        setError('Please fill in all required fields with valid values');
        return;
      }

      setLoading(true);
      setError('');
      setSuccess(false);
      setRecommendations(null);

      try {
        const payload = {
          soil_ph: formData.soil_ph,
          nitrogen: formData.nitrogen,
          phosphorus: formData.phosphorus,
          potassium: formData.potassium,
          location: formData.location,
          season: formData.season,
          area_size: formData.area_size,
        };

        const response = await apiClient.post('/api/crop/recommend', payload);

        if (response.data.success) {
          setRecommendations(response.data.recommendations);
          setSoilAnalysis(response.data.soil_analysis);
          setWarnings(response.data.warnings || []);
          setSuccess(true);
          setError('');
        } else {
          setError(response.data.error || 'Failed to get recommendations');
          setRecommendations(null);
        }
      } catch (err) {
        const errorMsg =
          err.response?.data?.detail ||
          err.message ||
          'Failed to fetch crop recommendations. Please try again.';
        setError(errorMsg);
        setRecommendations(null);
      } finally {
        setLoading(false);
      }
    },
    [formData]
  );

  const toggleCropExpand = (index) => {
    setExpandedCrop(expandedCrop === index ? null : index);
  };

  const getSoilInterpretation = () => {
    if (!soilAnalysis) return null;

    return (
      <div className="soil-analysis">
        <h4 className="analysis-title">
          <Droplets size={18} /> Soil Analysis
        </h4>
        <div className="analysis-grid">
          <div className="analysis-item">
            <span className="label">pH Level:</span>
            <span className={`value ph-${soilAnalysis.ph_level.toLowerCase()}`}>
              {soilAnalysis.ph_value} ({soilAnalysis.ph_level})
            </span>
          </div>
          <div className="analysis-item">
            <span className="label">Nitrogen:</span>
            <span className={`value n-${soilAnalysis.nitrogen_level.toLowerCase()}`}>
              {soilAnalysis.nitrogen_value} ppm ({soilAnalysis.nitrogen_level})
            </span>
          </div>
          <div className="analysis-item">
            <span className="label">Phosphorus:</span>
            <span className={`value p-${soilAnalysis.phosphorus_level.toLowerCase()}`}>
              {soilAnalysis.phosphorus_value} ppm ({soilAnalysis.phosphorus_level})
            </span>
          </div>
          <div className="analysis-item">
            <span className="label">Potassium:</span>
            <span className={`value k-${soilAnalysis.potassium_level.toLowerCase()}`}>
              {soilAnalysis.potassium_value} ppm ({soilAnalysis.potassium_level})
            </span>
          </div>
        </div>
      </div>
    );
  };

  const getRecommendationCard = (crop, index) => {
    const isExpanded = expandedCrop === index;
    const scoreColor =
      crop.compatibility_score >= 80
        ? '#10b981'
        : crop.compatibility_score >= 60
          ? '#f59e0b'
          : '#ef4444';

    return (
      <div key={index} className="recommendation-card">
        <div className="card-header" onClick={() => toggleCropExpand(index)}>
          <div className="card-title-section">
            <Leaf size={20} className="crop-icon" />
            <div className="crop-info">
              <h5 className="crop-name">{crop.crop}</h5>
              <p className="season-label">{formData.season}</p>
            </div>
          </div>
          <div className="score-section">
            <div
              className="compatibility-score"
              style={{ borderColor: scoreColor }}
            >
              <span className="score-value" style={{ color: scoreColor }}>
                {crop.compatibility_score}%
              </span>
              <span className="score-label">Match</span>
            </div>
            {isExpanded ? (
              <ChevronUp size={20} />
            ) : (
              <ChevronDown size={20} />
            )}
          </div>
        </div>

        {isExpanded && (
          <div className="card-body">
            <div className="reasons-section">
              <h6>Why this crop?</h6>
              <ul className="reasons-list">
                {crop.reasons.map((reason, idx) => (
                  <li key={idx}>
                    <CheckCircle size={16} className="check-icon" />
                    {reason}
                  </li>
                ))}
              </ul>
            </div>

            <div className="fertilizer-section">
              <h6>
                <Zap size={16} /> Fertilizer Recommendation
              </h6>
              <p className="fertilizer-text">{crop.recommended_fertilizer}</p>
            </div>

            <button
              className="learn-more-btn"
              onClick={() => {
                // Could navigate to crop-specific page or open more info
                console.log(`Learn more about ${crop.crop}`);
              }}
            >
              Learn More About {crop.crop}
            </button>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="crop-recommendation-advisor">
      <div className="advisor-header">
        <div className="header-content">
          <h2>
            <Leaf size={28} className="header-icon" /> Smart Crop Recommender
          </h2>
          <p>AI-powered recommendations based on your soil parameters</p>
        </div>
        {onClose && (
          <button className="close-btn" onClick={onClose} aria-label="Close">
            <X size={24} />
          </button>
        )}
      </div>

      <div className="advisor-container">
        {/* Left: Form */}
        <div className="form-section">
          <form onSubmit={handleSubmit} className="recommendation-form">
            <h3>
              <Activity size={20} /> Soil Parameters
            </h3>

            {/* Location */}
            <div className="form-group">
              <label htmlFor="location">
                <MapPin size={18} /> Location *
              </label>
              <input
                id="location"
                type="text"
                name="location"
                value={formData.location}
                onChange={handleInputChange}
                placeholder="Enter your farm location (city/region)"
                className="form-input"
                required
              />
            </div>

            {/* Season */}
            <div className="form-group">
              <label htmlFor="season">
                <TrendingUp size={18} /> Season
              </label>
              <select
                id="season"
                name="season"
                value={formData.season}
                onChange={handleInputChange}
                className="form-select"
              >
                <option value="kharif">🌾 Kharif (Jun-Oct)</option>
                <option value="rabi">🌾 Rabi (Oct-Mar)</option>
                <option value="summer">☀️ Summer (Mar-Jun)</option>
              </select>
            </div>

            {/* Soil pH */}
            <div className="form-group">
              <label htmlFor="soil_ph">
                <Droplets size={18} /> Soil pH {formData.soil_ph.toFixed(1)}
              </label>
              <input
                id="soil_ph"
                type="range"
                name="soil_ph"
                min="4.0"
                max="9.0"
                step="0.1"
                value={formData.soil_ph}
                onChange={handleInputChange}
                className="form-range"
              />
              <div className="range-labels">
                <span>Acidic (4)</span>
                <span>Neutral (7)</span>
                <span>Alkaline (9)</span>
              </div>
            </div>

            {/* Nitrogen */}
            <div className="form-group">
              <label htmlFor="nitrogen">
                <Zap size={18} /> Nitrogen (ppm): {formData.nitrogen}
              </label>
              <input
                id="nitrogen"
                type="range"
                name="nitrogen"
                min="0"
                max="100"
                step="1"
                value={formData.nitrogen}
                onChange={handleInputChange}
                className="form-range"
              />
              <div className="range-labels">
                <span>Low (0)</span>
                <span>Medium (50)</span>
                <span>High (100)</span>
              </div>
            </div>

            {/* Phosphorus */}
            <div className="form-group">
              <label htmlFor="phosphorus">
                <Zap size={18} /> Phosphorus (ppm): {formData.phosphorus}
              </label>
              <input
                id="phosphorus"
                type="range"
                name="phosphorus"
                min="0"
                max="50"
                step="1"
                value={formData.phosphorus}
                onChange={handleInputChange}
                className="form-range"
              />
              <div className="range-labels">
                <span>Low (0)</span>
                <span>Medium (25)</span>
                <span>High (50)</span>
              </div>
            </div>

            {/* Potassium */}
            <div className="form-group">
              <label htmlFor="potassium">
                <Zap size={18} /> Potassium (ppm): {formData.potassium}
              </label>
              <input
                id="potassium"
                type="range"
                name="potassium"
                min="0"
                max="300"
                step="5"
                value={formData.potassium}
                onChange={handleInputChange}
                className="form-range"
              />
              <div className="range-labels">
                <span>Low (0)</span>
                <span>Medium (150)</span>
                <span>High (300)</span>
              </div>
            </div>

            {/* Area Size */}
            <div className="form-group">
              <label htmlFor="area_size">Farm Area (hectares) - Optional</label>
              <input
                id="area_size"
                type="number"
                name="area_size"
                value={formData.area_size || ''}
                onChange={handleInputChange}
                placeholder="Enter farm size for detailed fertilizer recommendations"
                className="form-input"
                min="0.1"
                step="0.1"
              />
            </div>

            {/* Error Message */}
            {error && (
              <div className="alert alert-error">
                <AlertCircle size={18} />
                <span>{error}</span>
              </div>
            )}

            {/* Form Buttons */}
            <div className="form-buttons">
              <button
                type="submit"
                className="btn btn-primary"
                disabled={loading || !isFormValid()}
              >
                {loading ? (
                  <>
                    <Loader size={18} className="spinner" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Leaf size={18} />
                    Get Recommendations
                  </>
                )}
              </button>
              <button
                type="button"
                className="btn btn-secondary"
                onClick={handleReset}
                disabled={loading}
              >
                Reset
              </button>
            </div>
          </form>
        </div>

        {/* Right: Results */}
        <div className="results-section">
          {success && recommendations && soilAnalysis && (
            <>
              {/* Success Message */}
              <div className="alert alert-success">
                <CheckCircle size={18} />
                <span>Recommendations ready! Based on your soil analysis.</span>
              </div>

              {/* Soil Analysis */}
              {getSoilInterpretation()}

              {/* Warnings */}
              {warnings.length > 0 && (
                <div className="warnings-section">
                  <h4 className="warnings-title">
                    <AlertCircle size={18} /> Important Warnings
                  </h4>
                  <ul className="warnings-list">
                    {warnings.map((warning, idx) => (
                      <li key={idx} className="warning-item">
                        {warning}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Recommendations */}
              <div className="recommendations-list">
                <h4 className="recommendations-title">
                  <Leaf size={18} /> Recommended Crops
                </h4>
                <p className="recommendations-subtitle">
                  Click on any crop to see detailed information
                </p>
                <div className="crops-container">
                  {recommendations.map((crop, index) =>
                    getRecommendationCard(crop, index)
                  )}
                </div>
              </div>

              {/* Info Box */}
              <div className="info-box">
                <Info size={16} />
                <p>
                  These recommendations are based on soil parameters and regional climate
                  patterns. Consult with local agricultural experts for crop-specific
                  guidance.
                </p>
              </div>
            </>
          )}

          {!success && !loading && !recommendations && (
            <div className="empty-state">
              <Leaf size={48} className="empty-icon" />
              <h4>Enter Your Soil Parameters</h4>
              <p>Fill in the form on the left to get AI-powered crop recommendations</p>
            </div>
          )}

          {loading && (
            <div className="loading-state">
              <div className="loader">
                <Loader size={48} className="spinner-large" />
              </div>
              <h4>Analyzing Your Soil...</h4>
              <p>Our AI is processing your soil parameters to find the best crops</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
