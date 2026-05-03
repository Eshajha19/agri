import React, { useState } from 'react';
import { useCropRecommendation } from './hooks/useCropRecommendation';
import './SmartCropRecommendation.css';
import {
  Leaf,
  MapPin,
  Droplet,
  Thermometer,
  Cloud,
  ChevronRight,
  Search,
  RotateCcw,
  Navigation,
  CheckCircle,
  AlertCircle,
  X,
} from 'lucide-react';

function SmartCropRecommendation({ onClose }) {
  const {
    location,
    setLocation,
    soilType,
    setSoilType,
    season,
    setSeason,
    recommendations,
    weatherData,
    loading,
    error,
    hasSearched,
    selectedCrop,
    setSelectedCrop,
    soilTypeSuggestions,
    seasonOptions,
    fetchRecommendations,
    resetRecommendations,
    useGeolocation,
  } = useCropRecommendation();

  const [showSoilDropdown, setShowSoilDropdown] = useState(false);
  const [showSeasonDropdown, setShowSeasonDropdown] = useState(false);

  const handleSearch = (e) => {
    e.preventDefault();
    fetchRecommendations();
  };

  return (
    <div className="smart-crop-recommendation">
      <div className="recommendation-header">
        <div className="header-title">
          <Leaf size={28} className="icon" />
          <div>
            <h2>🌱 Smart Crop Recommender</h2>
            <p>AI-powered recommendations based on weather, soil & season</p>
          </div>
        </div>
        <button className="close-btn" onClick={onClose}>
          <X size={24} />
        </button>
      </div>

      <div className="recommendation-container">
        {/* Input Form Section */}
        <div className="input-section">
          <form onSubmit={handleSearch} className="recommendation-form">
            {/* Location Input */}
            <div className="form-group">
              <label>
                <MapPin size={18} /> Location
              </label>
              <div className="input-with-btn">
                <input
                  type="text"
                  placeholder="Enter city or region..."
                  value={location}
                  onChange={(e) => setLocation(e.target.value)}
                  className="form-input"
                />
                <button
                  type="button"
                  className="geo-btn"
                  onClick={useGeolocation}
                  title="Use current location"
                >
                  <Navigation size={18} />
                </button>
              </div>
            </div>

            {/* Soil Type Dropdown */}
            <div className="form-group">
              <label>
                <Droplet size={18} /> Soil Type
              </label>
              <div className="custom-dropdown">
                <button
                  type="button"
                  className="dropdown-btn"
                  onClick={() => setShowSoilDropdown(!showSoilDropdown)}
                >
                  {soilType}
                  <ChevronRight size={16} />
                </button>
                {showSoilDropdown && (
                  <div className="dropdown-menu">
                    {soilTypeSuggestions.map((soil) => (
                      <button
                        key={soil}
                        type="button"
                        className={`dropdown-item ${soilType === soil ? 'active' : ''}`}
                        onClick={() => {
                          setSoilType(soil);
                          setShowSoilDropdown(false);
                        }}
                      >
                        {soil}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* Season Dropdown */}
            <div className="form-group">
              <label>
                <Cloud size={18} /> Season
              </label>
              <div className="custom-dropdown">
                <button
                  type="button"
                  className="dropdown-btn"
                  onClick={() => setShowSeasonDropdown(!showSeasonDropdown)}
                >
                  {seasonOptions.find((s) => s.value === season)?.label || 'Select Season'}
                  <ChevronRight size={16} />
                </button>
                {showSeasonDropdown && (
                  <div className="dropdown-menu">
                    {seasonOptions.map((opt) => (
                      <button
                        key={opt.value}
                        type="button"
                        className={`dropdown-item ${season === opt.value ? 'active' : ''}`}
                        onClick={() => {
                          setSeason(opt.value);
                          setShowSeasonDropdown(false);
                        }}
                      >
                        {opt.label}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* Error Display */}
            {error && (
              <div className="error-box">
                <AlertCircle size={16} />
                {error}
              </div>
            )}

            {/* Action Buttons */}
            <div className="form-actions">
              <button
                type="submit"
                className="btn-primary"
                disabled={loading}
              >
                <Search size={18} />
                {loading ? 'Analyzing...' : 'Get Recommendations'}
              </button>
              {hasSearched && (
                <button
                  type="button"
                  className="btn-secondary"
                  onClick={resetRecommendations}
                >
                  <RotateCcw size={18} />
                  Reset
                </button>
              )}
            </div>
          </form>
        </div>

        {/* Weather Information */}
        {weatherData && (
          <div className="weather-info">
            <h4>📊 Current Weather Conditions</h4>
            <div className="weather-grid">
              <div className="weather-card">
                <Thermometer size={20} />
                <span>{weatherData.temperature}°C</span>
                <small>Temperature</small>
              </div>
              <div className="weather-card">
                <Droplet size={20} />
                <span>{Math.round(weatherData.rainfall)}mm</span>
                <small>Rainfall</small>
              </div>
              <div className="weather-card">
                <Cloud size={20} />
                <span>{weatherData.humidity}%</span>
                <small>Humidity</small>
              </div>
            </div>
          </div>
        )}

        {/* Recommendations Section */}
        {hasSearched && (
          <div className="recommendations-section">
            <h3>💡 Recommended Crops</h3>

            {recommendations.length === 0 ? (
              <div className="no-recommendations">
                <AlertCircle size={40} />
                <p>No suitable crops found for these conditions.</p>
                <small>Try adjusting your soil type or season.</small>
              </div>
            ) : (
              <div className="recommendations-list">
                {recommendations.map((crop, index) => (
                  <div
                    key={index}
                    className={`crop-card ${selectedCrop?.crop === crop.crop ? 'active' : ''}`}
                    onClick={() => setSelectedCrop(crop)}
                  >
                    <div className="crop-card-header">
                      <div className="crop-info">
                        <h4>{crop.crop}</h4>
                        <p>{crop.description}</p>
                      </div>
                      <div className="crop-score">
                        <div className="score-badge" style={{
                          background: `hsl(${crop.score * 1.2}, 70%, 60%)`
                        }}>
                          {crop.score}%
                        </div>
                        <small>Compatibility</small>
                      </div>
                    </div>

                    {/* Reasons */}
                    <div className="crop-reasons">
                      {crop.reasons.map((reason, rIdx) => (
                        <div key={rIdx} className="reason">
                          {reason.includes('✓') ? (
                            <CheckCircle size={14} className="icon-success" />
                          ) : reason.includes('✗') ? (
                            <AlertCircle size={14} className="icon-warning" />
                          ) : (
                            <span className="icon-warning">⚠</span>
                          )}
                          <span>{reason}</span>
                        </div>
                      ))}
                    </div>

                    <div className="crop-season">
                      🗓️ Optimal Season: <strong>{crop.optimal_season.toUpperCase()}</strong>
                    </div>

                    <button className="crop-action">
                      Learn More <ChevronRight size={16} />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Loading State */}
        {loading && (
          <div className="loading-container">
            <div className="spinner"></div>
            <p>Analyzing weather patterns and soil compatibility...</p>
          </div>
        )}

        {/* Initial State */}
        {!hasSearched && !loading && (
          <div className="initial-state">
            <div className="icon-large">🌾</div>
            <h3>Start Your Crop Journey</h3>
            <p>Fill in your location and soil details to get personalized crop recommendations</p>
            <div className="tips">
              <h4>💡 Tips:</h4>
              <ul>
                <li>Use geolocation for accurate weather data</li>
                <li>Select the correct soil type for better recommendations</li>
                <li>Choose your upcoming farming season</li>
              </ul>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default SmartCropRecommendation;
