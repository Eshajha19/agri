import React, { useState, useRef } from "react";
import "./CropQualityGrading.css";
import Loader from "./Loader";

const CropQualityGrading = () => {
  const [cropType, setCropType] = useState("tomato");
  const [images, setImages] = useState([]);
  const [assessments, setAssessments] = useState([]);
  const [loading, setLoading] = useState(false);
  const [batchMode, setBatchMode] = useState(false);
  const [trends, setTrends] = useState(null);
  const [supportedCrops, setSupportedCrops] = useState([
    "tomato",
    "potato",
    "grain",
    "fruit",
  ]);
  const fileInputRef = useRef(null);
  const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

  // Fetch supported crops
  React.useEffect(() => {
    const fetchCrops = async () => {
      try {
        const response = await fetch(`${API_BASE}/api/quality/supported-crops`);
        const data = await response.json();
        if (data.success) {
          setSupportedCrops(data.crops);
        }
      } catch (error) {
        console.error("Error fetching supported crops:", error);
      }
    };
    fetchCrops();
  }, []);

  // Handle file selection
  const handleFileSelect = (e) => {
    const files = Array.from(e.target.files);
    const validFiles = files.filter((f) => f.type.startsWith("image/"));
    setImages(validFiles);
  };

  // Convert file to base64
  const fileToBase64 = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const base64 = reader.result.split(",")[1];
        resolve(base64);
      };
      reader.onerror = reject;
    });
  };

  // Assess single crop
  const assessSingleCrop = async () => {
    if (!images || images.length === 0) {
      alert("Please select an image");
      return;
    }

    setLoading(true);
    try {
      const base64 = await fileToBase64(images[0]);
      const response = await fetch(
        `${API_BASE}/api/quality/assess-single`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            crop_type: cropType,
            image_base64: base64,
          }),
        }
      );

      const data = await response.json();
      if (data.success) {
        setAssessments([data.data]);
      } else {
        alert("Assessment failed: " + data.detail);
      }
    } catch (error) {
      console.error("Error:", error);
      alert("Error assessing crop quality");
    } finally {
      setLoading(false);
    }
  };

  // Assess batch crops
  const assessBatchCrops = async () => {
    if (!images || images.length === 0) {
      alert("Please select at least one image");
      return;
    }

    setLoading(true);
    try {
      const base64Images = await Promise.all(
        images.map((f) => fileToBase64(f))
      );
      const response = await fetch(
        `${API_BASE}/api/quality/assess-batch`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            crop_type: cropType,
            images_base64: base64Images,
          }),
        }
      );

      const data = await response.json();
      if (data.success) {
        setAssessments(data.data.assessments);
      } else {
        alert("Batch assessment failed: " + data.detail);
      }
    } catch (error) {
      console.error("Error:", error);
      alert("Error in batch assessment");
    } finally {
      setLoading(false);
    }
  };

  // Get quality trends
  const fetchQualityTrends = async () => {
    setLoading(true);
    try {
      const response = await fetch(
        `${API_BASE}/api/quality/trends`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            crop_type: cropType,
            days: 7,
          }),
        }
      );

      const data = await response.json();
      if (data.success) {
        setTrends(data.data);
      } else {
        alert("Failed to fetch trends");
      }
    } catch (error) {
      console.error("Error:", error);
      alert("Error fetching trends");
    } finally {
      setLoading(false);
    }
  };

  // Get grade color
  const getGradeColor = (grade) => {
    const colors = {
      A: "#27ae60",
      B: "#3498db",
      C: "#f39c12",
      D: "#e74c3c",
      F: "#95a5a6",
    };
    return colors[grade] || "#95a5a6";
  };

  // Get grade label
  const getGradeLabel = (grade) => {
    const labels = {
      A: "Premium",
      B: "Good",
      C: "Standard",
      D: "Below Average",
      F: "Reject",
    };
    return labels[grade] || "Unknown";
  };

  return (
    <div className="crop-quality-grading">
      <div className="quality-header">
        <h1>🌾 Advanced Crop Quality Grading System</h1>
        <p>Automatic quality assessment using computer vision and AI</p>
      </div>

      {loading && <Loader />}

      <div className="quality-container">
        {/* Settings Panel */}
        <div className="quality-settings">
          <div className="setting-group">
            <label>Crop Type</label>
            <select
              value={cropType}
              onChange={(e) => setCropType(e.target.value)}
            >
              {supportedCrops.map((crop) => (
                <option key={crop} value={crop}>
                  {crop.charAt(0).toUpperCase() + crop.slice(1)}
                </option>
              ))}
            </select>
          </div>

          <div className="setting-group">
            <label>Assessment Mode</label>
            <div className="mode-toggle">
              <button
                className={!batchMode ? "active" : ""}
                onClick={() => setBatchMode(false)}
              >
                Single Crop
              </button>
              <button
                className={batchMode ? "active" : ""}
                onClick={() => setBatchMode(true)}
              >
                Batch Processing
              </button>
            </div>
          </div>

          <div className="setting-group">
            <label>Upload Image{batchMode ? "s" : ""}</label>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              multiple={batchMode}
              onChange={handleFileSelect}
              hidden
            />
            <button
              className="upload-btn"
              onClick={() => fileInputRef.current.click()}
            >
              📷 Choose Image{batchMode ? "s" : ""}
            </button>
            {images.length > 0 && (
              <p className="selected-files">{images.length} file(s) selected</p>
            )}
          </div>

          <div className="action-buttons">
            {batchMode ? (
              <button className="assess-btn" onClick={assessBatchCrops}>
                🔍 Assess Batch
              </button>
            ) : (
              <button className="assess-btn" onClick={assessSingleCrop}>
                🔍 Assess Quality
              </button>
            )}
            <button className="trend-btn" onClick={fetchQualityTrends}>
              📊 View Trends
            </button>
          </div>
        </div>

        {/* Results Panel */}
        <div className="quality-results">
          {assessments.length > 0 && (
            <div className="results-container">
              <h2>Assessment Results</h2>

              {assessments.map((assessment, idx) => (
                <div key={idx} className="assessment-card">
                  {assessment.error ? (
                    <div className="error-message">{assessment.error}</div>
                  ) : (
                    <>
                      <div className="grade-display">
                        <div
                          className="grade-badge"
                          style={{
                            backgroundColor: getGradeColor(assessment.grade),
                          }}
                        >
                          <span className="grade-letter">
                            {assessment.grade}
                          </span>
                          <span className="grade-label">
                            {getGradeLabel(assessment.grade)}
                          </span>
                        </div>

                        <div className="score-display">
                          <div className="score-value">{assessment.score}</div>
                          <div className="score-label">Quality Score</div>
                        </div>
                      </div>

                      <div className="quality-metrics">
                        <div className="metric">
                          <span className="metric-label">Size Quality</span>
                          <div className="metric-bar">
                            <div
                              className="metric-fill"
                              style={{
                                width: `${assessment.size_quality}%`,
                              }}
                            ></div>
                          </div>
                          <span className="metric-value">
                            {assessment.size_quality}%
                          </span>
                        </div>

                        <div className="metric">
                          <span className="metric-label">Color Quality</span>
                          <div className="metric-bar">
                            <div
                              className="metric-fill"
                              style={{
                                width: `${assessment.color_quality}%`,
                              }}
                            ></div>
                          </div>
                          <span className="metric-value">
                            {assessment.color_quality}%
                          </span>
                        </div>

                        <div className="metric">
                          <span className="metric-label">Shape Quality</span>
                          <div className="metric-bar">
                            <div
                              className="metric-fill"
                              style={{
                                width: `${assessment.shape_quality}%`,
                              }}
                            ></div>
                          </div>
                          <span className="metric-value">
                            {assessment.shape_quality}%
                          </span>
                        </div>

                        <div className="metric">
                          <span className="metric-label">Defects Detected</span>
                          <div className="metric-bar">
                            <div
                              className="metric-fill error"
                              style={{
                                width: `${assessment.defect_percentage}%`,
                              }}
                            ></div>
                          </div>
                          <span className="metric-value">
                            {assessment.defect_percentage}%
                          </span>
                        </div>
                      </div>

                      <div className="market-info">
                        <h3>Market Pricing</h3>
                        <p>
                          Price Adjustment: ×
                          <strong>{assessment.market_price_adjustment}</strong>
                        </p>
                        <p className="market-note">
                          Applies to base market price for {assessment.crop_type}
                        </p>
                      </div>

                      <div className="recommendations">
                        <h3>Quality Improvement Recommendations</h3>
                        <ul>
                          {assessment.recommendations.map((rec, i) => (
                            <li key={i}>💡 {rec}</li>
                          ))}
                        </ul>
                      </div>

                      <div className="confidence">
                        <span>Assessment Confidence: {assessment.confidence}%</span>
                      </div>
                    </>
                  )}
                </div>
              ))}
            </div>
          )}

          {trends && (
            <div className="trends-container">
              <h2>Quality Trends</h2>
              <div className="trends-stats">
                <div className="stat-card">
                  <span className="stat-label">Total Assessments</span>
                  <span className="stat-value">
                    {trends.assessments_count}
                  </span>
                </div>
                <div className="stat-card">
                  <span className="stat-label">Average Score</span>
                  <span className="stat-value">{trends.average_score}</span>
                </div>
                <div className="stat-card">
                  <span className="stat-label">Grade Distribution</span>
                  <div className="grade-dist">
                    {Object.entries(trends.grade_distribution).map(
                      ([grade, count]) => (
                        <div
                          key={grade}
                          style={{
                            color: getGradeColor(grade),
                            fontWeight: "bold",
                          }}
                        >
                          {grade}: {count}
                        </div>
                      )
                    )}
                  </div>
                </div>
              </div>

              {trends.score_trend && (
                <div className="score-trend">
                  <h4>Score Trend (Last 5)</h4>
                  <div className="trend-values">
                    {trends.score_trend.map((score, i) => (
                      <div key={i} className="trend-item">
                        <span className="trend-score">{score}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {assessments.length === 0 && !trends && (
            <div className="empty-state">
              <p>📸 Upload images to start crop quality assessment</p>
              <p className="empty-desc">
                Get instant grading, market price adjustments, and quality
                improvement recommendations
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default CropQualityGrading;
