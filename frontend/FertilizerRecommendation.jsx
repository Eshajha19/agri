import React, { useState } from "react";
import { ChevronRight, FlaskConical, Info, Sprout, X } from "lucide-react";
import { generateFertilizerRecommendation } from "./utils/fertilizerRecommendation";

const INITIAL_FORM = {
  crop: "Wheat",
  soilType: "Loamy",
  soilPH: 6.8,
  nitrogen: "Medium",
  phosphorus: "Medium",
  potassium: "Medium",
  moisture: "Medium",
  acreage: 1,
  season: "Rabi",
};

export default function FertilizerRecommendation({ onClose }) {
  const [formData, setFormData] = useState(INITIAL_FORM);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (event) => {
    const { name, value } = event.target;
    setFormData((current) => ({
      ...current,
      [name]: name === "soilPH" || name === "acreage" ? Number(value) : value,
    }));
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    setLoading(true);

    window.setTimeout(() => {
      setResult(generateFertilizerRecommendation(formData));
      setLoading(false);
    }, 150);
  };

  return (
    <div className="yield-popup fertilizer-popup">
      <button className="close-btn" onClick={onClose} aria-label="Close fertilizer recommendation panel">
        <X size={24} />
      </button>

      <h2>
        <FlaskConical size={22} style={{ marginRight: "8px", verticalAlign: "-3px" }} />
        Fertilizer Recommendation
      </h2>

      {!result ? (
        <form className="yield-form" onSubmit={handleSubmit}>
          <div className="form-group">
            <label>
              <Sprout size={14} style={{ marginRight: "6px", verticalAlign: "-2px" }} />
              Crop
              <span className="tooltip-container">
                <Info className="tooltip-icon" size={14} />
                <span className="tooltip-text">Select the crop to tailor nutrient balance and product choices.</span>
              </span>
            </label>
            <select name="crop" value={formData.crop} onChange={handleChange}>
              <option value="Wheat">Wheat</option>
              <option value="Paddy">Paddy</option>
              <option value="Maize">Maize</option>
              <option value="Cotton">Cotton</option>
              <option value="Groundnut">Groundnut</option>
              <option value="Vegetables">Vegetables</option>
            </select>
          </div>

          <div className="form-group">
            <label>
              Soil Type
              <span className="tooltip-container">
                <Info className="tooltip-icon" size={14} />
                <span className="tooltip-text">Soil type influences retention, drainage, and fertilizer timing.</span>
              </span>
            </label>
            <select name="soilType" value={formData.soilType} onChange={handleChange}>
              <option value="Loamy">Loamy</option>
              <option value="Sandy">Sandy</option>
              <option value="Clay">Clay</option>
              <option value="Black">Black</option>
            </select>
          </div>

          <div className="form-group">
            <label>
              Soil pH
              <span className="tooltip-container">
                <Info className="tooltip-icon" size={14} />
                <span className="tooltip-text">Acidic or alkaline soil changes how strongly fertilizer is absorbed.</span>
              </span>
            </label>
            <input type="number" name="soilPH" step="0.1" min="0" max="14" value={formData.soilPH} onChange={handleChange} />
          </div>

          <div className="form-group">
            <label>
              Acreage
              <span className="tooltip-container">
                <Info className="tooltip-icon" size={14} />
                <span className="tooltip-text">The recommendation scales with the area you want to treat.</span>
              </span>
            </label>
            <input type="number" name="acreage" step="0.1" min="0.1" value={formData.acreage} onChange={handleChange} />
          </div>

          <div className="form-group">
            <label>Nitrogen Status</label>
            <select name="nitrogen" value={formData.nitrogen} onChange={handleChange}>
              <option value="Low">Low</option>
              <option value="Medium">Medium</option>
              <option value="High">High</option>
            </select>
          </div>

          <div className="form-group">
            <label>Phosphorus Status</label>
            <select name="phosphorus" value={formData.phosphorus} onChange={handleChange}>
              <option value="Low">Low</option>
              <option value="Medium">Medium</option>
              <option value="High">High</option>
            </select>
          </div>

          <div className="form-group">
            <label>Potassium Status</label>
            <select name="potassium" value={formData.potassium} onChange={handleChange}>
              <option value="Low">Low</option>
              <option value="Medium">Medium</option>
              <option value="High">High</option>
            </select>
          </div>

          <div className="form-group">
            <label>Soil Moisture</label>
            <select name="moisture" value={formData.moisture} onChange={handleChange}>
              <option value="Low">Low</option>
              <option value="Medium">Medium</option>
              <option value="High">High</option>
            </select>
          </div>

          <div className="form-group full-width">
            <label>Season</label>
            <select name="season" value={formData.season} onChange={handleChange}>
              <option value="Rabi">Rabi</option>
              <option value="Kharif">Kharif</option>
              <option value="Zaid">Zaid</option>
            </select>
          </div>

          <div className="form-group full-width form-actions">
            <button type="submit" className="action-btn" disabled={loading}>
              {loading ? "Analyzing..." : "Get Recommendation"}
              {!loading && <ChevronRight size={18} style={{ marginLeft: "8px" }} />}
            </button>
            <button type="button" className="action-btn secondary" onClick={onClose}>
              Cancel
            </button>
          </div>
        </form>
      ) : (
        <div>
          <p className="yield-result fertilizer-summary">{result.summary}</p>

          <div className="fertilizer-meta">
            <span className="fertilizer-chip">Priority: {result.priority}</span>
            <span className="fertilizer-chip">Crop: {result.crop}</span>
            <span className="fertilizer-chip">Area: {result.acreage} acre(s)</span>
          </div>

          <div className="fertilizer-grid">
            {result.products.length > 0 ? (
              result.products.map((item) => (
                <div className="fertilizer-card" key={`${item.nutrient}-${item.productName}`}>
                  <h4>{item.productName}</h4>
                  <p className="fertilizer-dose">{item.dose}</p>
                  <p>{item.reason}</p>
                  <small>{item.timing}</small>
                </div>
              ))
            ) : (
              <div className="fertilizer-card full-width">
                <h4>No heavy correction needed</h4>
                <p>Keep using compost or FYM as a maintenance dose and monitor the crop every 10-14 days.</p>
              </div>
            )}
          </div>

          <div className="fertilizer-note">
            <strong>Soil correction:</strong>
            <p>{result.phAdvice.join(" ") || "pH is acceptable, so no correction is required right now."}</p>
          </div>

          <div className="fertilizer-note">
            <strong>Organic boost:</strong>
            <p>{result.organicBoost}</p>
          </div>

          <div className="fertilizer-note">
            <strong>Soil and season notes:</strong>
            <p>{result.fieldNotes.join(" ")}</p>
          </div>

          <div className="fertilizer-note">
            <strong>Application focus:</strong>
            <p>{result.focus}</p>
          </div>

          <div className="form-actions fertilizer-actions">
            <button type="button" className="action-btn secondary" onClick={() => setResult(null)}>
              Recalculate
            </button>
            <button type="button" className="action-btn" onClick={onClose}>
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
