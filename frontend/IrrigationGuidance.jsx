import React, { useState } from 'react';
import { Droplets, Info, ThermometerSun, Leaf, Activity, ChevronRight, X, MapPin } from 'lucide-react';
import './IrrigationGuidance.css';

export default function IrrigationGuidance({ onClose }) {
  const [formData, setFormData] = useState({
    cropType: 'Wheat',
    soilType: 'Loamy',
    temperature: 30,
    humidity: 50,
    soilPH: 7.0,
    rainfall: 0,
    areaSize: 1,
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const calculateIrrigation = () => {
    setLoading(true);
    
    // Simulate API delay
    setTimeout(() => {
      // Mock ML Logic
      const baseWater = 4000; // liters per acre
      let cropFactor = 1;
      let soilFactor = 1;

      switch(formData.cropType) {
        case 'Rice': cropFactor = 1.8; break;
        case 'Sugarcane': cropFactor = 1.5; break;
        case 'Wheat': cropFactor = 1.0; break;
        case 'Maize': cropFactor = 0.8; break;
        case 'Cotton': cropFactor = 0.7; break;
        default: cropFactor = 1;
      }

      switch(formData.soilType) {
        case 'Sandy': soilFactor = 1.3; break; // Retains less water
        case 'Loamy': soilFactor = 1.0; break;
        case 'Clay': soilFactor = 0.8; break; // Retains more water
        default: soilFactor = 1;
      }

      const tempFactor = formData.temperature > 35 ? 1.2 : formData.temperature < 20 ? 0.8 : 1.0;
      const rainReduction = formData.rainfall > 0 ? (formData.rainfall * 100) : 0; 

      let totalWaterReq = (baseWater * cropFactor * soilFactor * tempFactor * formData.areaSize) - rainReduction;
      if (totalWaterReq < 0) totalWaterReq = 0;

      let days = 3; // base

      // Modify days based on soil
      if (formData.soilType === 'Sandy') days -= 1;
      if (formData.soilType === 'Clay') days += 2;

      // Modify days based on temp
      if (formData.temperature > 35) days -= 1;
      if (formData.temperature < 20) days += 1;

      // Modify days based on crop water needs
      if (cropFactor > 1.2) days -= 1; // thirsty crop
      if (cropFactor < 0.9) days += 1;

      // Handle rain
      if (formData.rainfall > 40) days += 4;
      else if (formData.rainfall > 15) days += 2;
      else if (formData.rainfall > 5) days += 1;

      // Boundaries
      if (days < 1) days = 1;

      let schedule = '';
      if (totalWaterReq === 0) {
        schedule = "Not needed currently";
      } else if (days === 1) {
        schedule = "Every day";
      } else {
        schedule = `Every ${days} days`;
      }

      setResult({
        waterNeeded: Math.round(totalWaterReq),
        schedule,
        efficiencyTip: formData.soilType === 'Sandy' 
          ? "Consider drip irrigation as sandy soils drain water fast." 
          : "Avoid overwatering to prevent root rot in clay-heavy soil."
      });
      setLoading(false);
    }, 1500);
  };

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  return (
    <div className="irri-container fade-in">
      <button className="irri-close-btn" onClick={onClose}>✕</button>
      <div className="irri-header">
        <div className="irri-header-icon">
          <Droplets size={36} color="#0ea5e9" />
        </div>
        <h2>Smart Irrigation Planner</h2>
        <p>AI-driven recommendations to optimize watering and reduce wastage.</p>
      </div>

      <div className="irri-content">
        {!result ? (
          <form className="irri-form" onSubmit={(e) => { e.preventDefault(); calculateIrrigation(); }}>
            <div className="form-grid">
              <div className="form-group">
                <label>
                  <Leaf size={16}/> Crop Type 
                  <span className="tooltip-container">
                    <Info className="tooltip-icon" size={14} />
                    <span className="tooltip-text">Select the type of crop you are currently growing.</span>
                  </span>
                </label>
                <select name="cropType" value={formData.cropType} onChange={handleChange}>
                  <option>Wheat</option>
                  <option>Rice</option>
                  <option>Maize</option>
                  <option>Sugarcane</option>
                  <option>Cotton</option>
                </select>
              </div>

              <div className="form-group">
                <label>
                  <MapPin size={16}/> Soil Type 
                  <span className="tooltip-container">
                    <Info className="tooltip-icon" size={14} />
                    <span className="tooltip-text">Select the primary soil type present in your farm area.</span>
                  </span>
                </label>
                <select name="soilType" value={formData.soilType} onChange={handleChange}>
                  <option>Loamy</option>
                  <option>Sandy</option>
                  <option>Clay</option>
                </select>
              </div>

              <div className="form-group">
                <label>
                  <ThermometerSun size={16}/> Avg Temperature (°C)
                  <span className="tooltip-container">
                    <Info className="tooltip-icon" size={14} />
                    <span className="tooltip-text">Average daily temperature in your region. Affects crop water evaporation.</span>
                  </span>
                </label>
                <input type="number" name="temperature" value={formData.temperature} onChange={handleChange} />
              </div>

              <div className="form-group">
                <label>
                  <Droplets size={16}/> Humidity (%)
                  <span className="tooltip-container">
                    <Info className="tooltip-icon" size={14} />
                    <span className="tooltip-text">Amount of moisture in the air. High humidity reduces crop water requirements.</span>
                  </span>
                </label>
                <input type="number" name="humidity" value={formData.humidity} onChange={handleChange} />
              </div>

              <div className="form-group">
                <label>
                  <Activity size={16}/> Soil pH (0–14)
                  <span className="tooltip-container">
                    <Info className="tooltip-icon" size={14} />
                    <span className="tooltip-text">Indicates acidity or alkalinity of soil. Value between 0 and 14.</span>
                  </span>
                </label>
                <input type="number" name="soilPH" value={formData.soilPH} step="0.1" onChange={handleChange} />
              </div>

              <div className="form-group">
                <label>
                  <Droplets size={16}/> Recent Rainfall (mm)
                  <span className="tooltip-container">
                    <Info className="tooltip-icon" size={14} />
                    <span className="tooltip-text">Amount of rain received over the last 7 days. Modifies irrigation calculations.</span>
                  </span>
                </label>
                <input type="number" name="rainfall" value={formData.rainfall} onChange={handleChange} />
              </div>

              <div className="form-group full-width">
                <label>
                  <Activity size={16}/> Farm Area (Acres)
                  <span className="tooltip-container">
                    <Info className="tooltip-icon" size={14} />
                    <span className="tooltip-text">Total size of the land to be irrigated.</span>
                  </span>
                </label>
                <input type="number" name="areaSize" value={formData.areaSize} step="0.1" onChange={handleChange} />
              </div>
            </div>

            <button type="submit" className="irri-btn-primary" disabled={loading}>
              {loading ? (
                <span className="irri-loader"></span>
              ) : (
                <>Calculate Needs <ChevronRight size={18} /></>
              )}
            </button>
          </form>
        ) : (
          <div className="irri-result-cards slide-up">
            <div className="result-card primary-result">
              <Droplets size={40} className="result-icon blue" />
              <h3>{result.waterNeeded} L</h3>
              <p>Estimated Water Quantity</p>
              <small>For {formData.areaSize} acres of {formData.cropType}</small>
            </div>
            
            <div className="result-card">
              <Activity size={32} className="result-icon green" />
              <h3>{result.schedule}</h3>
              <p>Recommended Schedule</p>
            </div>

            <div className="result-card tip-card full-width">
              <Info size={24} className="result-icon yellow" />
              <div className="tip-content">
                <h4>Pro Tip for Less Wastage</h4>
                <p>{result.efficiencyTip}</p>
              </div>
            </div>

            <button className="irri-btn-secondary" onClick={() => setResult(null)}>
              Recalculate
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
