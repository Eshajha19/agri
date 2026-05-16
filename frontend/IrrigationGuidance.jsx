import React, { useState, useEffect } from 'react';
import {
  Droplets, Info, ThermometerSun, Leaf,
  Activity, ChevronRight, MapPin, X
} from 'lucide-react';
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
    growthStage: 'Vegetative',
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [tips, setTips] = useState([]);

  // Load saved data
  useEffect(() => {
    const saved = localStorage.getItem("irrigationData");
    if (saved) setFormData(JSON.parse(saved));
  }, []);

  // Save data
  useEffect(() => {
    localStorage.setItem("irrigationData", JSON.stringify(formData));
  }, [formData]);

  // OPTIONAL: Auto-fetch weather (replace API key)
  useEffect(() => {
    navigator.geolocation.getCurrentPosition(async (pos) => {
      try {
        const { latitude, longitude } = pos.coords;

        // Replace with real API
        const res = await fetch(
          `https://api.openweathermap.org/data/2.5/weather?lat=${latitude}&lon=${longitude}&units=metric&appid=YOUR_API_KEY`
        );
        const data = await res.json();

        setFormData(prev => ({
          ...prev,
          temperature: data.main?.temp || prev.temperature,
          humidity: data.main?.humidity || prev.humidity,
          rainfall: data.rain?.['1h'] || 0
        }));
      } catch (err) {
        console.log("Weather fetch failed");
      }
    });
  }, []);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const calculateIrrigation = () => {
    setLoading(true);

    setTimeout(() => {
      const baseWater = 4000;

      let cropFactor = {
        Rice: 1.8,
        Sugarcane: 1.5,
        Wheat: 1.0,
        Maize: 0.8,
        Cotton: 0.7
      }[formData.cropType] || 1;

      let soilFactor = {
        Sandy: 1.3,
        Loamy: 1.0,
        Clay: 0.8
      }[formData.soilType] || 1;

      const stageFactor = {
        Seedling: 0.7,
        Vegetative: 1.0,
        Flowering: 1.3,
        Harvest: 0.8
      }[formData.growthStage] || 1;

      const tempFactor =
        formData.temperature > 35 ? 1.2 :
        formData.temperature < 20 ? 0.8 : 1;

      const humidityFactor =
        formData.humidity > 70 ? 0.85 :
        formData.humidity < 30 ? 1.15 : 1;

      const phFactor =
        formData.soilPH < 6 ? 1.1 :
        formData.soilPH > 8 ? 1.05 : 1;

      const rainReduction = formData.rainfall * 100;

      let totalWaterReq =
        baseWater *
        cropFactor *
        soilFactor *
        stageFactor *
        tempFactor *
        humidityFactor *
        phFactor *
        formData.areaSize -
        rainReduction;

      if (totalWaterReq < 0) totalWaterReq = 0;

      // Schedule logic
      let days = 3;

      if (soilFactor > 1) days -= 1;
      if (soilFactor < 1) days += 2;

      if (formData.temperature > 35) days -= 1;
      if (formData.temperature < 20) days += 1;

      if (cropFactor > 1.2) days -= 1;
      if (cropFactor < 0.9) days += 1;

      if (formData.rainfall > 40) days += 4;
      else if (formData.rainfall > 15) days += 2;
      else if (formData.rainfall > 5) days += 1;

      if (days < 1) days = 1;

      let schedule =
        totalWaterReq === 0
          ? "Not needed currently"
          : days === 1
          ? "Every day"
          : `Every ${days} days`;

      // SMART TIPS
      let smartTips = [];

      if (formData.soilType === 'Sandy') {
        smartTips.push("Use drip irrigation to reduce water loss in sandy soil.");
      }

      if (formData.temperature > 35) {
        smartTips.push("Irrigate early morning or evening to reduce evaporation.");
      }

      if (formData.rainfall > 20) {
        smartTips.push("Recent rainfall reduces immediate irrigation needs.");
      }

      if (formData.growthStage === 'Flowering') {
        smartTips.push("Ensure consistent watering during flowering stage.");
      }

      if (smartTips.length === 0) {
        smartTips.push("Maintain regular soil moisture without overwatering.");
      }

      setTips(smartTips);

      setResult({
        waterNeeded: Math.round(totalWaterReq),
        schedule,
      });

      setLoading(false);
    }, 1200);
  };

  return (
    <div className="irri-container fade-in">
      <button className="irri-close-btn" onClick={onClose}><X size={24} /></button>

      <div className="irri-header">
        <Droplets size={36} color="#0ea5e9" />
        <h2>Smart Irrigation Planner</h2>
        <p>AI-driven water optimization system for modern farming</p>
      </div>

      <div className="irri-content">
        {!result ? (
          <form className="irri-form" onSubmit={(e) => { e.preventDefault(); calculateIrrigation(); }}>

            <div className="form-grid">

              <div className="form-group">
                <label><Leaf size={16}/> Crop Type</label>
                <select name="cropType" value={formData.cropType} onChange={handleChange}>
                  <option>Wheat</option>
                  <option>Rice</option>
                  <option>Maize</option>
                  <option>Sugarcane</option>
                  <option>Cotton</option>
                </select>
              </div>

              <div className="form-group">
                <label><MapPin size={16}/> Soil Type</label>
                <select name="soilType" value={formData.soilType} onChange={handleChange}>
                  <option>Loamy</option>
                  <option>Sandy</option>
                  <option>Clay</option>
                </select>
              </div>

              <div className="form-group">
                <label><Activity size={16}/> Growth Stage</label>
                <select name="growthStage" value={formData.growthStage} onChange={handleChange}>
                  <option>Seedling</option>
                  <option>Vegetative</option>
                  <option>Flowering</option>
                  <option>Harvest</option>
                </select>
              </div>

              <div className="form-group">
                <label><ThermometerSun size={16}/> Temperature</label>
                <input type="number" name="temperature" value={formData.temperature} onChange={handleChange}/>
              </div>

              <div className="form-group">
                <label><Droplets size={16}/> Humidity</label>
                <input type="number" name="humidity" value={formData.humidity} onChange={handleChange}/>
              </div>

              <div className="form-group">
                <label><Activity size={16}/> Soil pH</label>
                <input type="number" step="0.1" name="soilPH" value={formData.soilPH} onChange={handleChange}/>
              </div>

              <div className="form-group">
                <label><Droplets size={16}/> Rainfall</label>
                <input type="number" name="rainfall" value={formData.rainfall} onChange={handleChange}/>
              </div>

              <div className="form-group full-width">
                <label>Farm Area (Acres)</label>
                <input type="number" step="0.1" name="areaSize" value={formData.areaSize} onChange={handleChange}/>
              </div>

            </div>

            <button type="submit" className="irri-btn-primary" disabled={loading}>
              {loading ? "Analyzing..." : <>Calculate <ChevronRight size={18}/></>}
            </button>

          </form>
        ) : (
          <div className="irri-result-cards">

            <div className="result-card primary-result">
              <Droplets size={40} />
              <h3>{result.waterNeeded} L</h3>
              <p>Water Requirement</p>
            </div>

            <div className="result-card">
              <Activity size={30} />
              <h3>{result.schedule}</h3>
              <p>Irrigation Schedule</p>
            </div>

            <div className="result-card tip-card full-width">
              <Info size={24} />
              <div>
                <h4>Smart Farming Tips</h4>
                <ul>
                  {tips.map((t, i) => <li key={i}>{t}</li>)}
                </ul>
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