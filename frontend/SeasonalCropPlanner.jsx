import React, { useState } from "react";
import { 
  Sprout, 
  MapPin, 
  Sun, 
  CloudRain, 
  Thermometer, 
  ArrowRight, 
  AlertTriangle, 
  CheckCircle2, 
  Calendar,
  Layers,
  Search,
  RefreshCw
} from "lucide-react";
import { getCropRecommendations, getYearlyCycle } from "./utils/cropPlanner";
import "./SeasonalCropPlanner.css";

const LOCATIONS = [
  "Maharashtra", "Punjab", "Gujarat", "Uttar Pradesh", "Haryana", 
  "Madhya Pradesh", "Rajasthan", "Karnataka", "Andhra Pradesh", 
  "Tamil Nadu", "West Bengal", "Bihar", "Telangana"
];

const SOIL_TYPES = ["Alluvial", "Black", "Red", "Laterite", "Sandy", "Clay", "Loamy"];

const SEASONS = [
  { id: "Kharif", label: "Kharif (Monsoon)", icon: <CloudRain size={18} /> },
  { id: "Rabi", label: "Rabi (Winter)", icon: <Sun size={18} /> },
  { id: "Zaid", label: "Zaid (Summer)", icon: <Thermometer size={18} /> }
];

export default function SeasonalCropPlanner() {
  const [formData, setFormData] = useState({
    location: "Maharashtra",
    season: "Kharif",
    soilType: "Black"
  });
  const [results, setResults] = useState(null);
  const [yearlyCycle, setYearlyCycle] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState("recommendations");

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handlePlan = (e) => {
    e.preventDefault();
    setLoading(true);
    
    // Simulate API call/processing
    setTimeout(() => {
      const recs = getCropRecommendations(formData.location, formData.season, formData.soilType);
      const cycle = getYearlyCycle(formData.location, formData.soilType);
      setResults(recs);
      setYearlyCycle(cycle);
      setLoading(false);
    }, 800);
  };

  return (
    <div className="crop-planner-page">
      <div className="planner-header">
        <div className="header-content">
          <h1><span className="notranslate">Seasonal Crop Planner</span></h1>
          <p>Optimize your yearly farming cycles and reduce crop failure risks with AI-driven insights.</p>
        </div>
      </div>

      <div className="planner-container">
        <div className="planner-sidebar">
          <div className="glass-card planning-form-card">
            <h3><Search size={20} /> Configure Plan</h3>
            <form onSubmit={handlePlan}>
              <div className="form-group">
                <label><MapPin size={16} /> Location (State)</label>
                <select name="location" value={formData.location} onChange={handleInputChange}>
                  {LOCATIONS.sort().map(loc => <option key={loc} value={loc}>{loc}</option>)}
                </select>
              </div>

              <div className="form-group">
                <label><Layers size={16} /> Soil Type</label>
                <select name="soilType" value={formData.soilType} onChange={handleInputChange}>
                  {SOIL_TYPES.map(soil => <option key={soil} value={soil}>{soil}</option>)}
                </select>
              </div>

              <div className="form-group">
                <label><Calendar size={16} /> Planning Season</label>
                <div className="season-toggle">
                  {SEASONS.map(s => (
                    <button 
                      key={s.id}
                      type="button"
                      className={`season-btn ${formData.season === s.id ? 'active' : ''}`}
                      onClick={() => setFormData(prev => ({ ...prev, season: s.id }))}
                    >
                      {s.icon}
                      <span>{s.id}</span>
                    </button>
                  ))}
                </div>
              </div>

              <button type="submit" className="plan-submit-btn" disabled={loading}>
                {loading ? <RefreshCw className="spin" /> : "Generate Plan"}
              </button>
            </form>
          </div>

           {results && (
             <div className="glass-card summary-card">
               <h4>Plan Summary</h4>
               <div className="summary-item">
                 <span>Recommended Crops</span>
                 <strong>{results.length}</strong>
               </div>
               <div className="summary-item">
                 <span>Risk Level</span>
                 <span className="risk-badge low">Low</span>
               </div>
               <div className="summary-item">
                 <span>Cycle Stability</span>
                 <span className="stability-badge high">High</span>
               </div>
             </div>
           )}
         </div>

        <div className="planner-main-content">
          {!results && !loading && (
            <div className="empty-state">
              <Sprout size={64} className="empty-icon" />
              <h2>Ready to plan your season?</h2>
              <p>Select your location, soil type, and target season to get personalized crop recommendations and a full yearly cycle plan.</p>
            </div>
          )}

          {loading && (
            <div className="loading-state">
              <div className="loader-ring"></div>
              <p>Analyzing soil conditions and weather patterns...</p>
            </div>
          )}

          {results && !loading && (
            <>
              <div className="tabs-navigation">
                <button 
                  className={`tab-link ${activeTab === 'recommendations' ? 'active' : ''}`}
                  onClick={() => setActiveTab('recommendations')}
                >
                  Current Recommendations
                </button>
                <button 
                  className={`tab-link ${activeTab === 'yearly' ? 'active' : ''}`}
                  onClick={() => setActiveTab('yearly')}
                >
                  Yearly Farming Cycle
                </button>
              </div>

               {activeTab === 'recommendations' ? (
                 <div className="recommendations-grid">
                   {results.length > 0 ? (
                     results.map((crop, idx) => (
                       <div className="crop-result-card glass-card" key={idx}>
                         <div className="crop-card-header">
                           <div className="crop-icon-wrapper">
                             <Sprout size={24} />
                           </div>
                           <div className="crop-title-group">
                             <h3>{crop.name}</h3>
                             <span className="crop-duration">{crop.duration}</span>
                           </div>
                         </div>
                         
                         <div className="crop-card-body">
                           <div className="info-section">
                             <label><CheckCircle2 size={14} /> Benefits</label>
                             <p>{crop.benefits}</p>
                           </div>
                           
                           <div className="info-section warning">
                             <label><AlertTriangle size={14} /> Risk Mitigation</label>
                             <p>{crop.riskMitigation}</p>
                           </div>
                         </div>

                        <div className="crop-card-footer">
                          <button className="details-btn">
                            Detailed Guide <ArrowRight size={16} />
                          </button>
                        </div>
                      </div>
                    ))
                  ) : (
                    <div className="no-results glass-card">
                      <AlertTriangle size={48} />
                      <h3>No direct matches found</h3>
                      <p>Try adjusting your soil type or exploring crops for adjacent regions.</p>
                    </div>
                  )}
                </div>
              ) : (
                <div className="yearly-cycle-view">
                  <div className="cycle-timeline">
                    {['Kharif', 'Rabi', 'Zaid'].map((seasonId) => {
                      const seasonCrops = yearlyCycle[seasonId.toLowerCase()];
                      return (
                        <div className="cycle-phase" key={seasonId}>
                          <div className="phase-marker">
                            <div className="marker-dot"></div>
                            <div className="marker-line"></div>
                          </div>
                          <div className="phase-content glass-card">
                            <div className="phase-header">
                              <h4>{seasonId} Season</h4>
                              <span className="phase-months">
                                {seasonId === 'Kharif' ? 'June - Oct' : seasonId === 'Rabi' ? 'Nov - Feb' : 'Mar - May'}
                              </span>
                            </div>
                            <div className="phase-crops">
                              {seasonCrops.length > 0 ? (
                                seasonCrops.map((c, i) => (
                                  <div key={i} className="mini-crop-badge">
                                    <Sprout size={14} />
                                    <span>{c.name}</span>
                                  </div>
                                ))
                              ) : (
                                <p className="no-crop-note">Fallow period or cover crops recommended</p>
                              )}
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                  <div className="cycle-footer-note glass-card">
                    <p><strong>Pro Tip:</strong> Following this crop rotation helps maintain soil nutrient balance and breaks pest cycles, significantly reducing the risk of total crop failure.</p>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
