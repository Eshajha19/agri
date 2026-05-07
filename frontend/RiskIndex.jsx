import React, { useState, useMemo } from "react";
import { 
  Radar, 
  RadarChart, 
  PolarGrid, 
  PolarAngleAxis, 
  PolarRadiusAxis, 
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell
} from "recharts";
import { 
  ShieldCheck, 
  AlertTriangle, 
  Info, 
  CloudRain, 
  Bug, 
  Droplets, 
  TrendingDown, 
  History,
  ArrowRight,
  ShieldAlert,
  HelpCircle
} from "lucide-react";
import "./RiskIndex.css";

const RiskIndex = () => {
  const [inputs, setInputs] = useState({
    weather: 45,
    disease: 30,
    water: 60, // 0 is best, 100 is critical shortage
    market: 50,
    historical: 25
  });

  const [activeTab, setActiveTab] = useState("analysis");

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setInputs(prev => ({ ...prev, [name]: parseInt(value) }));
  };

  const riskScore = useMemo(() => {
    const total = Object.values(inputs).reduce((acc, val) => acc + val, 0);
    return Math.round(total / 5);
  }, [inputs]);

  const riskData = [
    { subject: 'Weather', A: inputs.weather, fullMark: 100 },
    { subject: 'Disease', A: inputs.disease, fullMark: 100 },
    { subject: 'Water', A: inputs.water, fullMark: 100 },
    { subject: 'Market', A: inputs.market, fullMark: 100 },
    { subject: 'History', A: inputs.historical, fullMark: 100 },
  ];

  const getRiskStatus = (score) => {
    if (score < 30) return { label: "Low Risk", color: "#22c55e", icon: <ShieldCheck /> };
    if (score < 60) return { label: "Moderate Risk", color: "#eab308", icon: <Info /> };
    return { label: "High Risk", color: "#ef4444", icon: <AlertTriangle /> };
  };

  const status = getRiskStatus(riskScore);

  const recommendations = useMemo(() => {
    const recs = [];
    
    // Weather Recommendations
    if (inputs.weather > 70) {
      recs.push({ title: "Extreme Weather Shield", desc: "Emergency coverage for catastrophic heatwaves or flash floods.", type: "insurance" });
      recs.push({ title: "Crop Canopy Management", desc: "Immediate shading or mulching to protect against severe thermal stress.", type: "action" });
    } else if (inputs.weather > 35) {
      recs.push({ title: "Weather Indexed Plan", desc: "Protects your investment against moderate rainfall deficits.", type: "insurance" });
    }

    // Disease Recommendations
    if (inputs.disease > 70) {
      recs.push({ title: "Emergency Bio-Shield", desc: "Aggressive application of organic fungicides and isolation of affected zones.", type: "action" });
    } else if (inputs.disease > 35) {
      recs.push({ title: "IPM Monitoring", desc: "Install pheromone traps and increase scouting frequency to 3 times a week.", type: "action" });
    }

    // Water Recommendations
    if (inputs.water > 70) {
      recs.push({ title: "Drought-Resistant Shift", desc: "Transition to millet or sorghum for the next cycle to ensure survival.", type: "action" });
      recs.push({ title: "Water Credit Line", desc: "Secure emergency financing for private water tanker support.", type: "fintech" });
    } else if (inputs.water > 35) {
      recs.push({ title: "Precision Drip Upgrade", desc: "Implement sensor-based irrigation to reduce consumption by 30%.", type: "action" });
    }

    // Market Recommendations
    if (inputs.market > 70) {
      recs.push({ title: "Diversified Offtake", desc: "Splitting harvest between local Mandis and direct-to-consumer digital platforms.", type: "fintech" });
    } else if (inputs.market > 35) {
      recs.push({ title: "Forward Pricing", desc: "Lock in 40% of your expected yield at current market rates.", type: "fintech" });
    }

    // Default if very low risk
    if (recs.length === 0) {
      recs.push({ title: "Standard Multi-Peril", desc: "Comprehensive base coverage for standard environmental fluctuations.", type: "insurance" });
      recs.push({ title: "Yield Optimization", desc: "Focus on micro-nutrient application to exceed standard yield targets.", type: "action" });
    }

    // Limit to 4 most relevant recommendations to keep UI clean
    return recs.sort((a, b) => b.title.length - a.title.length).slice(0, 4);
  }, [inputs]);

  const highestRiskFactor = useMemo(() => {
    const sorted = Object.entries(inputs).sort((a, b) => b[1] - a[1]);
    return { name: sorted[0][0], value: sorted[0][1] };
  }, [inputs]);

  return (
    <div className="risk-container">
      <header className="risk-header">
        <div className="header-top">
          <span className="badge">AI-POWERED ANALYTICS</span>
          <h1>Farmer Risk Index</h1>
          <p>Advanced vulnerability assessment engine for precision agriculture and fintech planning.</p>
        </div>

        <div className="risk-summary-card">
          <div className="score-circle" style={{ borderColor: status.color }}>
            <span className="score-value">{riskScore}</span>
            <span className="score-label">INDEX</span>
          </div>
          <div className="summary-details">
            <h2 style={{ color: status.color }}>{status.label}</h2>
            <p>
              Based on current environmental and economic parameters, your farm's vulnerability is categorized as <strong>{status.label.toLowerCase()}</strong>. 
              {highestRiskFactor.value > 60 ? ` The primary driver of this score is elevated ${highestRiskFactor.name} risk.` : " All parameters are currently within manageable thresholds."}
            </p>
            <div className="status-icon" style={{ backgroundColor: `${status.color}22`, color: status.color }}>
              {status.icon}
            </div>
          </div>
        </div>
      </header>

      <div className="risk-content-grid">
        <aside className="risk-controls">
          <div className="control-section">
            <h3><CloudRain size={18} /> Weather Risk</h3>
            <input 
              type="range" name="weather" value={inputs.weather} 
              onChange={handleInputChange} min="0" max="100" 
              aria-label="Weather risk percentage"
            />
            <span className="val" aria-hidden="true">{inputs.weather}%</span>
          </div>
          
          <div className="control-section">
            <h3><Bug size={18} /> Disease Threat</h3>
            <input 
              type="range" name="disease" value={inputs.disease} 
              onChange={handleInputChange} min="0" max="100" 
              aria-label="Disease threat percentage"
            />
            <span className="val" aria-hidden="true">{inputs.disease}%</span>
          </div>

          <div className="control-section">
            <h3><Droplets size={18} /> Water Scarcity</h3>
            <input 
              type="range" name="water" value={inputs.water} 
              onChange={handleInputChange} min="0" max="100" 
              aria-label="Water scarcity percentage"
            />
            <span className="val" aria-hidden="true">{inputs.water}%</span>
          </div>

          <div className="control-section">
            <h3><TrendingDown size={18} /> Market Volatility</h3>
            <input 
              type="range" name="market" value={inputs.market} 
              onChange={handleInputChange} min="0" max="100" 
              aria-label="Market volatility percentage"
            />
            <span className="val" aria-hidden="true">{inputs.market}%</span>
          </div>

          <div className="control-section">
            <h3><History size={18} /> Historical Failure</h3>
            <input 
              type="range" name="historical" value={inputs.historical} 
              onChange={handleInputChange} min="0" max="100" 
              aria-label="Historical failure percentage"
            />
            <span className="val" aria-hidden="true">{inputs.historical}%</span>
          </div>
          
          <div className="info-box">
            <HelpCircle size={16} />
            <p>Adjust the sliders to simulate different environmental and economic scenarios for your region.</p>
          </div>
        </aside>

        <main className="risk-visualization">
          <div className="chart-container">
            <div className="chart-header">
              <h3>Vulnerability Distribution</h3>
              <p>Multi-factor risk analysis visualization</p>
            </div>
            <div 
               className="radar-wrapper"
               role="img"
               aria-label="Radar chart showing vulnerability distribution across weather, disease, water, market, and history."
             >
              <ResponsiveContainer width="100%" height={300}>
                <RadarChart cx="50%" cy="50%" outerRadius="80%" data={riskData}>
                  <PolarGrid stroke="#e2e8f0" />
                  <PolarAngleAxis dataKey="subject" tick={{ fill: '#64748b', fontSize: 12 }} />
                  <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                  <Radar
                    name="Risk"
                    dataKey="A"
                    stroke={status.color}
                    fill={status.color}
                    fillOpacity={0.4}
                  />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="recommendations-section">
            <div className="section-header">
              <ShieldAlert size={20} />
              <h3>Recommended Mitigation Strategies</h3>
            </div>
            <div className="recommendations-grid">
              {recommendations.map((rec, i) => (
                <div 
                  key={i} 
                  className={`rec-card ${rec.type}`}
                  role="article"
                  aria-label={`Mitigation Strategy: ${rec.title}. ${rec.desc}`}
                >
                  <div className="rec-type" aria-hidden="true">{rec.type}</div>
                  <h4>{rec.title}</h4>
                  <p>{rec.desc}</p>
                  <button 
                    className="rec-action"
                    aria-label={`Take action on ${rec.title}`}
                  >
                    Take Action <ArrowRight size={14} aria-hidden="true" />
                  </button>
                </div>
              ))}
            </div>
          </div>
        </main>
      </div>

      <footer className="risk-footer">
        <div className="disclaimer">
          <p><strong>Note:</strong> This risk index is generated by Fasal Saathi's analytical engine based on regional data trends. For official insurance quotes, please consult with registered financial providers.</p>
        </div>
      </footer>
    </div>
  );
};

export default RiskIndex;
