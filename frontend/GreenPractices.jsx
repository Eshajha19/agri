import React, { useState, useMemo } from "react";
import { 
  Leaf, 
  TrendingUp, 
  FileText, 
  CheckCircle2, 
  Circle, 
  ShieldCheck, 
  Globe, 
  Zap,
  Droplet,
  Recycle,
  Layers,
  ThermometerSun,
  X
} from "lucide-react";
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from "recharts";
import { generateSustainabilityPDF } from "./utils/exportService";
import "./GreenPractices.css";

const PRACTICES = [
  { id: "zero_tillage", label: "Zero Tillage", impact: 20, description: "Prevents soil erosion and sequesters carbon by avoiding soil disturbance.", icon: <Layers size={20} /> },
  { id: "organic_compost", label: "Organic Composting", impact: 25, description: "Reduces methane emissions from waste and improves soil organic matter.", icon: <Recycle size={20} /> },
  { id: "crop_rotation", label: "Crop Rotation", impact: 15, description: "Maintains soil fertility and breaks pest/disease cycles naturally.", icon: <Leaf size={20} /> },
  { id: "water_conservation", label: "Water Conservation", impact: 20, description: "Uses drip or sprinkler systems to minimize water waste and energy use.", icon: <Droplet size={20} /> },
  { id: "reduced_chemicals", label: "Reduced Chemical Usage", impact: 20, description: "Minimizes synthetic fertilizer and pesticide runoff into ecosystems.", icon: <ThermometerSun size={20} /> }
];

export default function GreenPractices({ userProfile, onClose }) {
  const [activePractices, setActivePractices] = useState({});
  const [isExporting, setIsExporting] = useState(false);

  const togglePractice = (id) => {
    setActivePractices(prev => ({
      ...prev,
      [id]: !prev[id]
    }));
  };

  const totalScore = useMemo(() => {
    return PRACTICES.reduce((acc, curr) => {
      return acc + (activePractices[curr.id] ? curr.impact : 0);
    }, 0);
  }, [activePractices]);

  // Rough estimation: 1 score point = 0.05 mtCO2e per acre (simplified for demo)
  const carbonCredits = useMemo(() => {
    const area = userProfile?.landArea || 1;
    return (totalScore * 0.05 * area);
  }, [totalScore, userProfile]);

  const chartData = useMemo(() => [
    { name: "Sustainability Progress", value: totalScore },
    { name: "Remaining", value: 100 - totalScore }
  ], [totalScore]);

  const COLORS = ["#10b981", "#e2e8f0"];

  const handleExport = () => {
    setIsExporting(true);
    const data = {
      farmerName: userProfile?.name || "Farmer",
      practices: PRACTICES.map(p => ({
        practice: p.label,
        status: !!activePractices[p.id],
        impact: activePractices[p.id] ? p.impact : 0
      })),
      totalScore,
      carbonCredits,
      date: new Date().toLocaleDateString()
    };
    
    setTimeout(() => {
      generateSustainabilityPDF(data);
      setIsExporting(false);
    }, 1000);
  };

  return (
    <div className="green-practices">
      <button className="green-close-top" onClick={onClose} aria-label="Close">
        &times;
      </button>
      <div className="green-header">
        <div className="title-area">
          <Leaf className="leaf-icon" size={28} />
          <h2>Green Practices & Carbon Credit Tracker</h2>
        </div>
        <p>Document your sustainable farming methods to earn carbon credits and contribute to a greener planet.</p>
      </div>

      <div className="green-content">
        <div className="practices-grid">
          <div className="practices-list">
            <h3>Record Your Practices</h3>
            {PRACTICES.map(practice => (
              <div 
                key={practice.id} 
                className={`practice-item ${activePractices[practice.id] ? 'active' : ''}`}
                onClick={() => togglePractice(practice.id)}
              >
                <div className="practice-icon">
                  {practice.icon}
                </div>
                <div className="practice-info">
                  <div className="practice-label">
                    <span>{practice.label}</span>
                    <span className="impact-tag">+{practice.impact} pts</span>
                  </div>
                  <p>{practice.description}</p>
                </div>
                <div className="practice-check">
                  {activePractices[practice.id] ? <CheckCircle2 size={24} className="checked" /> : <Circle size={24} className="unchecked" />}
                </div>
              </div>
            ))}
          </div>

          <div className="metrics-panel">
            <div className="metric-card main-score">
              <h3>Sustainability Score</h3>
              <div className="score-viz">
                <ResponsiveContainer width="100%" height={180}>
                  <PieChart>
                    <Pie
                      data={chartData}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={80}
                      paddingAngle={5}
                      dataKey="value"
                      startAngle={180}
                      endAngle={0}
                    >
                      {chartData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
                <div className="score-text">
                  <span className="big-score">{totalScore}</span>
                  <span className="total">/100</span>
                </div>
              </div>
              <p className="score-level">
                {totalScore >= 80 ? "🌟 Platinum Earth Saver" : 
                 totalScore >= 50 ? "🌿 Gold Eco Farmer" : 
                 totalScore >= 20 ? "🌱 Silver Contributor" : "🍂 Getting Started"}
              </p>
            </div>

            <div className="metric-card carbon-impact">
              <div className="metric-header">
                <Globe size={20} />
                <span>Estimated Carbon Impact</span>
              </div>
              <div className="metric-value">
                <span className="value">{carbonCredits.toFixed(2)}</span>
                <span className="unit">mtCO2e</span>
              </div>
              <p>Potential carbon credits sequestered annually from your practices.</p>
            </div>

            <div className="metric-card monetization">
              <div className="metric-header">
                <Zap size={20} />
                <span>Monetization Estimate</span>
              </div>
              <div className="metric-value">
                <span className="value">₹{(carbonCredits * 1200).toLocaleString()}</span>
                <span className="unit">/ year*</span>
              </div>
              <p>*Based on current voluntary carbon market rates (~₹1200 per mtCO2e).</p>
            </div>
            
            <div className="actions">
              <button 
                className="export-btn" 
                onClick={handleExport}
                disabled={isExporting || totalScore === 0}
              >
                {isExporting ? <span className="loader-small"></span> : <FileText size={20} />}
                {isExporting ? "Generating..." : "Download Sustainability Report"}
              </button>
              <div className="certification-badge">
                <ShieldCheck size={16} />
                <span>Future-Ready Certification APIs Enabled</span>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <div className="green-footer">
        <button className="close-main-btn" onClick={onClose}>Finish Tracking</button>
      </div>
    </div>
  );
}
