import React, { useState } from "react";
import { 
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, ReferenceLine
} from "recharts";
import { Thermometer, Droplets, TrendingDown, TrendingUp, AlertTriangle, X } from "lucide-react";
import "./ClimateSimulator.css";

const SENSITIVITIES = {
  rice:      { temp: -0.05, rain: 0.02 },
  wheat:     { temp: -0.06, rain: 0.03 },
  cotton:    { temp: -0.03, rain: 0.01 },
  maize:     { temp: -0.07, rain: 0.04 },
  sugarcane: { temp: -0.02, rain: 0.05 },
  soybean:   { temp: -0.04, rain: 0.03 },
  potato:    { temp: -0.05, rain: 0.04 },
  default:   { temp: -0.04, rain: 0.02 },
};

function computeSimulation(cropType, tempDelta, rainDelta) {
  const coeff = SENSITIVITIES[cropType?.toLowerCase()] || SENSITIVITIES.default;
  const yieldImpactTemp = tempDelta * coeff.temp;
  const yieldImpactRain = (rainDelta / 100) * coeff.rain;
  const totalYieldImpact = yieldImpactTemp + yieldImpactRain;
  const profitImpact = totalYieldImpact * 1.5;
  const suitability = Math.max(0, Math.min(100, 85 + totalYieldImpact * 100));
  return {
    yield_impact_pct: parseFloat((totalYieldImpact * 100).toFixed(2)),
    profit_impact_pct: parseFloat((profitImpact * 100).toFixed(2)),
    suitability_score: parseFloat(suitability.toFixed(1)),
    risk_level: totalYieldImpact < -0.15 ? "High" : totalYieldImpact < -0.05 ? "Medium" : "Low",
    recommendation:
      tempDelta > 2
        ? "Switch to heat-tolerant varieties"
        : rainDelta < -20
        ? "Ensure adequate irrigation"
        : "Conditions remain viable",
  };
}

function formatImpact(value) {
  if (value > 0) return `+${value}%`;
  if (value < 0) return `${value}%`;
  return "No change";
}

const ClimateSimulator = ({ isOpen, onClose, userData }) => {
  const [tempDelta, setTempDelta] = useState(0);
  const [rainDelta, setRainDelta] = useState(0);

  const cropType = userData?.cropType || "rice";
  const result = computeSimulation(cropType, tempDelta, rainDelta);

  if (!isOpen) return null;

  // Build chart data as deviation from 100 — amplified so small changes are visible
  const simulatedYield  = parseFloat((100 + result.yield_impact_pct).toFixed(2));
  const simulatedProfit = parseFloat((100 + result.profit_impact_pct).toFixed(2));

  const chartData = [
    { name: "Baseline",  Yield: 100, Profit: 100 },
    { name: "Simulated", Yield: simulatedYield, Profit: simulatedProfit },
  ];

  // Dynamic Y-axis domain so changes are always visible
  const allVals = [100, simulatedYield, simulatedProfit];
  const minVal  = Math.min(...allVals) - 5;
  const maxVal  = Math.max(...allVals) + 5;

  const isNegYield  = result.yield_impact_pct < 0;
  const isNegProfit = result.profit_impact_pct < 0;

  return (
    <div className="simulator-overlay">
      <div className="simulator-modal">
        <div className="simulator-header">
          <h2>🌍 Climate Risk Simulator</h2>
          <button className="close-btn" onClick={onClose}><X /></button>
        </div>

        <div className="simulator-content">
          {/* ── Controls ── */}
          <div className="simulator-controls">
            <div className="control-group">
              <label>
                <Thermometer size={18} />
                Temperature Anomaly:&nbsp;
                <strong>{tempDelta > 0 ? "+" : ""}{tempDelta}°C</strong>
              </label>
              <input
                type="range"
                min="-5" max="5" step="0.5"
                value={tempDelta}
                onChange={(e) => setTempDelta(parseFloat(e.target.value))}
              />
              <div className="range-labels">
                <span>-5°C</span><span>Normal</span><span>+5°C</span>
              </div>
            </div>

            <div className="control-group">
              <label>
                <Droplets size={18} />
                Rainfall Change:&nbsp;
                <strong>{rainDelta > 0 ? "+" : ""}{rainDelta}%</strong>
              </label>
              <input
                type="range"
                min="-100" max="100" step="5"
                value={rainDelta}
                onChange={(e) => setRainDelta(parseFloat(e.target.value))}
              />
              <div className="range-labels">
                <span>Drought</span><span>Normal</span><span>Flood</span>
              </div>
            </div>

            <div className="simulator-status">
              <div className="status-item">
                <span className="label">Current Crop:</span>
                <span className="value">{cropType.toUpperCase()}</span>
              </div>
              <div className="status-item">
                <span className="label">Risk Level:</span>
                <span className={`value risk-${result.risk_level.toLowerCase()}`}>
                  {result.risk_level}
                </span>
              </div>
              <div className="status-item">
                <span className="label">Suitability Score:</span>
                <span className="value">{result.suitability_score}%</span>
              </div>
            </div>

            <div className="recommendation-box">
              <AlertTriangle size={20} />
              <p>{result.recommendation}</p>
            </div>
          </div>

          {/* ── Visualization ── */}
          <div className="simulator-viz">
            <h3>Yield &amp; Profit Impact (%)</h3>
            <div className="chart-container">
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={chartData} barCategoryGap="35%">
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="name" />
                  <YAxis domain={[minVal, maxVal]} tickFormatter={(v) => `${v}%`} />
                  <Tooltip formatter={(v) => `${v}%`} />
                  <Legend />
                  <ReferenceLine y={100} stroke="#64748b" strokeDasharray="4 4" label={{ value: "Baseline", position: "right", fontSize: 11 }} />
                  <Bar dataKey="Yield"  fill="#10b981" radius={[6,6,0,0]} />
                  <Bar dataKey="Profit" fill="#3b82f6" radius={[6,6,0,0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="impact-stats">
              <div className={`stat-card ${isNegYield ? "negative" : "positive"}`}>
                {isNegYield ? <TrendingDown size={28} /> : <TrendingUp size={28} />}
                <div>
                  <h4>Yield Impact</h4>
                  <p>{formatImpact(result.yield_impact_pct)}</p>
                </div>
              </div>
              <div className={`stat-card ${isNegProfit ? "negative" : "positive"}`}>
                {isNegProfit ? <TrendingDown size={28} /> : <TrendingUp size={28} />}
                <div>
                  <h4>Profit Impact</h4>
                  <p>{formatImpact(result.profit_impact_pct)}</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ClimateSimulator;
