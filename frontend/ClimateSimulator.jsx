import React, { useState, useEffect } from "react";
import {
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, ReferenceLine,
} from "recharts";
import {
  Thermometer, Droplets, TrendingDown, TrendingUp, AlertTriangle, X, Loader2,
} from "lucide-react";
import "./ClimateSimulator.css";
import apiClient from "./services/api";

// ---------------------------------------------------------------------------
// Derive display-friendly fields from the knowledge.py response shape:
//
//   impact.score  — projected yield as % of baseline (100 = no change,
//                   <100 = reduction, >100 = improvement, clamped 0–150)
//   recommendations[] — array of actionable strings
//   baseline / simulated — temperature and rainfall values
//
// We convert impact.score into yield_impact_pct (deviation from 100) so the
// existing chart and stat-card logic works without changes.
// ---------------------------------------------------------------------------
function deriveDisplayFields(data) {
  const score = data?.impact?.score ?? 100;
  const yieldImpactPct = parseFloat((score - 100).toFixed(2));
  // Profit impact is estimated as 1.5× yield impact (same heuristic as the
  // old platform.py endpoint used).
  const profitImpactPct = parseFloat((yieldImpactPct * 1.5).toFixed(2));

  let riskLevel = "Low";
  if (score < 60) riskLevel = "High";
  else if (score < 85) riskLevel = "Medium";

  const suitabilityScore = Math.min(100, Math.round(score));

  // Use the first recommendation as the primary advisory message.
  const recommendation =
    Array.isArray(data?.recommendations) && data.recommendations.length > 0
      ? data.recommendations[0]
      : "Conditions are within the acceptable range. Continue standard practices.";

  return { yieldImpactPct, profitImpactPct, riskLevel, suitabilityScore, recommendation };
}

const ClimateSimulator = ({ isOpen, onClose, userData }) => {
  const [tempDelta, setTempDelta]   = useState(0);
  const [rainDelta, setRainDelta]   = useState(0);
  const [result, setResult]         = useState(null);
  const [display, setDisplay]       = useState(null);
  const [isLoading, setIsLoading]   = useState(false);
  const [error, setError]           = useState(null);

  const cropType = userData?.cropType || "rice";

  useEffect(() => {
    if (!isOpen) return;

    const fetchSimulation = async () => {
      setIsLoading(true);
      setError(null);
      try {
        // Use apiClient so the Firebase auth token is automatically injected
        // via the Axios request interceptor. The /api/knowledge/simulate-climate
        // endpoint requires authentication and returns HTTP 401 for raw fetch
        // calls that omit the Authorization header.
        const res = await apiClient.post("/api/knowledge/simulate-climate", {
          crop_type: cropType,
          temp_delta: tempDelta,
          // knowledge.py rain_delta is in mm/month; the slider is ±100 mm/month
          rain_delta: rainDelta,
        });
        const data = res.data;
        setResult(data);
        setDisplay(deriveDisplayFields(data));
      } catch (err) {
        console.error("Simulation error:", err);
        const status = err?.response?.status;
        if (status === 401) {
          setError("Please log in to use the climate simulator.");
        } else if (status === 422) {
          setError("Invalid simulation parameters. Please adjust the sliders.");
        } else {
          setError("Error connecting to simulation service.");
        }
      } finally {
        setIsLoading(false);
      }
    };

    const timeoutId = setTimeout(fetchSimulation, 300); // debounce slider changes
    return () => clearTimeout(timeoutId);
  }, [isOpen, tempDelta, rainDelta, cropType]);

  if (!isOpen) return null;

  // Build chart data as deviation from 100 — amplified so small changes are visible
  const simulatedYield  = display ? parseFloat((100 + display.yieldImpactPct).toFixed(2))  : 100;
  const simulatedProfit = display ? parseFloat((100 + display.profitImpactPct).toFixed(2)) : 100;

  const chartData = [
    { name: "Baseline",  Yield: 100, Profit: 100 },
    { name: "Simulated", Yield: simulatedYield, Profit: simulatedProfit },
  ];

  const allVals = [100, simulatedYield, simulatedProfit];
  const minVal  = Math.min(...allVals) - 5;
  const maxVal  = Math.max(...allVals) + 5;

  const isNegYield  = display ? display.yieldImpactPct  < 0 : false;
  const isNegProfit = display ? display.profitImpactPct < 0 : false;

  const formatImpact = (value) => {
    if (value > 0) return `+${value}%`;
    if (value < 0) return `${value}%`;
    return "No change";
  };

  return (
    <div className="simulator-overlay">
      <div className="simulator-modal">
        <div className="simulator-header">
          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            <h2>🌍 Climate Risk Simulator</h2>
            {isLoading && <Loader2 className="animate-spin" size={18} color="#22c55e" />}
          </div>
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
                <strong>{rainDelta > 0 ? "+" : ""}{rainDelta} mm/month</strong>
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

            {error ? (
              <div className="error-box">
                <AlertTriangle size={20} />
                <p>{error}</p>
              </div>
            ) : display ? (
              <>
                <div className="simulator-status">
                  <div className="status-item">
                    <span className="label">Current Crop:</span>
                    <span className="value">{cropType.toUpperCase()}</span>
                  </div>
                  <div className="status-item">
                    <span className="label">Risk Level:</span>
                    <span className={`value risk-${display.riskLevel.toLowerCase()}`}>
                      {display.riskLevel}
                    </span>
                  </div>
                  <div className="status-item">
                    <span className="label">Suitability Score:</span>
                    <span className="value">{display.suitabilityScore}%</span>
                  </div>
                </div>

                <div className="recommendation-box">
                  <AlertTriangle size={20} />
                  <p>{display.recommendation}</p>
                </div>

                {/* Show all recommendations if more than one */}
                {Array.isArray(result?.recommendations) && result.recommendations.length > 1 && (
                  <ul className="recommendations-list">
                    {result.recommendations.slice(1).map((rec, i) => (
                      <li key={i}>{rec}</li>
                    ))}
                  </ul>
                )}
              </>
            ) : (
              <div className="loading-placeholder">
                <p>Calculating impacts…</p>
              </div>
            )}
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
                  <ReferenceLine
                    y={100}
                    stroke="#64748b"
                    strokeDasharray="4 4"
                    label={{ value: "Baseline", position: "right", fontSize: 11 }}
                  />
                  <Bar dataKey="Yield"  fill="#10b981" radius={[6, 6, 0, 0]} />
                  <Bar dataKey="Profit" fill="#3b82f6" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="impact-stats">
              <div className={`stat-card ${isNegYield ? "negative" : "positive"}`}>
                {isNegYield ? <TrendingDown size={28} /> : <TrendingUp size={28} />}
                <div>
                  <h4>Yield Impact</h4>
                  <p>{display ? formatImpact(display.yieldImpactPct) : "--"}</p>
                </div>
              </div>
              <div className={`stat-card ${isNegProfit ? "negative" : "positive"}`}>
                {isNegProfit ? <TrendingDown size={28} /> : <TrendingUp size={28} />}
                <div>
                  <h4>Profit Impact</h4>
                  <p>{display ? formatImpact(display.profitImpactPct) : "--"}</p>
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
