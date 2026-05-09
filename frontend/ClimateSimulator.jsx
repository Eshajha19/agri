import React, { useState, useEffect } from "react";
import { 
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, ReferenceLine
} from "recharts";
import { Thermometer, Droplets, TrendingDown, TrendingUp, AlertTriangle, X, Loader2 } from "lucide-react";
import "./ClimateSimulator.css";

const ClimateSimulator = ({ isOpen, onClose, userData }) => {
  const [tempDelta, setTempDelta] = useState(0);
  const [rainDelta, setRainDelta] = useState(0);
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const cropType = userData?.cropType || "rice";

  useEffect(() => {
    if (!isOpen) return;

    const fetchSimulation = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const response = await fetch(`/api/simulate-climate`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            crop_type: cropType,
            temp_delta: tempDelta,
            rain_delta: rainDelta,
          }),
        });

        if (!response.ok) {
          throw new Error("Failed to fetch simulation results");
        }

        const data = await response.json();
        setResult(data);
      } catch (err) {
        console.error("Simulation error:", err);
        setError("Error connecting to simulation service.");
      } finally {
        setIsLoading(false);
      }
    };

    const timeoutId = setTimeout(fetchSimulation, 300); // Debounce API calls
    return () => clearTimeout(timeoutId);
  }, [isOpen, tempDelta, rainDelta, cropType]);

  if (!isOpen) return null;

  // Build chart data as deviation from 100 — amplified so small changes are visible
  const simulatedYield  = result ? parseFloat((100 + result.yield_impact_pct).toFixed(2)) : 100;
  const simulatedProfit = result ? parseFloat((100 + result.profit_impact_pct).toFixed(2)) : 100;

  const chartData = [
    { name: "Baseline",  Yield: 100, Profit: 100 },
    { name: "Simulated", Yield: simulatedYield, Profit: simulatedProfit },
  ];

  // Dynamic Y-axis domain so changes are always visible
  const allVals = [100, simulatedYield, simulatedProfit];
  const minVal  = Math.min(...allVals) - 5;
  const maxVal  = Math.max(...allVals) + 5;

  const isNegYield  = result?.yield_impact_pct < 0;
  const isNegProfit = result?.profit_impact_pct < 0;

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

            {error ? (
              <div className="error-box">
                <AlertTriangle size={20} />
                <p>{error}</p>
              </div>
            ) : result ? (
              <>
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
              </>
            ) : (
              <div className="loading-placeholder">
                <p>Calculating impacts...</p>
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
                  <p>{result ? formatImpact(result.yield_impact_pct) : "--"}</p>
                </div>
              </div>
              <div className={`stat-card ${isNegProfit ? "negative" : "positive"}`}>
                {isNegProfit ? <TrendingDown size={28} /> : <TrendingUp size={28} />}
                <div>
                  <h4>Profit Impact</h4>
                  <p>{result ? formatImpact(result.profit_impact_pct) : "--"}</p>
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

