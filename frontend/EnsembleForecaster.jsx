import { useState, useEffect, useCallback } from "react";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

// ── colour tokens ─────────────────────────────────────────────────────────────
const MODEL_COLORS = {
  xgboost:       "#4ade80",
  lstm:          "#38bdf8",
  random_forest: "#fbbf24",
  ensemble:      "#a78bfa",
};

const Card = ({ children, style }) => (
  <div style={{
    background: "#111827", border: "1px solid #1f2937",
    borderRadius: 12, padding: "20px 24px", ...style,
  }}>
    {children}
  </div>
);

const SectionTitle = ({ children }) => (
  <h3 style={{
    margin: "0 0 14px", fontSize: 11, fontWeight: 700,
    letterSpacing: "0.12em", textTransform: "uppercase", color: "#6b7280",
  }}>
    {children}
  </h3>
);

const Spinner = () => (
  <span style={{
    display: "inline-block", width: 13, height: 13,
    border: "2px solid #374151", borderTopColor: "#4ade80",
    borderRadius: "50%", animation: "spin 0.7s linear infinite",
  }} />
);

const Stat = ({ label, value, color = "#e5e7eb", sub }) => (
  <Card>
    <div style={{ fontSize: 11, color: "#6b7280", marginBottom: 8, letterSpacing: "0.08em", textTransform: "uppercase" }}>
      {label}
    </div>
    <div style={{ fontSize: 22, fontWeight: 700, color }}>{value ?? "—"}</div>
    {sub && <div style={{ fontSize: 11, color: "#4b5563", marginTop: 4 }}>{sub}</div>}
  </Card>
);

// ── confidence band chart ─────────────────────────────────────────────────────
const ConfidenceBandChart = ({ prediction }) => {
  if (!prediction) return null;
  const { point_estimate, confidence_interval, model_predictions } = prediction;
  const lower = confidence_interval?.lower;
  const upper = confidence_interval?.upper;
  const models = model_predictions || {};

  const w = 600, h = 160, pad = 30;
  const allVals = [lower, point_estimate, upper, ...Object.values(models)].filter(v => v != null);
  const minV = Math.min(...allVals) * 0.95;
  const maxV = Math.max(...allVals) * 1.05;
  const range = maxV - minV || 1;

  const yFor = (v) => h - pad - ((v - minV) / range) * (h - pad * 2);
  const xCenter = w / 2;

  return (
    <Card style={{ marginTop: 16 }}>
      <SectionTitle>Prediction with Confidence Band</SectionTitle>
      <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`} style={{ overflow: "visible" }}>
        {/* Confidence band */}
        {lower != null && upper != null && (
          <rect
            x={xCenter - 60}
            y={yFor(upper)}
            width={120}
            height={yFor(lower) - yFor(upper)}
            fill="rgba(74,222,128,0.08)"
            stroke="#166534"
            strokeWidth="1"
            rx={6}
          />
        )}
        {/* Ensemble point */}
        <circle cx={xCenter} cy={yFor(point_estimate)} r={6} fill="#a78bfa" stroke="#e5e7eb" strokeWidth="2">
          <title>Ensemble: {point_estimate?.toFixed(2)}</title>
        </circle>
        {/* Model points */}
        {Object.entries(models).map(([name, val], i) => {
          const offset = (i - Object.keys(models).length / 2) * 35;
          return (
            <g key={name}>
              <line x1={xCenter + offset} y1={yFor(val)} x2={xCenter} y2={yFor(point_estimate)} stroke={MODEL_COLORS[name]} strokeWidth="1" strokeDasharray="3,3" opacity="0.5" />
              <circle cx={xCenter + offset} cy={yFor(val)} r={4} fill={MODEL_COLORS[name]} stroke="#111827" strokeWidth="1.5">
                <title>{name}: {val?.toFixed(2)}</title>
              </circle>
              <text x={xCenter + offset} y={yFor(val) - 12} fill={MODEL_COLORS[name]} fontSize="9" textAnchor="middle" fontFamily="monospace">
                {name.slice(0, 3).toUpperCase()}
              </text>
            </g>
          );
        })}
        {/* Labels */}
        <text x={pad} y={h - 4} fill="#6b7280" fontSize="10" fontFamily="monospace">{minV.toFixed(0)}</text>
        <text x={w - pad - 40} y={14} fill="#6b7280" fontSize="10" fontFamily="monospace">{maxV.toFixed(0)}</text>
        {lower != null && (
          <text x={xCenter + 70} y={yFor(lower) + 4} fill="#4ade80" fontSize="10" fontFamily="monospace">↓ {lower.toFixed(0)}</text>
        )}
        {upper != null && (
          <text x={xCenter + 70} y={yFor(upper) - 4} fill="#4ade80" fontSize="10" fontFamily="monospace">↑ {upper.toFixed(0)}</text>
        )}
      </svg>
      <div style={{ display: "flex", justifyContent: "center", gap: 16, marginTop: 8 }}>
        {Object.entries(models).map(([name, val]) => (
          <div key={name} style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 11, color: "#9ca3af" }}>
            <span style={{ width: 8, height: 8, borderRadius: "50%", background: MODEL_COLORS[name] }} />
            {name}: {val?.toFixed(2)}
          </div>
        ))}
        <div style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 11, color: "#e5e7eb", fontWeight: 700 }}>
          <span style={{ width: 8, height: 8, borderRadius: "50%", background: MODEL_COLORS.ensemble }} />
          Ensemble: {point_estimate?.toFixed(2)}
        </div>
      </div>
    </Card>
  );
};

// ── multi-step forecast chart ─────────────────────────────────────────────────
const MultiStepChart = ({ forecasts }) => {
  if (!forecasts || forecasts.length < 2) return null;
  const allPoints = forecasts.flatMap(f => [
    f.point_estimate,
    f.confidence_interval?.lower,
    f.confidence_interval?.upper,
  ]).filter(v => v != null);
  const minV = Math.min(...allPoints) * 0.95;
  const maxV = Math.max(...allPoints) * 1.05;
  const range = maxV - minV || 1;
  const w = 600, h = 180, pad = 30;

  const xFor = (i) => pad + (i / (forecasts.length - 1)) * (w - pad * 2);
  const yFor = (v) => h - pad - ((v - minV) / range) * (h - pad * 2);

  const pointPath = forecasts.map((f, i) => `${xFor(i)},${yFor(f.point_estimate)}`).join(" ");
  const upperPath = forecasts.map((f, i) => `${xFor(i)},${yFor(f.confidence_interval?.upper)}`).join(" ");
  const lowerPath = forecasts.map((f, i) => `${xFor(i)},${yFor(f.confidence_interval?.lower)}`).reverse().join(" ");

  return (
    <Card style={{ marginTop: 16 }}>
      <SectionTitle>Multi-Step Forecast</SectionTitle>
      <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`} style={{ overflow: "visible" }}>
        <polygon points={`${upperPath} ${lowerPath}`} fill="rgba(167,139,250,0.08)" stroke="none" />
        <polyline fill="none" stroke="#a78bfa" strokeWidth="2" points={pointPath} style={{ filter: "drop-shadow(0 0 4px rgba(167,139,250,0.3))" }} />
        {forecasts.map((f, i) => (
          <g key={i}>
            <circle cx={xFor(i)} cy={yFor(f.point_estimate)} r={5} fill="#a78bfa" stroke="#111827" strokeWidth="2" />
            <text x={xFor(i)} y={yFor(f.point_estimate) - 14} fill="#e5e7eb" fontSize="10" textAnchor="middle" fontFamily="monospace" fontWeight="700">
              {f.point_estimate?.toFixed(0)}
            </text>
            <text x={xFor(i)} y={h - 8} fill="#6b7280" fontSize="10" textAnchor="middle" fontFamily="monospace">
              Step {f.step}
            </text>
          </g>
        ))}
        <text x={pad} y={h - 4} fill="#6b7280" fontSize="10" fontFamily="monospace">{minV.toFixed(0)}</text>
        <text x={w - pad - 40} y={14} fill="#6b7280" fontSize="10" fontFamily="monospace">{maxV.toFixed(0)}</text>
      </svg>
    </Card>
  );
};

// ── disagreement alert ────────────────────────────────────────────────────────
const DisagreementAlert = ({ prediction }) => {
  if (!prediction?.disagreement?.high_disagreement) return null;
  return (
    <Card style={{ marginTop: 16, borderColor: "#7a4f00", background: "#2a1f0a" }}>
      <SectionTitle>⚠ Model Disagreement Alert</SectionTitle>
      <p style={{ margin: 0, fontSize: 13, color: "#fbbf24" }}>
        The three models show significant disagreement (CV: {prediction.disagreement.coefficient_of_variation}).
        This prediction has low confidence — consider additional ground-truth validation before making decisions.
      </p>
    </Card>
  );
};

// ── main component ────────────────────────────────────────────────────────────
export default function EnsembleForecaster() {
  const [weights, setWeights]         = useState(null);
  const [prediction, setPrediction]   = useState(null);
  const [forecasts, setForecasts]     = useState([]);
  const [activeTab, setActiveTab]     = useState("forecast");
  const [loading, setLoading]         = useState(true);
  const [refreshing, setRefreshing]   = useState(false);
  const [error, setError]             = useState(null);

  // form state
  const [inputJson, setInputJson]     = useState(JSON.stringify({
    Crop: "Wheat", CropCoveredArea: 2.5, CHeight: 120, CNext: "Rice", CLast: "Maize",
    CTransp: "High", IrriType: "Drip", IrriSource: "Groundwater", IrriCount: 3,
    WaterCov: 85, Season: "Rabi",
    lag_1: 2400, lag_2: 2350, lag_3: 2300, lag_4: 2280, lag_5: 2200,
  }, null, 2));
  const [steps, setSteps]             = useState(3);
  const [predicting, setPredicting]   = useState(false);
  const [predictError, setPredictError] = useState(null);

  const fetchWeights = useCallback(async () => {
    try {
      const r = await fetch(`${API_BASE}/api/ensemble/weights`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const d = await r.json();
      setWeights(d);
    } catch (e) {
      setError(e.message);
    }
  }, []);

  const loadAll = useCallback(async () => {
    setLoading(true);
    setError(null);
    await fetchWeights();
    setLoading(false);
  }, [fetchWeights]);

  useEffect(() => { loadAll(); }, [loadAll]);

  const handlePredict = async () => {
    setPredicting(true);
    setPredictError(null);
    setPrediction(null);
    setForecasts([]);
    try {
      let inputData;
      try {
        inputData = JSON.parse(inputJson);
      } catch (e) {
        throw new Error("Invalid JSON in input field");
      }

      const r = await fetch(`${API_BASE}/api/ensemble/forecast`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(inputData),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail || `HTTP ${r.status}`);
      setPrediction(d.prediction);
    } catch (e) {
      setPredictError(e.message);
    } finally {
      setPredicting(false);
    }
  };

  const handleMultiStep = async () => {
    setPredicting(true);
    setPredictError(null);
    setPrediction(null);
    setForecasts([]);
    try {
      let inputData;
      try {
        inputData = JSON.parse(inputJson);
      } catch (e) {
        throw new Error("Invalid JSON in input field");
      }

      const r = await fetch(`${API_BASE}/api/ensemble/multi-step?steps=${steps}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(inputData),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail || `HTTP ${r.status}`);
      if (d.task_id) {
        // Poll for result (simplified: just show task queued)
        setPredictError(null);
        setPrediction({ task_id: d.task_id, message: d.message });
      } else {
        setForecasts(d.forecast || []);
      }
    } catch (e) {
      setPredictError(e.message);
    } finally {
      setPredicting(false);
    }
  };

  return (
    <div style={{
      minHeight: "100vh", background: "#030712",
      color: "#e5e7eb", fontFamily: "'IBM Plex Mono', 'Courier New', monospace",
      padding: "32px 24px",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;700&display=swap');
        @keyframes spin    { to { transform: rotate(360deg); } }
        @keyframes fadeIn  { from { opacity:0; transform:translateY(5px); } to { opacity:1; transform:none; } }
        .tab-btn { background:none; border:none; cursor:pointer; padding:8px 18px;
          border-radius:8px; font-family:inherit; font-size:13px; color:#9ca3af; transition:all 0.15s; }
        .tab-btn:hover { background:#1f2937; }
        .tab-btn.active { background:#1f2937; color:#4ade80; }
        .action-btn { border:none; padding:10px 24px; border-radius:8px; cursor:pointer;
          font-family:inherit; font-size:13px; font-weight:700; transition:all 0.15s; }
        .action-btn:disabled { opacity:0.45; cursor:not-allowed; }
      `}</style>

      <div style={{ maxWidth: 960, margin: "0 auto" }}>
        {/* Header */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 28 }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
              <span style={{ fontSize: 20, color: "#4ade80" }}>◈</span>
              <h1 style={{ margin: 0, fontSize: 20, fontWeight: 700, color: "#f9fafb" }}>
                Ensemble Forecaster
              </h1>
            </div>
            <p style={{ margin: 0, fontSize: 12, color: "#6b7280" }}>
              Stacked XGBoost + LSTM + Random Forest with confidence intervals
            </p>
          </div>
          <button
            onClick={() => { setRefreshing(true); fetchWeights().then(() => setRefreshing(false)); }}
            disabled={refreshing}
            style={{
              display: "flex", alignItems: "center", gap: 7,
              background: "#111827", border: "1px solid #1f2937",
              color: "#9ca3af", padding: "8px 16px", borderRadius: 8,
              cursor: "pointer", fontFamily: "inherit", fontSize: 12,
            }}
          >
            {refreshing ? <Spinner /> : "↻"} Refresh
          </button>
        </div>

        {/* Tabs */}
        <div style={{ display: "flex", gap: 4, marginBottom: 24, borderBottom: "1px solid #1f2937", paddingBottom: 8 }}>
          {["forecast", "weights"].map(t => (
            <button key={t} className={`tab-btn${activeTab === t ? " active" : ""}`} onClick={() => setActiveTab(t)}>
              {{ forecast: "Forecast", weights: "Model Weights" }[t]}
            </button>
          ))}
        </div>

        {loading && (
          <div style={{ textAlign: "center", padding: 60, color: "#6b7280" }}>
            <Spinner /> <span style={{ marginLeft: 10 }}>Loading…</span>
          </div>
        )}
        {error && !loading && (
          <Card style={{ borderColor: "#7a1a1a", background: "#2a0a0a" }}>
            <p style={{ margin: 0, color: "#f87171", fontSize: 13 }}>⚠ Could not reach backend: {error}</p>
          </Card>
        )}

        {!loading && !error && weights && (
          <>
            {/* ── FORECAST TAB ── */}
            {activeTab === "forecast" && (
              <div style={{ animation: "fadeIn 0.25s ease" }}>
                {/* Input */}
                <Card style={{ marginBottom: 16 }}>
                  <SectionTitle>Input Features (JSON)</SectionTitle>
                  <textarea
                    value={inputJson}
                    onChange={e => setInputJson(e.target.value)}
                    rows={8}
                    style={{
                      width: "100%", boxSizing: "border-box",
                      background: "#0d1117", border: "1px solid #1f2937",
                      borderRadius: 8, color: "#e5e7eb", fontFamily: "inherit",
                      fontSize: 12, padding: "12px", resize: "vertical",
                    }}
                  />
                  <div style={{ display: "flex", gap: 10, marginTop: 12 }}>
                    <button
                      className="action-btn"
                      onClick={handlePredict}
                      disabled={predicting}
                      style={{ background: "#14532d", border: "1px solid #166534", color: "#4ade80" }}
                    >
                      {predicting ? <><Spinner /> &nbsp;Predicting…</> : "▶ Single Forecast"}
                    </button>
                    <button
                      className="action-btn"
                      onClick={handleMultiStep}
                      disabled={predicting}
                      style={{ background: "#1e3a5f", border: "1px solid #1a4a7a", color: "#7dd3fc" }}
                    >
                      {predicting ? <><Spinner /> &nbsp;Forecasting…</> : `▶ ${steps}-Step Forecast`}
                    </button>
                    <input
                      type="number"
                      min={1}
                      max={5}
                      value={steps}
                      onChange={e => setSteps(Math.min(5, Math.max(1, parseInt(e.target.value) || 1)))}
                      style={{
                        width: 50, background: "#0d1117", border: "1px solid #1f2937",
                        borderRadius: 8, color: "#e5e7eb", fontFamily: "inherit",
                        fontSize: 13, textAlign: "center", padding: "8px",
                      }}
                    />
                  </div>
                  {predictError && (
                    <p style={{ margin: "12px 0 0", color: "#f87171", fontSize: 12 }}>⚠ {predictError}</p>
                  )}
                </Card>

                {/* Results */}
                {prediction && !prediction.task_id && (
                  <>
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(170px,1fr))", gap: 12, marginBottom: 16 }}>
                      <Stat label="Point Estimate" value={prediction.point_estimate} color="#a78bfa" />
                      <Stat label="Lower Bound" value={prediction.confidence_interval?.lower} color="#4ade80" sub="90% CI" />
                      <Stat label="Upper Bound" value={prediction.confidence_interval?.upper} color="#4ade80" sub="90% CI" />
                      <Stat label="Models Used" value={prediction.models_used?.length} color="#fbbf24" sub={prediction.models_used?.join(", ")} />
                    </div>
                    <ConfidenceBandChart prediction={prediction} />
                    <DisagreementAlert prediction={prediction} />
                  </>
                )}
                {prediction?.task_id && (
                  <Card style={{ borderColor: "#1a4a7a", background: "#0a1a2a" }}>
                    <SectionTitle>Async Task Queued</SectionTitle>
                    <p style={{ margin: 0, fontSize: 13, color: "#7dd3fc" }}>
                      Task ID: <span style={{ color: "#e5e7eb" }}>{prediction.task_id}</span>
                    </p>
                    <p style={{ margin: "8px 0 0", fontSize: 11, color: "#6b7280" }}>
                      {prediction.message}
                    </p>
                  </Card>
                )}
                {forecasts.length > 0 && <MultiStepChart forecasts={forecasts} />}
              </div>
            )}

            {/* ── WEIGHTS TAB ── */}
            {activeTab === "weights" && (
              <div style={{ animation: "fadeIn 0.25s ease" }}>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(170px,1fr))", gap: 12, marginBottom: 16 }}>
                  <Stat label="XGBoost Weight" value={weights.weights?.xgboost?.toFixed(3)} color={MODEL_COLORS.xgboost} />
                  <Stat label="LSTM Weight" value={weights.weights?.lstm?.toFixed(3)} color={MODEL_COLORS.lstm} />
                  <Stat label="RF Weight" value={weights.weights?.random_forest?.toFixed(3)} color={MODEL_COLORS.random_forest} />
                  <Stat label="XGBoost Loaded" value={weights.models_loaded?.xgboost ? "Yes" : "No"} color={weights.models_loaded?.xgboost ? "#4ade80" : "#f87171"} />
                  <Stat label="LSTM Loaded" value={weights.models_loaded?.lstm ? "Yes" : "No"} color={weights.models_loaded?.lstm ? "#4ade80" : "#f87171"} />
                  <Stat label="RF Loaded" value={weights.models_loaded?.random_forest ? "Yes" : "No"} color={weights.models_loaded?.random_forest ? "#4ade80" : "#f87171"} />
                </div>
                <Card>
                  <SectionTitle>Weight Distribution</SectionTitle>
                  <div style={{ display: "flex", alignItems: "center", gap: 0, height: 32, borderRadius: 8, overflow: "hidden", marginTop: 8 }}>
                    {Object.entries(weights.weights || {}).map(([name, w]) => (
                      <div
                        key={name}
                        style={{
                          width: `${(w || 0) * 100}%`,
                          height: "100%",
                          background: MODEL_COLORS[name],
                          display: "flex", alignItems: "center", justifyContent: "center",
                          fontSize: 11, fontWeight: 700, color: "#030712",
                          minWidth: w > 0.05 ? undefined : 0,
                        }}
                        title={`${name}: ${(w * 100).toFixed(1)}%`}
                      >
                        {w > 0.08 ? `${(w * 100).toFixed(0)}%` : ""}
                      </div>
                    ))}
                  </div>
                  <div style={{ display: "flex", gap: 16, marginTop: 10 }}>
                    {Object.entries(weights.weights || {}).map(([name, w]) => (
                      <div key={name} style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 11, color: "#9ca3af" }}>
                        <span style={{ width: 8, height: 8, borderRadius: "50%", background: MODEL_COLORS[name] }} />
                        {name}: {w?.toFixed(3)}
                      </div>
                    ))}
                  </div>
                </Card>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}