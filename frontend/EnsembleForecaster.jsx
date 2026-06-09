  import { useState, useEffect, useCallback, memo, useTransition, useDeferredValue } from "react";
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Scatter, ScatterChart, ZAxis,
  ComposedChart, Bar, Line,
} from "recharts";

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

// ── confidence band chart (recharts, memoized) ────────────────────────────────
const ConfidenceBandChart = memo(function ConfidenceBandChart({ prediction }) {
  if (!prediction) return null;
  const { point_estimate, confidence_interval, model_predictions } = prediction;
  const lower = confidence_interval?.lower;
  const upper = confidence_interval?.upper;
  const models = model_predictions || {};

  // Build data array for recharts: one entry per model + ensemble
  const data = [
    { name: "ensemble", value: point_estimate, color: MODEL_COLORS.ensemble, z: 200 },
    ...Object.entries(models).map(([name, val]) => ({
      name,
      value: val,
      color: MODEL_COLORS[name] || "#9ca3af",
      z: 150,
    })),
  ];

  const allVals = [lower, point_estimate, upper, ...Object.values(models)].filter(v => v != null);
  const minV = Math.min(...allVals) * 0.95;
  const maxV = Math.max(...allVals) * 1.05;

  return (
    <Card style={{ marginTop: 16 }}>
      <SectionTitle>Prediction with Confidence Band</SectionTitle>
      <ResponsiveContainer width="100%" height={160}>
        <ComposedChart data={data} margin={{ top: 10, right: 30, bottom: 10, left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <YAxis domain={[minV, maxV]} tick={{ fill: "#6b7280", fontSize: 10, fontFamily: "monospace" }} axisLine={{ stroke: "#374151" }} />
          <Tooltip
            contentStyle={{ background: "#111827", border: "1px solid #1f2937", borderRadius: 8, color: "#e5e7eb", fontSize: 12 }}
            itemStyle={{ color: "#e5e7eb" }}
            formatter={(value, name, props) => [value?.toFixed(2), props.payload.name]}
          />
          {/* Confidence band as reference area */}
          {lower != null && upper != null && (
            <ReferenceLine y={lower} stroke="#166534" strokeDasharray="3 3" label={{ value: `↓ ${lower.toFixed(0)}`, fill: "#4ade80", fontSize: 10, position: "insideBottomRight" }} />
          )}
          {upper != null && lower != null && (
            <ReferenceLine y={upper} stroke="#166534" strokeDasharray="3 3" label={{ value: `↑ ${upper.toFixed(0)}`, fill: "#4ade80", fontSize: 10, position: "insideTopRight" }} />
          )}
          {/* Model points */}
          <Scatter dataKey="value" fill="#8884d8">
            {data.map((entry, index) => (
              <cell key={`cell-${index}`} fill={entry.color} />
            ))}
          </Scatter>
        </ComposedChart>
      </ResponsiveContainer>
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
});

// ── multi-step forecast chart (recharts, memoized) ────────────────────────────
const MultiStepChart = memo(function MultiStepChart({ forecasts }) {
  if (!forecasts || forecasts.length < 2) return null;

  const data = forecasts.map((f, i) => ({
    step: `Step ${f.step}`,
    point: f.point_estimate,
    upper: f.confidence_interval?.upper,
    lower: f.confidence_interval?.lower,
  }));

  const allPoints = forecasts.flatMap(f => [
    f.point_estimate,
    f.confidence_interval?.lower,
    f.confidence_interval?.upper,
  ]).filter(v => v != null);
  const minV = Math.min(...allPoints) * 0.95;
  const maxV = Math.max(...allPoints) * 1.05;

  return (
    <Card style={{ marginTop: 16 }}>
      <SectionTitle>Multi-Step Forecast</SectionTitle>
      <ResponsiveContainer width="100%" height={180}>
        <ComposedChart data={data} margin={{ top: 10, right: 30, bottom: 20, left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis dataKey="step" tick={{ fill: "#6b7280", fontSize: 10, fontFamily: "monospace" }} axisLine={{ stroke: "#374151" }} />
          <YAxis domain={[minV, maxV]} tick={{ fill: "#6b7280", fontSize: 10, fontFamily: "monospace" }} axisLine={{ stroke: "#374151" }} />
          <Tooltip
            contentStyle={{ background: "#111827", border: "1px solid #1f2937", borderRadius: 8, color: "#e5e7eb", fontSize: 12 }}
            formatter={(value) => [value?.toFixed(2), "Yield"]}
          />
          {/* Confidence band */}
          <Area type="monotone" dataKey="upper" stroke="none" fill="rgba(167,139,250,0.08)" />
          <Area type="monotone" dataKey="lower" stroke="none" fill="#030712" />
          {/* Main line */}
          <Line type="monotone" dataKey="point" stroke="#a78bfa" strokeWidth={2} dot={{ r: 5, fill: "#a78bfa", stroke: "#111827", strokeWidth: 2 }} activeDot={{ r: 7, fill: "#c4b5fd" }} />
        </ComposedChart>
      </ResponsiveContainer>
    </Card>
  );
});

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
  const [isPending, startTransition]  = useTransition();
  const deferredForecasts             = useDeferredValue(forecasts);
  const deferredPrediction            = useDeferredValue(prediction);
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
      startTransition(() => {
        setPrediction(d.prediction);
      });
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
        startTransition(() => {
          setForecasts(d.forecast || []);
        });
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
                      disabled={predicting || isPending}
                      style={{ background: "#14532d", border: "1px solid #166534", color: "#4ade80" }}
                    >
                      {predicting || isPending ? <><Spinner /> &nbsp;Predicting…</> : "▶ Single Forecast"}
                    </button>
                    <button
                      className="action-btn"
                      onClick={handleMultiStep}
                      disabled={predicting || isPending}
                      style={{ background: "#1e3a5f", border: "1px solid #1a4a7a", color: "#7dd3fc" }}
                    >
                      {predicting || isPending ? <><Spinner /> &nbsp;Forecasting…</> : `▶ ${steps}-Step Forecast`}
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
                {deferredPrediction && !deferredPrediction.task_id && (
                  <>
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(170px,1fr))", gap: 12, marginBottom: 16 }}>
                      <Stat label="Point Estimate" value={deferredPrediction.point_estimate} color="#a78bfa" />
                      <Stat label="Lower Bound" value={deferredPrediction.confidence_interval?.lower} color="#4ade80" sub="90% CI" />
                      <Stat label="Upper Bound" value={deferredPrediction.confidence_interval?.upper} color="#4ade80" sub="90% CI" />
                      <Stat label="Models Used" value={deferredPrediction.models_used?.length} color="#fbbf24" sub={deferredPrediction.models_used?.join(", ")} />
                    </div>
                    <ConfidenceBandChart prediction={deferredPrediction} />
                    <DisagreementAlert prediction={deferredPrediction} />
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
                {deferredForecasts.length > 0 && <MultiStepChart forecasts={deferredForecasts} />}
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