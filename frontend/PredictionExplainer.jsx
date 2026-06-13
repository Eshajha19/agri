import { useState, useEffect, useCallback } from "react";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

// ── colour tokens ─────────────────────────────────────────────────────────────
const DIR_COLORS = {
  positive: { bar: "#22c55e", bg: "#0f2a1a", border: "#1a5c30", text: "#4ade80" },
  negative: { bar: "#ef4444", bg: "#2a0a0a", border: "#7a1a1a", text: "#f87171" },
  neutral:  { bar: "#6b7280", bg: "#111827", border: "#1f2937", text: "#9ca3af" },
};
const MAG_OPACITY = { high: 1, medium: 0.72, low: 0.45 };

// ── tiny shared components ────────────────────────────────────────────────────
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

const Field = ({ label, children }) => (
  <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
    <label style={{ fontSize: 11, color: "#6b7280", letterSpacing: "0.08em", textTransform: "uppercase" }}>
      {label}
    </label>
    {children}
  </div>
);

const inputStyle = {
  background: "#0d1117", border: "1px solid #1f2937", borderRadius: 8,
  color: "#e5e7eb", fontFamily: "inherit", fontSize: 13, padding: "9px 12px",
  width: "100%", boxSizing: "border-box",
};

// ── SHAP waterfall bar ────────────────────────────────────────────────────────
const ShapBar = ({ feature, shap_value, direction, magnitude, explanation, maxAbs }) => {
  const c       = DIR_COLORS[direction] || DIR_COLORS.neutral;
  const opacity = MAG_OPACITY[magnitude] || 1;
  const pct     = maxAbs > 0 ? Math.min((Math.abs(shap_value) / maxAbs) * 100, 100) : 0;
  const isPos   = direction === "positive";

  return (
    <div style={{
      background: c.bg, border: `1px solid ${c.border}`,
      borderRadius: 8, padding: "10px 14px", opacity,
      animation: "fadeIn 0.2s ease",
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
        <span style={{ fontWeight: 700, fontSize: 13, color: "#e5e7eb" }}>{feature}</span>
        <span style={{
          fontSize: 13, fontWeight: 700, fontFamily: "monospace",
          color: c.text,
        }}>
          {isPos ? "+" : ""}{shap_value.toFixed(4)}
        </span>
      </div>
      {/* Bar */}
      <div style={{ height: 6, background: "#0d1117", borderRadius: 3, overflow: "hidden", marginBottom: 6 }}>
        <div style={{
          height: "100%", width: `${pct}%`, borderRadius: 3,
          background: c.bar, transition: "width 0.4s ease",
        }} />
      </div>
      <div style={{ fontSize: 11, color: "#6b7280" }}>{explanation}</div>
    </div>
  );
};

// ── global importance bar ─────────────────────────────────────────────────────
const ImportanceBar = ({ feature, mean_abs_shap, rank, maxVal }) => {
  const pct = maxVal > 0 ? (mean_abs_shap / maxVal) * 100 : 0;
  const hue = Math.max(120 - rank * 15, 0);
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
        <span style={{ fontSize: 13, color: "#e5e7eb" }}>{feature}</span>
        <span style={{ fontSize: 12, color: "#9ca3af", fontFamily: "monospace" }}>
          {mean_abs_shap.toFixed(4)}
        </span>
      </div>
      <div style={{ height: 8, background: "#0d1117", borderRadius: 4, overflow: "hidden" }}>
        <div style={{
          height: "100%", width: `${pct}%`, borderRadius: 4,
          background: `hsl(${hue}, 70%, 55%)`, transition: "width 0.5s ease",
        }} />
      </div>
    </div>
  );
};

// ── default form values ───────────────────────────────────────────────────────
const DEFAULT_FORM = {
  Crop: "Rice", CropCoveredArea: "5.0", CHeight: "100",
  CNext: "Wheat", CLast: "Maize", CTransp: "Manual",
  IrriType: "Drip", IrriSource: "Canal", IrriCount: "4",
  WaterCov: "80", Season: "Kharif",
};

// ── main component ────────────────────────────────────────────────────────────
export default function PredictionExplainer() {
  const [activeTab, setActiveTab]       = useState("explain");
  const [form, setForm]                 = useState(DEFAULT_FORM);
  const [result, setResult]             = useState(null);
  const [loading, setLoading]           = useState(false);
  const [error, setError]               = useState(null);
  const [globalData, setGlobalData]     = useState(null);
  const [globalLoading, setGlobalLoading] = useState(false);

  const fetchGlobal = useCallback(async () => {
    setGlobalLoading(true);
    try {
      const r = await fetch(`${API_BASE}/api/explain/global-importance?limit=100`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      setGlobalData(await r.json());
    } catch (e) {
      // non-fatal
    } finally {
      setGlobalLoading(false);
    }
  }, []);

  useEffect(() => {
    if (activeTab === "global") fetchGlobal();
  }, [activeTab, fetchGlobal]);

  const handleChange = (field, value) => setForm(f => ({ ...f, [field]: value }));

  const handleExplain = async () => {
    setLoading(true);
    setResult(null);
    setError(null);
    try {
      const payload = {
        Crop:            form.Crop,
        CropCoveredArea: parseFloat(form.CropCoveredArea),
        CHeight:         parseInt(form.CHeight, 10),
        CNext:           form.CNext,
        CLast:           form.CLast,
        CTransp:         form.CTransp,
        IrriType:        form.IrriType,
        IrriSource:      form.IrriSource,
        IrriCount:       parseInt(form.IrriCount, 10),
        WaterCov:        parseInt(form.WaterCov, 10),
        Season:          form.Season,
      };
      const r = await fetch(`${API_BASE}/api/explain/prediction`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail || `HTTP ${r.status}`);
      setResult(d);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const maxAbs = result
    ? Math.max(...result.feature_contributions.map(c => Math.abs(c.shap_value)))
    : 1;

  return (
    <div style={{
      minHeight: "100vh", background: "#030712",
      color: "#e5e7eb", fontFamily: "'IBM Plex Mono','Courier New',monospace",
      padding: "32px 24px",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;700&display=swap');
        @keyframes spin   { to { transform: rotate(360deg); } }
        @keyframes fadeIn { from { opacity:0; transform:translateY(5px); } to { opacity:1; transform:none; } }
        .tab-btn { background:none; border:none; cursor:pointer; padding:8px 18px;
          border-radius:8px; font-family:inherit; font-size:13px; color:#9ca3af; transition:all 0.15s; }
        .tab-btn:hover  { background:#1f2937; }
        .tab-btn.active { background:#1f2937; color:#4ade80; }
        .explain-btn { background:#14532d; border:1px solid #166534; color:#4ade80;
          padding:11px 24px; border-radius:8px; cursor:pointer; font-family:inherit;
          font-size:13px; font-weight:700; width:100%; transition:all 0.15s; }
        .explain-btn:hover:not(:disabled) { background:#166534; }
        .explain-btn:disabled { opacity:0.45; cursor:not-allowed; }
        input, select { outline:none; }
        input:focus, select:focus { border-color:#374151 !important; }
      `}</style>

      <div style={{ maxWidth: 980, margin: "0 auto" }}>
        {/* Header */}
        <div style={{ marginBottom: 28 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
            <span style={{ fontSize: 20, color: "#4ade80" }}>⬡</span>
            <h1 style={{ margin: 0, fontSize: 20, fontWeight: 700, color: "#f9fafb" }}>
              Prediction Explainer
            </h1>
          </div>
          <p style={{ margin: 0, fontSize: 12, color: "#6b7280" }}>
            SHAP feature importance · Fasal Saathi XGBoost Yield Model
          </p>
        </div>

        {/* Tabs */}
        <div style={{ display: "flex", gap: 4, marginBottom: 24, borderBottom: "1px solid #1f2937", paddingBottom: 8 }}>
          {["explain", "global"].map(t => (
            <button key={t} className={`tab-btn${activeTab === t ? " active" : ""}`} onClick={() => setActiveTab(t)}>
              {{ explain: "Explain Prediction", global: "Global Feature Importance" }[t]}
            </button>
          ))}
        </div>

        {/* ── EXPLAIN TAB ── */}
        {activeTab === "explain" && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1.4fr", gap: 16, animation: "fadeIn 0.25s ease" }}>
            {/* Input form */}
            <Card>
              <SectionTitle>Input Features</SectionTitle>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 16 }}>
                {[
                  ["Crop",            "text",   "Crop"],
                  ["CNext",           "text",   "Next Crop"],
                  ["CLast",           "text",   "Last Crop"],
                  ["CTransp",         "text",   "Transport"],
                  ["IrriType",        "text",   "Irrigation Type"],
                  ["IrriSource",      "text",   "Irrigation Source"],
                  ["Season",          "text",   "Season"],
                  ["CropCoveredArea", "number", "Area (acres)"],
                  ["CHeight",         "number", "Crop Height (cm)"],
                  ["IrriCount",       "number", "Irrigation Count"],
                  ["WaterCov",        "number", "Water Coverage (%)"],
                ].map(([field, type, label]) => (
                  <Field key={field} label={label}>
                    <input
                      type={type}
                      value={form[field]}
                      onChange={e => handleChange(field, e.target.value)}
                      style={inputStyle}
                    />
                  </Field>
                ))}
              </div>
              <button
                className="explain-btn"
                onClick={handleExplain}
                disabled={loading}
              >
                {loading ? <><Spinner /> &nbsp;Computing SHAP…</> : "▶  Explain This Prediction"}
              </button>
              {error && (
                <p style={{ margin: "12px 0 0", color: "#f87171", fontSize: 12 }}>⚠ {error}</p>
              )}
            </Card>

            {/* Results */}
            <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              {!result && !loading && (
                <Card style={{ display: "flex", alignItems: "center", justifyContent: "center", minHeight: 200 }}>
                  <p style={{ color: "#4b5563", fontSize: 13, textAlign: "center" }}>
                    Fill in the features and click<br />"Explain This Prediction" to see SHAP values.
                  </p>
                </Card>
              )}

              {result && (
                <>
                  {/* Prediction + summary */}
                  <Card style={{ borderColor: "#1a5c30", background: "#0a1a0f" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
                      <div>
                        <div style={{ fontSize: 11, color: "#6b7280", letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: 4 }}>
                          Predicted Yield
                        </div>
                        <div style={{ fontSize: 32, fontWeight: 700, color: "#4ade80", fontFamily: "monospace" }}>
                          {result.prediction.toFixed(2)}
                        </div>
                      </div>
                      <div style={{ textAlign: "right" }}>
                        <div style={{ fontSize: 11, color: "#6b7280", marginBottom: 4, textTransform: "uppercase", letterSpacing: "0.08em" }}>Base Value</div>
                        <div style={{ fontSize: 16, color: "#9ca3af", fontFamily: "monospace" }}>{result.bias.toFixed(2)}</div>
                      </div>
                    </div>
                    <div style={{
                      padding: "10px 14px", borderRadius: 8,
                      background: "#0d1117", border: "1px solid #1f2937",
                      fontSize: 13, color: "#d1fae5", lineHeight: 1.6,
                    }}>
                      {result.explanation_summary}
                    </div>
                    <div style={{ marginTop: 8, fontSize: 11, color: "#4b5563" }}>
                      Explained at {result.explained_at.replace("T", " ").slice(0, 19)} UTC
                    </div>
                  </Card>

                  {/* SHAP waterfall */}
                  <Card>
                    <SectionTitle>
                      Feature Contributions — SHAP Values ({result.feature_contributions.length} features)
                    </SectionTitle>
                    <div style={{ display: "flex", gap: 12, marginBottom: 12 }}>
                      <span style={{ fontSize: 11, color: "#4ade80" }}>■ Positive = increases yield</span>
                      <span style={{ fontSize: 11, color: "#f87171" }}>■ Negative = decreases yield</span>
                    </div>
                    <div style={{ display: "flex", flexDirection: "column", gap: 6, maxHeight: 380, overflowY: "auto" }}>
                      {result.feature_contributions.map((c, i) => (
                        <ShapBar key={i} {...c} maxAbs={maxAbs} />
                      ))}
                    </div>
                  </Card>
                </>
              )}
            </div>
          </div>
        )}

        {/* ── GLOBAL TAB ── */}
        {activeTab === "global" && (
          <div style={{ animation: "fadeIn 0.25s ease" }}>
            <Card>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
                <SectionTitle>Global Feature Importance</SectionTitle>
                <button
                  onClick={fetchGlobal}
                  disabled={globalLoading}
                  style={{
                    background: "none", border: "1px solid #1f2937",
                    color: "#9ca3af", padding: "6px 14px", borderRadius: 8,
                    cursor: "pointer", fontFamily: "inherit", fontSize: 12,
                  }}
                >
                  {globalLoading ? <Spinner /> : "↻ Refresh"}
                </button>
              </div>

              {globalLoading && (
                <div style={{ textAlign: "center", padding: 40, color: "#6b7280" }}>
                  <Spinner /> <span style={{ marginLeft: 8 }}>Loading…</span>
                </div>
              )}

              {!globalLoading && globalData && (
                globalData.global_importance?.length > 0 ? (
                  <>
                    <div style={{ fontSize: 12, color: "#6b7280", marginBottom: 16 }}>
                      Mean |SHAP| across {globalData.sample_count} recent predictions.
                      Higher = more influential on yield.
                    </div>
                    {(() => {
                      const maxVal = globalData.global_importance[0]?.mean_abs_shap || 1;
                      return globalData.global_importance.map((item, i) => (
                        <ImportanceBar key={item.feature} rank={i} maxVal={maxVal} {...item} />
                      ));
                    })()}
                  </>
                ) : (
                  <div style={{ textAlign: "center", padding: 40 }}>
                    <p style={{ color: "#4b5563", fontSize: 13 }}>
                      No data yet. Run predictions on the Explain tab first.
                    </p>
                    <button
                      onClick={() => setActiveTab("explain")}
                      style={{
                        marginTop: 12, background: "#14532d", border: "1px solid #166534",
                        color: "#4ade80", padding: "8px 20px", borderRadius: 8,
                        cursor: "pointer", fontFamily: "inherit", fontSize: 13,
                      }}
                    >
                      Go to Explain tab →
                    </button>
                  </div>
                )
              )}
            </Card>
          </div>
        )}
      </div>
    </div>
  );
}