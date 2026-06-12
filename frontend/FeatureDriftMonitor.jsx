import { useState, useEffect, useCallback } from "react";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

// ─── colour tokens ───────────────────────────────────────────────────────────
const STATUS_COLORS = {
  ok:         { bg: "#0f2a1a", border: "#1a5c30", text: "#4ade80", dot: "#22c55e" },
  warn:       { bg: "#2a1f0a", border: "#7a4f00", text: "#fbbf24", dot: "#f59e0b" },
  alert:      { bg: "#2a0a0a", border: "#7a1a1a", text: "#f87171", dot: "#ef4444" },
  missing:    { bg: "#1a1a2a", border: "#3a3a7a", text: "#a5b4fc", dot: "#818cf8" },
  extra:      { bg: "#1a1a2a", border: "#3a3a7a", text: "#a5b4fc", dot: "#818cf8" },
  type_error: { bg: "#2a1a00", border: "#7a4a00", text: "#fb923c", dot: "#f97316" },
};
const statusColor = (s) => STATUS_COLORS[s] || STATUS_COLORS.ok;

// ─── tiny helpers ────────────────────────────────────────────────────────────
const Badge = ({ status, label }) => {
  const c = statusColor(status);
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 5,
      padding: "2px 10px", borderRadius: 999,
      background: c.bg, border: `1px solid ${c.border}`,
      color: c.text, fontSize: 11, fontWeight: 700,
      letterSpacing: "0.06em", textTransform: "uppercase",
    }}>
      <span style={{ width: 6, height: 6, borderRadius: "50%", background: c.dot, flexShrink: 0 }} />
      {label || status}
    </span>
  );
};

const Card = ({ children, style }) => (
  <div style={{
    background: "#111827", border: "1px solid #1f2937",
    borderRadius: 12, padding: "20px 24px",
    ...style,
  }}>
    {children}
  </div>
);

const SectionTitle = ({ children }) => (
  <h3 style={{
    margin: "0 0 14px", fontSize: 12, fontWeight: 700,
    letterSpacing: "0.12em", textTransform: "uppercase",
    color: "#6b7280",
  }}>{children}</h3>
);

const Spinner = () => (
  <span style={{
    display: "inline-block", width: 14, height: 14,
    border: "2px solid #374151", borderTopColor: "#4ade80",
    borderRadius: "50%",
    animation: "spin 0.7s linear infinite",
  }} />
);

// ─── main component ──────────────────────────────────────────────────────────
export default function FeatureDriftMonitor() {
  const [status, setStatus]       = useState(null);
  const [logs, setLogs]           = useState([]);
  const [loading, setLoading]     = useState(true);
  const [error, setError]         = useState(null);
  const [activeTab, setActiveTab] = useState("overview"); // "overview" | "validate" | "logs"
  const [refreshing, setRefreshing] = useState(false);

  // validate form state
  const [validateInput, setValidateInput]   = useState(
    `{\n  "Crop": "Rice",\n  "CropCoveredArea": 5.0,\n  "CHeight": 100,\n  "CNext": "Wheat",\n  "CLast": "Maize",\n  "CTransp": "Manual",\n  "IrriType": "Drip",\n  "IrriSource": "Canal",\n  "IrriCount": 4,\n  "WaterCov": 80,\n  "Season": "Kharif"\n}`
  );
  const [validateResult, setValidateResult] = useState(null);
  const [validating, setValidating]         = useState(false);
  const [validateError, setValidateError]   = useState(null);

  // baseline update state
  const [updatingBaseline, setUpdatingBaseline] = useState(false);
  const [baselineMsg, setBaselineMsg]           = useState(null);

  const fetchStatus = useCallback(async () => {
    try {
      const r = await fetch(`${API_BASE}/api/feature-drift/status`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      setStatus(await r.json());
    } catch (e) {
      setError(e.message);
    }
  }, []);

  const fetchLogs = useCallback(async () => {
    try {
      const r = await fetch(`${API_BASE}/api/feature-drift/logs?limit=50`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const d = await r.json();
      setLogs(d.entries || []);
    } catch (e) {
      // non-fatal — logs section just stays empty
    }
  }, []);

  const loadAll = useCallback(async () => {
    setLoading(true);
    setError(null);
    await Promise.all([fetchStatus(), fetchLogs()]);
    setLoading(false);
  }, [fetchStatus, fetchLogs]);

  const handleRefresh = async () => {
    setRefreshing(true);
    await Promise.all([fetchStatus(), fetchLogs()]);
    setRefreshing(false);
  };

  useEffect(() => { loadAll(); }, [loadAll]);

  const handleValidate = async () => {
    setValidating(true);
    setValidateResult(null);
    setValidateError(null);
    try {
      let features;
      try { features = JSON.parse(validateInput); }
      catch { throw new Error("Invalid JSON in the features box."); }

      const r = await fetch(`${API_BASE}/api/feature-drift/validate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features }),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail || `HTTP ${r.status}`);
      setValidateResult(d);
      // refresh logs + status after a validate call
      fetchStatus();
      fetchLogs();
    } catch (e) {
      setValidateError(e.message);
    } finally {
      setValidating(false);
    }
  };

  const handleUpdateBaseline = async () => {
    setUpdatingBaseline(true);
    setBaselineMsg(null);
    try {
      const r = await fetch(`${API_BASE}/api/feature-drift/baseline/update`, {
        method: "POST",
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail || `HTTP ${r.status}`);
      setBaselineMsg({ ok: true, text: d.message });
      fetchStatus();
    } catch (e) {
      setBaselineMsg({ ok: false, text: e.message });
    } finally {
      setUpdatingBaseline(false);
    }
  };

  // ── render ────────────────────────────────────────────────────────────────
  return (
    <div style={{
      minHeight: "100vh", background: "#030712",
      color: "#e5e7eb", fontFamily: "'IBM Plex Mono', 'Courier New', monospace",
      padding: "32px 24px",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;700&display=swap');
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes fadeIn { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:none; } }
        .drift-row:hover { background: #1f2937 !important; }
        .tab-btn { background:none; border:none; cursor:pointer; padding:8px 18px;
          border-radius:8px; font-family:inherit; font-size:13px; transition:all 0.15s; }
        .tab-btn:hover { background:#1f2937; }
        .tab-btn.active { background:#1f2937; color:#4ade80; }
        .validate-btn { background:#14532d; border:1px solid #166534; color:#4ade80;
          padding:10px 22px; border-radius:8px; cursor:pointer; font-family:inherit;
          font-size:13px; font-weight:700; transition:all 0.15s; }
        .validate-btn:hover:not(:disabled) { background:#166534; }
        .validate-btn:disabled { opacity:0.5; cursor:not-allowed; }
        .baseline-btn { background:#1e3a5f; border:1px solid #1d4ed8; color:#93c5fd;
          padding:8px 18px; border-radius:8px; cursor:pointer; font-family:inherit;
          font-size:12px; font-weight:700; transition:all 0.15s; }
        .baseline-btn:hover:not(:disabled) { background:#1d4ed8; color:#fff; }
        .baseline-btn:disabled { opacity:0.5; cursor:not-allowed; }
        textarea { resize:vertical; }
      `}</style>

      {/* Header */}
      <div style={{ maxWidth: 960, margin: "0 auto" }}>
        <div style={{ display:"flex", justifyContent:"space-between", alignItems:"flex-start", marginBottom: 28 }}>
          <div>
            <div style={{ display:"flex", alignItems:"center", gap:10, marginBottom:6 }}>
              <span style={{ fontSize:22, color:"#4ade80" }}>⬡</span>
              <h1 style={{ margin:0, fontSize:20, fontWeight:700, letterSpacing:"-0.01em", color:"#f9fafb" }}>
                Feature Drift Monitor
              </h1>
            </div>
            <p style={{ margin:0, fontSize:12, color:"#6b7280" }}>
              Training-serving skew detection · Fasal Saathi ML Pipeline
            </p>
          </div>
          <button
            onClick={handleRefresh}
            disabled={refreshing}
            style={{
              display:"flex", alignItems:"center", gap:7,
              background:"#111827", border:"1px solid #1f2937",
              color:"#9ca3af", padding:"8px 16px", borderRadius:8,
              cursor:"pointer", fontFamily:"inherit", fontSize:12,
            }}
          >
            {refreshing ? <Spinner /> : "↻"} Refresh
          </button>
        </div>

        {/* Tabs */}
        <div style={{ display:"flex", gap:4, marginBottom:24, borderBottom:"1px solid #1f2937", paddingBottom:8 }}>
          {["overview", "validate", "logs"].map(t => (
            <button
              key={t}
              className={`tab-btn${activeTab === t ? " active" : ""}`}
              style={{ color: activeTab === t ? "#4ade80" : "#9ca3af" }}
              onClick={() => setActiveTab(t)}
            >
              {{ overview:"Overview", validate:"Validate Payload", logs:"Drift Logs" }[t]}
            </button>
          ))}
        </div>

        {loading && (
          <div style={{ textAlign:"center", padding:60, color:"#6b7280" }}>
            <Spinner /> <span style={{ marginLeft:10 }}>Loading…</span>
          </div>
        )}
        {error && !loading && (
          <Card style={{ borderColor:"#7a1a1a", background:"#2a0a0a" }}>
            <p style={{ margin:0, color:"#f87171", fontSize:13 }}>⚠ Could not reach backend: {error}</p>
          </Card>
        )}

        {!loading && !error && (
          <>
            {/* ── OVERVIEW TAB ── */}
            {activeTab === "overview" && status && (
              <div style={{ animation:"fadeIn 0.25s ease" }}>
                {/* Stats row */}
                <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fit,minmax(180px,1fr))", gap:12, marginBottom:20 }}>
                  {[
                    { label:"Baseline", value: status.baseline_exists ? "Active" : "Missing", color: status.baseline_exists ? "#4ade80" : "#f87171" },
                    { label:"Features Tracked", value: status.total_features_tracked, color:"#e5e7eb" },
                    { label:"Numeric", value: status.numeric_features, color:"#93c5fd" },
                    { label:"Categorical", value: status.categorical_features, color:"#fbbf24" },
                    { label:"Log Entries", value: status.log_entry_count, color:"#e5e7eb" },
                  ].map(s => (
                    <Card key={s.label}>
                      <div style={{ fontSize:11, color:"#6b7280", marginBottom:8, letterSpacing:"0.08em", textTransform:"uppercase" }}>{s.label}</div>
                      <div style={{ fontSize:24, fontWeight:700, color: s.color }}>{s.value}</div>
                    </Card>
                  ))}
                </div>

                {/* Baseline info */}
                <Card style={{ marginBottom:16 }}>
                  <SectionTitle>Baseline Info</SectionTitle>
                  {status.baseline_exists ? (
                    <div style={{ fontSize:13, color:"#9ca3af", lineHeight:1.8 }}>
                      <div>Generated at: <span style={{ color:"#e5e7eb" }}>{status.baseline_generated_at}</span></div>
                      <div>Numeric features: <span style={{ color:"#93c5fd" }}>{status.numeric_features}</span></div>
                      <div>Categorical features: <span style={{ color:"#fbbf24" }}>{status.categorical_features}</span></div>
                    </div>
                  ) : (
                    <p style={{ margin:0, color:"#f87171", fontSize:13 }}>
                      No baseline found. Run <code style={{ color:"#4ade80" }}>train_model.py</code> or use the button below.
                    </p>
                  )}
                  <div style={{ marginTop:16, display:"flex", alignItems:"center", gap:12, flexWrap:"wrap" }}>
                    <button
                      className="baseline-btn"
                      onClick={handleUpdateBaseline}
                      disabled={updatingBaseline}
                    >
                      {updatingBaseline ? <><Spinner /> Rebuilding…</> : "⟳  Rebuild Baseline from Train.csv"}
                    </button>
                    {baselineMsg && (
                      <span style={{ fontSize:12, color: baselineMsg.ok ? "#4ade80" : "#f87171" }}>
                        {baselineMsg.ok ? "✓ " : "✗ "}{baselineMsg.text}
                      </span>
                    )}
                  </div>
                </Card>

                {/* Recent alerts */}
                <Card>
                  <SectionTitle>Recent Alerts & Warnings</SectionTitle>
                  {status.recent_alerts.length === 0 ? (
                    <p style={{ margin:0, color:"#6b7280", fontSize:13 }}>No alerts in recent log entries. ✓</p>
                  ) : (
                    <div style={{ display:"flex", flexDirection:"column", gap:8 }}>
                      {status.recent_alerts.slice(-10).reverse().map((a, i) => (
                        <div key={i} className="drift-row" style={{
                          background:"#0d1117", borderRadius:8, padding:"10px 14px",
                          border:`1px solid ${a.overall_status === "alert" ? "#7a1a1a" : "#7a4f00"}`,
                          animation:`fadeIn 0.2s ease ${i * 0.04}s both`,
                        }}>
                          <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:4 }}>
                            <Badge status={a.overall_status} />
                            <span style={{ fontSize:11, color:"#4b5563" }}>{a.timestamp?.replace("T", " ").slice(0, 19)}</span>
                          </div>
                          {a.drift_alerts?.slice(0, 3).map((d, j) => (
                            <div key={j} style={{ fontSize:12, color:"#9ca3af", marginTop:3 }}>
                              <span style={{ color:"#e5e7eb" }}>{d.feature}</span>: {d.message?.slice(0, 80)}
                              {d.message?.length > 80 ? "…" : ""}
                            </div>
                          ))}
                        </div>
                      ))}
                    </div>
                  )}
                </Card>
              </div>
            )}

            {/* ── VALIDATE TAB ── */}
            {activeTab === "validate" && (
              <div style={{ animation:"fadeIn 0.25s ease" }}>
                <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:16 }}>
                  {/* Input */}
                  <Card>
                    <SectionTitle>Feature Payload (JSON)</SectionTitle>
                    <textarea
                      value={validateInput}
                      onChange={e => setValidateInput(e.target.value)}
                      rows={16}
                      style={{
                        width:"100%", boxSizing:"border-box",
                        background:"#0d1117", border:"1px solid #1f2937",
                        borderRadius:8, color:"#4ade80", fontFamily:"inherit",
                        fontSize:12, padding:12, lineHeight:1.6,
                      }}
                    />
                    <button
                      className="validate-btn"
                      style={{ marginTop:12, width:"100%" }}
                      onClick={handleValidate}
                      disabled={validating}
                    >
                      {validating ? <><Spinner /> Validating…</> : "▶  Run Drift Validation"}
                    </button>
                    {validateError && (
                      <p style={{ margin:"10px 0 0", color:"#f87171", fontSize:12 }}>⚠ {validateError}</p>
                    )}
                  </Card>

                  {/* Result */}
                  <Card>
                    <SectionTitle>Validation Result</SectionTitle>
                    {!validateResult && (
                      <p style={{ color:"#4b5563", fontSize:13 }}>Run validation to see results.</p>
                    )}
                    {validateResult && (
                      <div style={{ animation:"fadeIn 0.2s ease" }}>
                        <div style={{ display:"flex", gap:10, alignItems:"center", marginBottom:16 }}>
                          <Badge status={validateResult.overall_status} label={validateResult.overall_status.toUpperCase()} />
                          <Badge
                            status={validateResult.valid ? "ok" : "alert"}
                            label={validateResult.valid ? "Schema Valid" : "Schema Errors"}
                          />
                        </div>

                        {/* Schema errors */}
                        {validateResult.schema_errors.length > 0 && (
                          <div style={{ marginBottom:14 }}>
                            <div style={{ fontSize:11, color:"#6b7280", marginBottom:6, letterSpacing:"0.08em", textTransform:"uppercase" }}>Schema Errors</div>
                            {validateResult.schema_errors.map((e, i) => (
                              <div key={i} style={{
                                background:"#2a0a0a", border:"1px solid #7a1a1a",
                                borderRadius:6, padding:"6px 10px", fontSize:12, color:"#f87171", marginBottom:4,
                              }}>
                                {e}
                              </div>
                            ))}
                          </div>
                        )}

                        {/* Per-feature drift */}
                        <div style={{ fontSize:11, color:"#6b7280", marginBottom:8, letterSpacing:"0.08em", textTransform:"uppercase" }}>
                          Per-Feature Drift ({validateResult.drift_results.length})
                        </div>
                        <div style={{ display:"flex", flexDirection:"column", gap:6, maxHeight:320, overflowY:"auto" }}>
                          {validateResult.drift_results.map((r, i) => {
                            const c = statusColor(r.status);
                            return (
                              <div key={i} style={{
                                background: c.bg, border:`1px solid ${c.border}`,
                                borderRadius:8, padding:"8px 12px",
                              }}>
                                <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center" }}>
                                  <span style={{ fontWeight:700, fontSize:13, color:"#e5e7eb" }}>{r.feature}</span>
                                  <Badge status={r.status} />
                                </div>
                                <div style={{ fontSize:11, color:"#9ca3af", marginTop:4 }}>{r.message}</div>
                                {r.ks_statistic !== null && r.ks_statistic !== undefined && (
                                  <div style={{ fontSize:11, color:"#6b7280", marginTop:2 }}>
                                    KS={r.ks_statistic} · p={r.p_value}
                                  </div>
                                )}
                              </div>
                            );
                          })}
                        </div>

                        <div style={{ marginTop:12, fontSize:11, color:"#4b5563" }}>
                          Validated at: {validateResult.validated_at?.replace("T", " ").slice(0, 19)} UTC
                        </div>
                      </div>
                    )}
                  </Card>
                </div>
              </div>
            )}

            {/* ── LOGS TAB ── */}
            {activeTab === "logs" && (
              <div style={{ animation:"fadeIn 0.25s ease" }}>
                <Card>
                  <SectionTitle>Drift Log Entries ({logs.length})</SectionTitle>
                  {logs.length === 0 ? (
                    <p style={{ color:"#4b5563", fontSize:13 }}>No log entries yet. Validate a payload to generate entries.</p>
                  ) : (
                    <div style={{ display:"flex", flexDirection:"column", gap:8 }}>
                      {[...logs].reverse().map((entry, i) => (
                        <div key={i} className="drift-row" style={{
                          background:"#0d1117", borderRadius:8, padding:"12px 16px",
                          border:`1px solid ${entry.overall_status === "alert" ? "#7a1a1a" : entry.overall_status === "warn" ? "#7a4f00" : "#1f2937"}`,
                          animation:`fadeIn 0.18s ease ${Math.min(i, 10) * 0.03}s both`,
                        }}>
                          <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:6 }}>
                            <div style={{ display:"flex", gap:8, alignItems:"center" }}>
                              <Badge status={entry.overall_status} />
                              <span style={{ fontSize:12, color:"#9ca3af" }}>
                                {entry.feature_count} features · {entry.schema_error_count} schema errors
                              </span>
                            </div>
                            <span style={{ fontSize:11, color:"#4b5563" }}>
                              {entry.timestamp?.replace("T", " ").slice(0, 19)} UTC
                            </span>
                          </div>
                          {entry.drift_alerts?.length > 0 && (
                            <div style={{ display:"flex", flexWrap:"wrap", gap:6, marginTop:6 }}>
                              {entry.drift_alerts.slice(0, 5).map((a, j) => (
                                <span key={j} style={{
                                  fontSize:11, background:"#1a0a0a", border:"1px solid #7a1a1a",
                                  color:"#f87171", borderRadius:4, padding:"2px 8px",
                                }}>
                                  {a.feature}: {a.status}
                                </span>
                              ))}
                              {entry.drift_alerts.length > 5 && (
                                <span style={{ fontSize:11, color:"#6b7280" }}>+{entry.drift_alerts.length - 5} more</span>
                              )}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </Card>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}