import { useState, useEffect, useCallback } from "react";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

// ── colour tokens ─────────────────────────────────────────────────────────────
const STATUS_COLORS = {
  pending:  { bg: "#1a1a2a", border: "#3a3a7a", text: "#a5b4fc", dot: "#818cf8" },
  progress: { bg: "#0a1a2a", border: "#1a4a7a", text: "#7dd3fc", dot: "#38bdf8" },
  success:  { bg: "#0f2a1a", border: "#1a5c30", text: "#4ade80", dot: "#22c55e" },
  failure:  { bg: "#2a0a0a", border: "#7a1a1a", text: "#f87171", dot: "#ef4444" },
  unknown:  { bg: "#1a1a1a", border: "#374151", text: "#9ca3af", dot: "#6b7280" },
};
const sc = (s) => STATUS_COLORS[s?.toLowerCase()] || STATUS_COLORS.unknown;

const PARAM_COLORS = [
  "#4ade80", "#38bdf8", "#fbbf24", "#f87171", "#a78bfa", "#2dd4bf", "#fb923c", "#e879f9",
];

// ── tiny shared components ────────────────────────────────────────────────────
const Badge = ({ status, label }) => {
  const c = sc(status);
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 5,
      padding: "2px 10px", borderRadius: 999,
      background: c.bg, border: `1px solid ${c.border}`,
      color: c.text, fontSize: 11, fontWeight: 700,
      letterSpacing: "0.06em", textTransform: "uppercase",
    }}>
      <span style={{ width: 6, height: 6, borderRadius: "50%", background: c.dot }} />
      {label || status}
    </span>
  );
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

// ── bar chart (svg) for param importance ──────────────────────────────────────
const ParamImportanceChart = ({ trials }) => {
  if (!trials || trials.length < 2) return null;

  // Compute simple correlation-based importance: variance of each param vs RMSE
  const params = Object.keys(trials[0].params || {});
  const importance = params.map((param) => {
    const values = trials.map((t) => t.params[param]);
    const rmses = trials.map((t) => t.rmse);
    const meanV = values.reduce((a, b) => a + b, 0) / values.length;
    const meanR = rmses.reduce((a, b) => a + b, 0) / rmses.length;
    let num = 0, denV = 0, denR = 0;
    for (let i = 0; i < values.length; i++) {
      const dv = values[i] - meanV;
      const dr = rmses[i] - meanR;
      num += dv * dr;
      denV += dv * dv;
      denR += dr * dr;
    }
    const corr = denV && denR ? Math.abs(num / Math.sqrt(denV * denR)) : 0;
    return { param, importance: corr };
  }).sort((a, b) => b.importance - a.importance);

  const maxImp = Math.max(...importance.map((i) => i.importance), 0.001);

  return (
    <Card style={{ marginTop: 16 }}>
      <SectionTitle>Parameter Importance (correlation with RMSE)</SectionTitle>
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {importance.map((item, idx) => {
          const pct = (item.importance / maxImp) * 100;
          return (
            <div key={item.param} style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <div style={{ width: 110, fontSize: 11, color: "#9ca3af", textAlign: "right" }}>{item.param}</div>
              <div style={{ flex: 1, height: 18, background: "#0d1117", borderRadius: 4, overflow: "hidden" }}>
                <div style={{
                  width: `${pct}%`, height: "100%",
                  background: PARAM_COLORS[idx % PARAM_COLORS.length],
                  borderRadius: 4, transition: "width 0.4s ease",
                }} />
              </div>
              <div style={{ width: 40, fontSize: 11, color: "#6b7280" }}>{pct.toFixed(0)}%</div>
            </div>
          );
        })}
      </div>
    </Card>
  );
};

// ── benchmark comparison card ─────────────────────────────────────────────────
const BenchmarkCard = ({ benchmark }) => {
  if (!benchmark || !benchmark.production_model_exists) {
    return (
      <Card style={{ marginTop: 16, borderColor: "#374151" }}>
        <SectionTitle>Benchmark vs Production</SectionTitle>
        <p style={{ margin: 0, fontSize: 13, color: "#6b7280" }}>
          No production model found for benchmark comparison.
        </p>
      </Card>
    );
  }

  const improved = benchmark.improved;
  const color = improved ? "#4ade80" : "#f87171";

  return (
    <Card style={{ marginTop: 16, borderColor: improved ? "#1a5c30" : "#7a1a1a", background: improved ? "#0f2a1a" : "#2a0a0a" }}>
      <SectionTitle>Benchmark vs Production</SectionTitle>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12 }}>
        <div>
          <div style={{ fontSize: 10, color: "#6b7280", marginBottom: 4, textTransform: "uppercase", letterSpacing: "0.08em" }}>Production RMSE</div>
          <div style={{ fontSize: 18, fontWeight: 700, color: "#e5e7eb" }}>{benchmark.production_rmse?.toFixed(4)}</div>
        </div>
        <div>
          <div style={{ fontSize: 10, color: "#6b7280", marginBottom: 4, textTransform: "uppercase", letterSpacing: "0.08em" }}>Optimized RMSE</div>
          <div style={{ fontSize: 18, fontWeight: 700, color }}>{benchmark.optimized_rmse?.toFixed(4)}</div>
        </div>
        <div>
          <div style={{ fontSize: 10, color: "#6b7280", marginBottom: 4, textTransform: "uppercase", letterSpacing: "0.08em" }}>Improvement</div>
          <div style={{ fontSize: 18, fontWeight: 700, color }}>
            {improved ? "↓" : "↑"} {Math.abs(benchmark.improvement_pct || 0).toFixed(2)}%
          </div>
        </div>
      </div>
      <div style={{ marginTop: 10, fontSize: 12, color }}>
        {improved
          ? "✓ Optimized model outperforms production. Ready for promotion."
          : "✗ Optimized model does not outperform production. Keep current model."}
      </div>
    </Card>
  );
};

// ── trial history sparkline ───────────────────────────────────────────────────
const TrialSparkline = ({ trials }) => {
  if (!trials || trials.length < 2) return null;
  const rmses = trials.map((t) => t.rmse);
  const minR = Math.min(...rmses);
  const maxR = Math.max(...rmses);
  const range = maxR - minR || 1;
  const w = 600, h = 120, pad = 20;
  const points = rmses.map((r, i) => {
    const x = pad + (i / (rmses.length - 1)) * (w - pad * 2);
    const y = h - pad - ((r - minR) / range) * (h - pad * 2);
    return `${x},${y}`;
  }).join(" ");

  return (
    <Card style={{ marginTop: 16 }}>
      <SectionTitle>RMSE Convergence</SectionTitle>
      <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`} style={{ overflow: "visible" }}>
        <polyline
          fill="none"
          stroke="#4ade80"
          strokeWidth="2"
          points={points}
          style={{ filter: "drop-shadow(0 0 4px rgba(74,222,128,0.3))" }}
        />
        {rmses.map((r, i) => {
          const x = pad + (i / (rmses.length - 1)) * (w - pad * 2);
          const y = h - pad - ((r - minR) / range) * (h - pad * 2);
          return (
            <circle key={i} cx={x} cy={y} r="3" fill="#111827" stroke="#4ade80" strokeWidth="1.5">
              <title>Trial {i + 1}: RMSE {r.toFixed(4)}</title>
            </circle>
          );
        })}
        <text x={pad} y={h - 4} fill="#6b7280" fontSize="10" fontFamily="monospace">{minR.toFixed(4)}</text>
        <text x={w - pad - 40} y={14} fill="#6b7280" fontSize="10" fontFamily="monospace">{maxR.toFixed(4)}</text>
      </svg>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "#4b5563", marginTop: 4 }}>
        <span>Trial 1</span>
        <span>Trial {rmses.length}</span>
      </div>
    </Card>
  );
};

// ── main component ────────────────────────────────────────────────────────────
export default function HyperparameterOptimizer() {
  const [status, setStatus]         = useState(null);
  const [trials, setTrials]         = useState([]);
  const [activeTab, setActiveTab]   = useState("overview");
  const [loading, setLoading]       = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError]           = useState(null);

  // trigger form state
  const [csvPath, setCsvPath]       = useState("Train.csv");
  const [nTrials, setNTrials]       = useState(50);
  const [cvFolds, setCvFolds]       = useState(5);
  const [triggering, setTriggering] = useState(false);
  const [triggerResult, setTriggerResult] = useState(null);
  const [triggerError, setTriggerError]     = useState(null);

  // task polling state
  const [pollingTaskId, setPollingTaskId] = useState(null);
  const [taskState, setTaskState]         = useState(null);

  const fetchStatus = useCallback(async () => {
    try {
      const url = pollingTaskId
        ? `${API_BASE}/api/hyperopt/status?task_id=${pollingTaskId}`
        : `${API_BASE}/api/hyperopt/status`;
      const r = await fetch(url);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const d = await r.json();
      setStatus(d.status);
      if (d.status?.task_state) setTaskState(d.status.task_state);
    } catch (e) {
      setError(e.message);
    }
  }, [pollingTaskId]);

  const fetchTrials = useCallback(async () => {
    try {
      const r = await fetch(`${API_BASE}/api/hyperopt/trials`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const d = await r.json();
      setTrials(d.trials || []);
    } catch (e) {
      // non-fatal
    }
  }, []);

  const loadAll = useCallback(async () => {
    setLoading(true);
    setError(null);
    await Promise.all([fetchStatus(), fetchTrials()]);
    setLoading(false);
  }, [fetchStatus, fetchTrials]);

  useEffect(() => { loadAll(); }, [loadAll]);

  // Auto-poll while a task is in flight
  useEffect(() => {
    if (!pollingTaskId) return;
    if (taskState === "SUCCESS" || taskState === "FAILURE") {
      setPollingTaskId(null);
      fetchTrials();
      fetchStatus();
      return;
    }
    const id = setInterval(fetchStatus, 3000);
    return () => clearInterval(id);
  }, [pollingTaskId, taskState, fetchStatus, fetchTrials]);

  const handleRefresh = async () => {
    setRefreshing(true);
    await Promise.all([fetchStatus(), fetchTrials()]);
    setRefreshing(false);
  };

  const handleTrigger = async () => {
    setTriggering(true);
    setTriggerResult(null);
    setTriggerError(null);
    try {
      const r = await fetch(
        `${API_BASE}/api/hyperopt/start?csv_path=${encodeURIComponent(csvPath)}&n_trials=${nTrials}&cv_folds=${cvFolds}`,
        { method: "POST" }
      );
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail || `HTTP ${r.status}`);
      setTriggerResult(d);
      if (d.task_id) {
        setPollingTaskId(d.task_id);
        setTaskState("PENDING");
      }
      fetchStatus();
    } catch (e) {
      setTriggerError(e.message);
    } finally {
      setTriggering(false);
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
        @keyframes pulse   { 0%,100% { opacity:1; } 50% { opacity:0.4; } }
        .tab-btn { background:none; border:none; cursor:pointer; padding:8px 18px;
          border-radius:8px; font-family:inherit; font-size:13px; color:#9ca3af; transition:all 0.15s; }
        .tab-btn:hover { background:#1f2937; }
        .tab-btn.active { background:#1f2937; color:#4ade80; }
        .trial-row:hover { background:#1a2332 !important; }
        .action-btn { border:none; padding:10px 24px; border-radius:8px; cursor:pointer;
          font-family:inherit; font-size:13px; font-weight:700; transition:all 0.15s; }
        .action-btn:disabled { opacity:0.45; cursor:not-allowed; }
      `}</style>

      <div style={{ maxWidth: 960, margin: "0 auto" }}>
        {/* Header */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 28 }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
              <span style={{ fontSize: 20, color: "#4ade80" }}>⚡</span>
              <h1 style={{ margin: 0, fontSize: 20, fontWeight: 700, color: "#f9fafb" }}>
                Hyperparameter Optimizer
              </h1>
            </div>
            <p style={{ margin: 0, fontSize: 12, color: "#6b7280" }}>
              Bayesian search with Optuna · Cross-validated · Benchmarked before promotion
            </p>
          </div>
          <button
            onClick={handleRefresh}
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
          {["overview", "optimize", "trials"].map(t => (
            <button key={t} className={`tab-btn${activeTab === t ? " active" : ""}`} onClick={() => setActiveTab(t)}>
              {{ overview: "Overview", optimize: "Run Optimization", trials: "Trial History" }[t]}
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

        {!loading && !error && status && (
          <>
            {/* ── OVERVIEW TAB ── */}
            {activeTab === "overview" && (
              <div style={{ animation: "fadeIn 0.25s ease" }}>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(170px,1fr))", gap: 12, marginBottom: 20 }}>
                  <Stat label="Config" value={status.config_exists ? "Ready" : "Missing"} color={status.config_exists ? "#4ade80" : "#f87171"} />
                  <Stat label="Best RMSE" value={status.best_rmse?.toFixed(4)} color="#4ade80" />
                  <Stat label="Best MAE"  value={status.best_mean_mae?.toFixed(4)} color="#38bdf8" />
                  <Stat label="Best R²"   value={status.best_mean_r2?.toFixed(4)} color="#fbbf24" />
                  <Stat label="Optimized" value={status.optimized_at?.replace("T", " ").slice(0, 19) ?? "Never"} color="#9ca3af" />
                </div>

                {/* Active task */}
                {pollingTaskId && (
                  <Card style={{ marginBottom: 16, borderColor: "#1a4a7a", background: "#0a1a2a" }}>
                    <SectionTitle>Active Optimization</SectionTitle>
                    <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                      {(taskState === "PENDING" || taskState === "PROGRESS") && (
                        <span style={{ animation: "pulse 1.5s ease infinite", color: "#38bdf8" }}><Spinner /></span>
                      )}
                      <div>
                        <div style={{ fontSize: 12, color: "#9ca3af", marginBottom: 4 }}>
                          Task ID: <span style={{ color: "#e5e7eb" }}>{pollingTaskId}</span>
                        </div>
                        <Badge status={taskState?.toLowerCase() === "success" ? "success" : taskState?.toLowerCase()} label={taskState} />
                        {status.task_info?.step && (
                          <span style={{ marginLeft: 10, fontSize: 12, color: "#6b7280" }}>
                            Step: {status.task_info.step}
                          </span>
                        )}
                        {status.task_info?.trials_completed !== undefined && (
                          <span style={{ marginLeft: 10, fontSize: 12, color: "#fbbf24" }}>
                            Trials: {status.task_info.trials_completed} / {status.task_info.trials_total}
                          </span>
                        )}
                        {status.task_info?.best_rmse && (
                          <span style={{ marginLeft: 10, fontSize: 12, color: "#4ade80" }}>
                            Best RMSE: {status.task_info.best_rmse.toFixed(4)}
                          </span>
                        )}
                      </div>
                    </div>
                  </Card>
                )}

                {/* Best params */}
                {status.best_params && (
                  <Card style={{ marginBottom: 16 }}>
                    <SectionTitle>Best Parameters</SectionTitle>
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: 8 }}>
                      {Object.entries(status.best_params).map(([k, v]) => (
                        <div key={k} style={{ background: "#0d1117", borderRadius: 6, padding: "8px 12px" }}>
                          <div style={{ fontSize: 10, color: "#6b7280", marginBottom: 3, textTransform: "uppercase", letterSpacing: "0.08em" }}>{k}</div>
                          <div style={{ fontSize: 13, color: "#e5e7eb" }}>{typeof v === "number" ? v.toFixed(4) : String(v)}</div>
                        </div>
                      ))}
                    </div>
                  </Card>
                )}

                {/* Benchmark from last completed run */}
                {trials.length > 0 && (
                  <BenchmarkCard benchmark={trials[0]?.benchmark || status.last_retraining_run?.benchmark} />
                )}

                <ParamImportanceChart trials={trials} />
                <TrialSparkline trials={trials} />
              </div>
            )}

            {/* ── OPTIMIZE TAB ── */}
            {activeTab === "optimize" && (
              <div style={{ animation: "fadeIn 0.25s ease", maxWidth: 520 }}>
                <Card>
                  <SectionTitle>Run Bayesian Optimization</SectionTitle>

                  <div style={{ marginBottom: 16 }}>
                    <label style={{ fontSize: 12, color: "#9ca3af", display: "block", marginBottom: 6 }}>
                      Training CSV Path
                    </label>
                    <input
                      value={csvPath}
                      onChange={e => setCsvPath(e.target.value)}
                      style={{
                        width: "100%", boxSizing: "border-box",
                        background: "#0d1117", border: "1px solid #1f2937",
                        borderRadius: 8, color: "#e5e7eb", fontFamily: "inherit",
                        fontSize: 13, padding: "10px 12px",
                      }}
                    />
                  </div>

                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 20 }}>
                    <div>
                      <label style={{ fontSize: 12, color: "#9ca3af", display: "block", marginBottom: 6 }}>
                        Trials (1–200)
                      </label>
                      <input
                        type="number"
                        min={1} max={200}
                        value={nTrials}
                        onChange={e => setNTrials(Math.min(200, Math.max(1, parseInt(e.target.value) || 1)))}
                        style={{
                          width: "100%", boxSizing: "border-box",
                          background: "#0d1117", border: "1px solid #1f2937",
                          borderRadius: 8, color: "#e5e7eb", fontFamily: "inherit",
                          fontSize: 13, padding: "10px 12px",
                        }}
                      />
                    </div>
                    <div>
                      <label style={{ fontSize: 12, color: "#9ca3af", display: "block", marginBottom: 6 }}>
                        CV Folds (2–10)
                      </label>
                      <input
                        type="number"
                        min={2} max={10}
                        value={cvFolds}
                        onChange={e => setCvFolds(Math.min(10, Math.max(2, parseInt(e.target.value) || 2)))}
                        style={{
                          width: "100%", boxSizing: "border-box",
                          background: "#0d1117", border: "1px solid #1f2937",
                          borderRadius: 8, color: "#e5e7eb", fontFamily: "inherit",
                          fontSize: 13, padding: "10px 12px",
                        }}
                      />
                    </div>
                  </div>

                  <button
                    className="action-btn"
                    onClick={handleTrigger}
                    disabled={triggering || pollingTaskId}
                    style={{
                      width: "100%",
                      background: triggering ? "#1a2a1a" : "#14532d",
                      border: "1px solid #166534", color: "#4ade80",
                    }}
                  >
                    {triggering ? <><Spinner /> &nbsp;Queuing…</> : "▶  Start Optimization"}
                  </button>

                  {pollingTaskId && !triggering && (
                    <div style={{ marginTop: 10, fontSize: 11, color: "#6b7280" }}>
                      An optimization is already running. Switch to Overview to monitor.
                    </div>
                  )}

                  {triggerResult && (
                    <div style={{
                      marginTop: 14, padding: "12px 14px", borderRadius: 8, animation: "fadeIn 0.2s ease",
                      background: triggerResult.triggered ? "#0f2a1a" : "#2a1f0a",
                      border: `1px solid ${triggerResult.triggered ? "#1a5c30" : "#7a4f00"}`,
                    }}>
                      <div style={{ fontSize: 13, color: triggerResult.triggered ? "#4ade80" : "#fbbf24", marginBottom: 4 }}>
                        {triggerResult.triggered ? "✓ Task queued successfully" : "○ Not triggered"}
                      </div>
                      {triggerResult.task_id && (
                        <div style={{ fontSize: 12, color: "#9ca3af" }}>
                          Task ID: <span style={{ color: "#e5e7eb" }}>{triggerResult.task_id}</span>
                        </div>
                      )}
                      {triggerResult.message && (
                        <div style={{ fontSize: 11, color: "#6b7280", marginTop: 6 }}>
                          {triggerResult.message}
                        </div>
                      )}
                    </div>
                  )}
                  {triggerError && (
                    <p style={{ margin: "12px 0 0", color: "#f87171", fontSize: 12 }}>⚠ {triggerError}</p>
                  )}
                </Card>
              </div>
            )}

            {/* ── TRIALS TAB ── */}
            {activeTab === "trials" && (
              <div style={{ animation: "fadeIn 0.25s ease" }}>
                <Card>
                  <SectionTitle>Trial History ({trials.length})</SectionTitle>
                  {trials.length === 0 ? (
                    <p style={{ color: "#4b5563", fontSize: 13 }}>
                      No trials yet. Run an optimization to see results here.
                    </p>
                  ) : (
                    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                      {trials.map((trial, i) => (
                        <div
                          key={i}
                          className="trial-row"
                          style={{
                            background: "#0d1117",
                            border: "1px solid #1f2937",
                            borderRadius: 8, padding: "12px 16px",
                            animation: `fadeIn 0.18s ease ${Math.min(i, 10) * 0.03}s both`,
                          }}
                        >
                          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                              <span style={{ fontSize: 11, color: "#4b5563", fontWeight: 700 }}>
                                TRIAL {trial.trial_number}
                              </span>
                              <span style={{ fontSize: 11, color: "#fbbf24" }}>
                                RMSE {trial.rmse?.toFixed(4)}
                              </span>
                            </div>
                            <span style={{ fontSize: 11, color: "#4b5563" }}>
                              {trial.duration_ms ? `${trial.duration_ms}ms` : ""}
                            </span>
                          </div>
                          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(100px, 1fr))", gap: 6 }}>
                            {Object.entries(trial.params || {}).map(([k, v]) => (
                              <div key={k} style={{ fontSize: 11, color: "#6b7280" }}>
                                <span style={{ textTransform: "uppercase", letterSpacing: "0.05em" }}>{k}: </span>
                                <span style={{ color: "#9ca3af" }}>{typeof v === "number" ? v.toFixed(3) : String(v)}</span>
                              </div>
                            ))}
                          </div>
                          <div style={{ display: "flex", gap: 12, marginTop: 6, fontSize: 11, color: "#4b5563" }}>
                            <span>MAE: {trial.mean_mae?.toFixed(4) ?? "—"}</span>
                            <span>R²: {trial.mean_r2?.toFixed(4) ?? "—"}</span>
                            <span>Folds: {trial.fold_scores?.length ?? 0}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </Card>
                <TrialSparkline trials={trials} />
                <ParamImportanceChart trials={trials} />
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}