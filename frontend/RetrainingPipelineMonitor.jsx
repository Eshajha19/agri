import { useState, useEffect, useCallback } from "react";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

// ── colour tokens ─────────────────────────────────────────────────────────────
const OUTCOME_COLORS = {
  promoted: { bg: "#0f2a1a", border: "#1a5c30", text: "#4ade80", dot: "#22c55e" },
  rejected: { bg: "#2a1f0a", border: "#7a4f00", text: "#fbbf24", dot: "#f59e0b" },
  failed:   { bg: "#2a0a0a", border: "#7a1a1a", text: "#f87171", dot: "#ef4444" },
  pending:  { bg: "#1a1a2a", border: "#3a3a7a", text: "#a5b4fc", dot: "#818cf8" },
  progress: { bg: "#0a1a2a", border: "#1a4a7a", text: "#7dd3fc", dot: "#38bdf8" },
};
const oc = (s) => OUTCOME_COLORS[s?.toLowerCase()] || OUTCOME_COLORS.pending;

// ── tiny shared components ────────────────────────────────────────────────────
const Badge = ({ status, label }) => {
  const c = oc(status);
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

const Stat = ({ label, value, color = "#e5e7eb" }) => (
  <Card>
    <div style={{ fontSize: 11, color: "#6b7280", marginBottom: 8, letterSpacing: "0.08em", textTransform: "uppercase" }}>
      {label}
    </div>
    <div style={{ fontSize: 22, fontWeight: 700, color }}>{value ?? "—"}</div>
  </Card>
);

// ── main component ────────────────────────────────────────────────────────────
export default function RetrainingPipelineMonitor() {
  const [pipeline, setPipeline]       = useState(null);
  const [history, setHistory]         = useState([]);
  const [activeTab, setActiveTab]     = useState("overview");
  const [loading, setLoading]         = useState(true);
  const [refreshing, setRefreshing]   = useState(false);
  const [error, setError]             = useState(null);

  // trigger form state
  const [force, setForce]             = useState(false);
  const [csvPath, setCsvPath]         = useState("Train.csv");
  const [triggering, setTriggering]   = useState(false);
  const [triggerResult, setTriggerResult] = useState(null);
  const [triggerError, setTriggerError]   = useState(null);

  // task polling state
  const [pollingTaskId, setPollingTaskId] = useState(null);
  const [taskStatus, setTaskStatus]       = useState(null);

  const fetchPipeline = useCallback(async () => {
    try {
      const url = pollingTaskId
        ? `${API_BASE}/api/retraining/status?task_id=${pollingTaskId}`
        : `${API_BASE}/api/retraining/status`;
      const r = await fetch(url);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const d = await r.json();
      setPipeline(d.pipeline);
      if (d.pipeline?.task_state) setTaskStatus(d.pipeline.task_state);
    } catch (e) {
      setError(e.message);
    }
  }, [pollingTaskId]);

  const fetchHistory = useCallback(async () => {
    try {
      const r = await fetch(`${API_BASE}/api/retraining/history?limit=30`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const d = await r.json();
      setHistory(d.runs || []);
    } catch (e) {
      // non-fatal
    }
  }, []);

  const loadAll = useCallback(async () => {
    setLoading(true);
    setError(null);
    await Promise.all([fetchPipeline(), fetchHistory()]);
    setLoading(false);
  }, [fetchPipeline, fetchHistory]);

  useEffect(() => { loadAll(); }, [loadAll]);

  // Auto-poll while a task is in flight
  useEffect(() => {
    if (!pollingTaskId) return;
    if (taskStatus === "SUCCESS" || taskStatus === "FAILURE") {
      setPollingTaskId(null);
      fetchHistory();
      return;
    }
    const id = setInterval(fetchPipeline, 3000);
    return () => clearInterval(id);
  }, [pollingTaskId, taskStatus, fetchPipeline, fetchHistory]);

  const handleRefresh = async () => {
    setRefreshing(true);
    await Promise.all([fetchPipeline(), fetchHistory()]);
    setRefreshing(false);
  };

  const handleTrigger = async () => {
    setTriggering(true);
    setTriggerResult(null);
    setTriggerError(null);
    try {
      const r = await fetch(
        `${API_BASE}/api/retraining/trigger?csv_path=${encodeURIComponent(csvPath)}&force=${force}`,
        { method: "POST" }
      );
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail || `HTTP ${r.status}`);
      setTriggerResult(d);
      if (d.task_id) {
        setPollingTaskId(d.task_id);
        setTaskStatus("PENDING");
      }
      fetchPipeline();
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
        .history-row:hover { background:#1a2332 !important; }
        .action-btn { border:none; padding:10px 24px; border-radius:8px; cursor:pointer;
          font-family:inherit; font-size:13px; font-weight:700; transition:all 0.15s; }
        .action-btn:disabled { opacity:0.45; cursor:not-allowed; }
        .toggle-box { display:flex; align-items:center; gap:10px; cursor:pointer; user-select:none; }
      `}</style>

      <div style={{ maxWidth: 960, margin: "0 auto" }}>
        {/* Header */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 28 }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
              <span style={{ fontSize: 20, color: "#4ade80" }}>⟳</span>
              <h1 style={{ margin: 0, fontSize: 20, fontWeight: 700, color: "#f9fafb" }}>
                Retraining Pipeline Monitor
              </h1>
            </div>
            <p style={{ margin: 0, fontSize: 12, color: "#6b7280" }}>
              Automated model retraining on drift breach · Fasal Saathi ML
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
          {["overview", "trigger", "history"].map(t => (
            <button key={t} className={`tab-btn${activeTab === t ? " active" : ""}`} onClick={() => setActiveTab(t)}>
              {{ overview: "Overview", trigger: "Trigger Retraining", history: "Run History" }[t]}
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

        {!loading && !error && pipeline && (
          <>
            {/* ── OVERVIEW TAB ── */}
            {activeTab === "overview" && (
              <div style={{ animation: "fadeIn 0.25s ease" }}>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(170px,1fr))", gap: 12, marginBottom: 20 }}>
                  <Stat label="Model"    value={pipeline.model_exists    ? "Ready" : "Missing"} color={pipeline.model_exists    ? "#4ade80" : "#f87171"} />
                  <Stat label="Baseline" value={pipeline.baseline_exists ? "Ready" : "Missing"} color={pipeline.baseline_exists ? "#4ade80" : "#f87171"} />
                  <Stat label="Drift Breach" value={pipeline.drift_threshold_breached ? "YES" : "No"} color={pipeline.drift_threshold_breached ? "#f87171" : "#4ade80"} />
                  <Stat label="Total Runs"   value={pipeline.total_runs ?? 0} />
                  <Stat label="Backup Ready" value={pipeline.backup_model_exists ? "Yes" : "No"} color={pipeline.backup_model_exists ? "#fbbf24" : "#6b7280"} />
                </div>

                {/* Drift status */}
                <Card style={{ marginBottom: 16 }}>
                  <SectionTitle>Drift Status</SectionTitle>
                  <div style={{
                    padding: "10px 14px", borderRadius: 8,
                    background: pipeline.drift_threshold_breached ? "#2a0a0a" : "#0f2a1a",
                    border: `1px solid ${pipeline.drift_threshold_breached ? "#7a1a1a" : "#1a5c30"}`,
                    fontSize: 13,
                    color: pipeline.drift_threshold_breached ? "#f87171" : "#4ade80",
                  }}>
                    {pipeline.drift_reason || "No drift data available."}
                  </div>
                </Card>

                {/* Active task polling */}
                {pollingTaskId && (
                  <Card style={{ marginBottom: 16, borderColor: "#1a4a7a", background: "#0a1a2a" }}>
                    <SectionTitle>Active Task</SectionTitle>
                    <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                      {(taskStatus === "PENDING" || taskStatus === "PROGRESS") && (
                        <span style={{ animation: "pulse 1.5s ease infinite", color: "#38bdf8" }}><Spinner /></span>
                      )}
                      <div>
                        <div style={{ fontSize: 12, color: "#9ca3af", marginBottom: 4 }}>
                          Task ID: <span style={{ color: "#e5e7eb" }}>{pollingTaskId}</span>
                        </div>
                        <Badge status={taskStatus?.toLowerCase() === "success" ? "promoted" : taskStatus?.toLowerCase()} label={taskStatus} />
                        {pipeline.task_info?.step && (
                          <span style={{ marginLeft: 10, fontSize: 12, color: "#6b7280" }}>
                            Step: {pipeline.task_info.step}
                          </span>
                        )}
                        {pipeline.task_info?.candidate_rmse && (
                          <span style={{ marginLeft: 10, fontSize: 12, color: "#fbbf24" }}>
                            Candidate RMSE: {pipeline.task_info.candidate_rmse.toFixed(4)}
                          </span>
                        )}
                      </div>
                    </div>
                  </Card>
                )}

                {/* Last run */}
                {pipeline.last_run && (
                  <Card>
                    <SectionTitle>Last Run</SectionTitle>
                    <div style={{ display: "flex", gap: 10, alignItems: "center", marginBottom: 10 }}>
                      <Badge status={pipeline.last_run.outcome} />
                      <span style={{ fontSize: 12, color: "#6b7280" }}>
                        {pipeline.last_run.triggered_at?.replace("T", " ").slice(0, 19)} UTC
                      </span>
                    </div>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                      {[
                        ["RMSE",          pipeline.last_run.rmse?.toFixed(4)],
                        ["Previous RMSE", pipeline.last_run.previous_rmse?.toFixed(4) ?? "First run"],
                        ["CSV",           pipeline.last_run.csv_path],
                        ["Completed",     pipeline.last_run.completed_at?.replace("T", " ").slice(0, 19)],
                      ].map(([k, v]) => (
                        <div key={k} style={{ background: "#0d1117", borderRadius: 6, padding: "8px 12px" }}>
                          <div style={{ fontSize: 10, color: "#6b7280", marginBottom: 3, textTransform: "uppercase", letterSpacing: "0.08em" }}>{k}</div>
                          <div style={{ fontSize: 13, color: "#e5e7eb" }}>{v ?? "—"}</div>
                        </div>
                      ))}
                    </div>
                  </Card>
                )}
              </div>
            )}

            {/* ── TRIGGER TAB ── */}
            {activeTab === "trigger" && (
              <div style={{ animation: "fadeIn 0.25s ease", maxWidth: 520 }}>
                <Card>
                  <SectionTitle>Trigger Retraining Job</SectionTitle>

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

                  <label className="toggle-box" style={{ marginBottom: 20 }}>
                    <div style={{
                      width: 36, height: 20, borderRadius: 999, position: "relative",
                      background: force ? "#166534" : "#374151", transition: "background 0.2s",
                      border: `1px solid ${force ? "#22c55e" : "#4b5563"}`,
                    }}>
                      <div style={{
                        position: "absolute", top: 2,
                        left: force ? 16 : 2,
                        width: 14, height: 14, borderRadius: "50%",
                        background: force ? "#4ade80" : "#9ca3af",
                        transition: "left 0.2s",
                      }} />
                    </div>
                    <input type="checkbox" checked={force} onChange={e => setForce(e.target.checked)} style={{ display: "none" }} />
                    <div>
                      <div style={{ fontSize: 13, color: "#e5e7eb" }}>Force retrain</div>
                      <div style={{ fontSize: 11, color: "#6b7280" }}>
                        {force ? "Bypass drift threshold check" : "Only retrain if drift threshold breached"}
                      </div>
                    </div>
                  </label>

                  {/* Drift warning */}
                  {!force && !pipeline.drift_threshold_breached && (
                    <div style={{
                      marginBottom: 16, padding: "10px 14px", borderRadius: 8,
                      background: "#2a1f0a", border: "1px solid #7a4f00",
                      fontSize: 12, color: "#fbbf24",
                    }}>
                      ⚠ Drift threshold not currently breached. Enable "Force retrain" to proceed anyway.
                    </div>
                  )}

                  <button
                    className="action-btn"
                    onClick={handleTrigger}
                    disabled={triggering}
                    style={{
                      width: "100%",
                      background: triggering ? "#1a2a1a" : "#14532d",
                      border: "1px solid #166534", color: "#4ade80",
                    }}
                  >
                    {triggering ? <><Spinner /> &nbsp;Queuing…</> : "▶  Queue Retraining Job"}
                  </button>

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
                      {triggerResult.reason && (
                        <div style={{ fontSize: 12, color: "#9ca3af", marginTop: 4 }}>{triggerResult.reason}</div>
                      )}
                      {triggerResult.task_id && (
                        <div style={{ fontSize: 11, color: "#6b7280", marginTop: 6 }}>
                          Switch to Overview tab to monitor task progress.
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

            {/* ── HISTORY TAB ── */}
            {activeTab === "history" && (
              <div style={{ animation: "fadeIn 0.25s ease" }}>
                <Card>
                  <SectionTitle>Run History ({history.length} shown)</SectionTitle>
                  {history.length === 0 ? (
                    <p style={{ color: "#4b5563", fontSize: 13 }}>
                      No runs yet. Trigger a retraining job to see records here.
                    </p>
                  ) : (
                    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                      {history.map((run, i) => {
                        const c = oc(run.outcome);
                        const improved = run.previous_rmse && run.rmse < run.previous_rmse;
                        return (
                          <div
                            key={i}
                            className="history-row"
                            style={{
                              background: "#0d1117",
                              border: `1px solid ${c.border}`,
                              borderRadius: 8, padding: "12px 16px",
                              animation: `fadeIn 0.18s ease ${Math.min(i, 10) * 0.03}s both`,
                            }}
                          >
                            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                              <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                                <Badge status={run.outcome} />
                                {improved && (
                                  <span style={{ fontSize: 11, color: "#4ade80" }}>
                                    ↓ {((run.previous_rmse - run.rmse) / run.previous_rmse * 100).toFixed(1)}% better
                                  </span>
                                )}
                              </div>
                              <span style={{ fontSize: 11, color: "#4b5563" }}>
                                {run.triggered_at?.replace("T", " ").slice(0, 19)} UTC
                              </span>
                            </div>
                            <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 6 }}>
                              {[
                                ["RMSE",     run.rmse?.toFixed(4)      ?? "—"],
                                ["Prev RMSE", run.previous_rmse?.toFixed(4) ?? "First"],
                                ["CSV",      run.csv_path ?? "—"],
                              ].map(([k, v]) => (
                                <div key={k} style={{ fontSize: 11, color: "#6b7280" }}>
                                  <span style={{ textTransform: "uppercase", letterSpacing: "0.05em" }}>{k}: </span>
                                  <span style={{ color: "#9ca3af" }}>{v}</span>
                                </div>
                              ))}
                            </div>
                            {run.error && (
                              <div style={{ marginTop: 6, fontSize: 11, color: "#f87171" }}>
                                Error: {run.error}
                              </div>
                            )}
                          </div>
                        );
                      })}
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