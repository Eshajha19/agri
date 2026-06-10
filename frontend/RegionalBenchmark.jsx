import { useState, useEffect, useCallback } from "react";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

const REGIONS = ["Maharashtra", "Punjab", "Haryana", "Karnataka", "Tamil Nadu", "Telangana", "Gujarat", "Rajasthan", "Uttar Pradesh", "Other"];
const CROPS = ["Wheat", "Rice", "Cotton", "Sugarcane", "Maize", "Soybean", "Potato", "Onion", "Tomato", "Vegetables", "Fruits", "Other"];

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

// ── percentile gauge ────────────────────────────────────────────────────────
const PercentileGauge = ({ percentile }) => {
  if (percentile == null) return null;
  const pct = Math.min(100, Math.max(0, percentile));
  const color = pct >= 75 ? "#4ade80" : pct >= 50 ? "#fbbf24" : pct >= 25 ? "#f59e0b" : "#f87171";

  return (
    <Card style={{ marginTop: 16 }}>
      <SectionTitle>Percentile Rank</SectionTitle>
      <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
        <svg width={120} height={120} viewBox="0 0 100 100">
          <circle cx={50} cy={50} r={40} fill="none" stroke="#1f2937" strokeWidth={8} />
          <circle
            cx={50} cy={50} r={40} fill="none" stroke={color} strokeWidth={8}
            strokeDasharray={`${pct * 2.513} ${(100 - pct) * 2.513}`}
            strokeDashoffset={-75}
            transform="rotate(-90 50 50)"
            style={{ transition: "stroke-dasharray 0.5s ease" }}
          />
          <text x={50} y={48} textAnchor="middle" fill={color} fontSize="18" fontWeight="700" fontFamily="monospace">
            {pct.toFixed(0)}%
          </text>
          <text x={50} y={62} textAnchor="middle" fill="#6b7280" fontSize="8" fontFamily="monospace">
            PERCENTILE
          </text>
        </svg>
        <div style={{ fontSize: 13, color: "#9ca3af", lineHeight: 1.6 }}>
          <div style={{ color: "#e5e7eb", fontWeight: 700, fontSize: 16, marginBottom: 4 }}>
            {pct >= 75 ? "Top Performer 🏆" : pct >= 50 ? "Above Average" : pct >= 25 ? "Average" : "Below Average"}
          </div>
          You rank higher than {pct.toFixed(1)}% of peers in your region and crop category.
        </div>
      </div>
    </Card>
  );
};

// ── significance badge ───────────────────────────────────────────────────────
const SignificanceBadge = ({ result }) => {
  if (!result) return null;
  const { overall_significant, interpretation } = result;
  const color = overall_significant ? (interpretation === "significantly_above" ? "#4ade80" : "#f87171") : "#9ca3af";
  const bg = overall_significant ? (interpretation === "significantly_above" ? "#0f2a1a" : "#2a0a0a") : "#1a1a1a";
  const border = overall_significant ? (interpretation === "significantly_above" ? "#1a5c30" : "#7a1a1a") : "#374151";

  return (
    <div style={{
      background: bg, border: `1px solid ${border}`, borderRadius: 8,
      padding: "12px 16px", marginTop: 16,
    }}>
      <div style={{ fontSize: 13, color, fontWeight: 700, marginBottom: 4 }}>
        {overall_significant ? (interpretation === "significantly_above" ? "✓ Significantly Above Regional Baseline" : "⚠ Significantly Below Regional Baseline") : "○ Not Statistically Significant"}
      </div>
      <div style={{ fontSize: 11, color: "#6b7280" }}>
        t-test p-value: {result.t_test?.p_value?.toFixed(6)} · Mann-Whitney U p-value: {result.mann_whitney_u?.p_value?.toFixed(6)}
      </div>
    </div>
  );
};

// ── regional heatmap table ──────────────────────────────────────────────────
const RegionalHeatmap = ({ cohorts }) => {
  if (!cohorts || Object.keys(cohorts).length === 0) return null;

  const entries = Object.values(cohorts);
  const maxMean = Math.max(...entries.map(c => c.mean_yield || 0));

  return (
    <Card style={{ marginTop: 16 }}>
      <SectionTitle>Regional Heatmap</SectionTitle>
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {entries.map((c, i) => {
          const intensity = (c.mean_yield || 0) / maxMean;
          const bg = `rgba(74, 222, 128, ${intensity * 0.15})`;
          return (
            <div key={i} style={{
              display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr 1fr",
              gap: 12, alignItems: "center", padding: "10px 14px",
              background: bg, borderRadius: 6, fontSize: 12, color: "#9ca3af",
              border: `1px solid ${intensity > 0.7 ? 'rgba(74,222,128,0.3)' : 'transparent'}`,
            }}>
              <span style={{ color: "#e5e7eb", fontWeight: 700 }}>{c.region}</span>
              <span>{c.crop_type}</span>
              <span style={{ color: "#4ade80" }}>₹{c.mean_yield?.toFixed(0)}</span>
              <span>n={c.sample_size}</span>
              <span>CV: {c.coefficient_of_variation?.toFixed(2)}</span>
            </div>
          );
        })}
      </div>
    </Card>
  );
};

// ── main component ──────────────────────────────────────────────────────────
export default function RegionalBenchmark() {
  const [region, setRegion]       = useState("Maharashtra");
  const [cropType, setCropType]   = useState("Wheat");
  const [farmerYield, setFarmerYield] = useState("");
  const [activeTab, setActiveTab] = useState("stats");
  const [loading, setLoading]     = useState(false);
  const [stats, setStats]         = useState(null);
  const [percentile, setPercentile] = useState(null);
  const [significance, setSignificance] = useState(null);
  const [reportLoading, setReportLoading] = useState(false);
  const [reportResult, setReportResult] = useState(null);

  const fetchStats = useCallback(async () => {
    try {
      const r = await fetch(`${API_BASE}/api/benchmark/regional-stats?region=${encodeURIComponent(region)}`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const d = await r.json();
      setStats(d.data);
    } catch (e) {
      console.error("Stats fetch failed:", e);
    }
  }, [region]);

  const loadAll = useCallback(async () => {
    setLoading(true);
    await fetchStats();
    setLoading(false);
  }, [fetchStats]);

  useEffect(() => { loadAll(); }, [loadAll]);

  const handlePercentile = async () => {
    const fy = parseFloat(farmerYield);
    if (!fy || fy <= 0) return alert("Enter a valid yield");
    try {
      const r = await fetch(`${API_BASE}/api/benchmark/farmer-percentile`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ farmer_yield: fy, region, crop_type: cropType }),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail || `HTTP ${r.status}`);
      setPercentile(d.data);
      setSignificance(null);
    } catch (e) {
      alert(`Failed: ${e.message}`);
    }
  };

  const handleSignificance = async () => {
    const fy = parseFloat(farmerYield);
    if (!fy || fy <= 0) return alert("Enter a valid yield");
    try {
      const r = await fetch(`${API_BASE}/api/benchmark/significance-test`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ farmer_yield: fy, region, crop_type: cropType }),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail || `HTTP ${r.status}`);
      setSignificance(d.data);
      setPercentile(null);
    } catch (e) {
      alert(`Failed: ${e.message}`);
    }
  };

  const handleGenerateReport = async () => {
    const fy = parseFloat(farmerYield);
    if (!fy || fy <= 0) return alert("Enter a valid yield");
    setReportLoading(true);
    setReportResult(null);
    try {
      const r = await fetch(`${API_BASE}/api/benchmark/report/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ farmer_yield: fy, region, crop_type: cropType }),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail || `HTTP ${r.status}`);
      setReportResult(d);
    } catch (e) {
      alert(`Failed: ${e.message}`);
    } finally {
      setReportLoading(false);
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
              <span style={{ fontSize: 20, color: "#4ade80" }}>📊</span>
              <h1 style={{ margin: 0, fontSize: 20, fontWeight: 700, color: "#f9fafb" }}>
                Regional Benchmark
              </h1>
            </div>
            <p style={{ margin: 0, fontSize: 12, color: "#6b7280" }}>
              Federated yield intelligence · Statistical significance · Peer comparison
            </p>
          </div>
        </div>

        {/* Controls */}
        <Card style={{ marginBottom: 16 }}>
          <div style={{ display: "flex", gap: 12, flexWrap: "wrap", alignItems: "flex-end" }}>
            <div>
              <label style={{ fontSize: 11, color: "#6b7280", display: "block", marginBottom: 6, textTransform: "uppercase" }}>Region</label>
              <select value={region} onChange={e => setRegion(e.target.value)} style={{ background: "#0d1117", border: "1px solid #1f2937", borderRadius: 8, color: "#e5e7eb", fontFamily: "inherit", fontSize: 13, padding: "8px 12px" }}>
                {REGIONS.map(r => <option key={r} value={r}>{r}</option>)}
              </select>
            </div>
            <div>
              <label style={{ fontSize: 11, color: "#6b7280", display: "block", marginBottom: 6, textTransform: "uppercase" }}>Crop</label>
              <select value={cropType} onChange={e => setCropType(e.target.value)} style={{ background: "#0d1117", border: "1px solid #1f2937", borderRadius: 8, color: "#e5e7eb", fontFamily: "inherit", fontSize: 13, padding: "8px 12px" }}>
                {CROPS.map(c => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>
            <div>
              <label style={{ fontSize: 11, color: "#6b7280", display: "block", marginBottom: 6, textTransform: "uppercase" }}>Your Yield (₹/qtl proxy)</label>
              <input
                type="number"
                value={farmerYield}
                onChange={e => setFarmerYield(e.target.value)}
                placeholder="e.g. 2500"
                style={{ width: 140, background: "#0d1117", border: "1px solid #1f2937", borderRadius: 8, color: "#e5e7eb", fontFamily: "inherit", fontSize: 13, padding: "8px 12px" }}
              />
            </div>
            <div style={{ flex: 1 }} />
            <button className="action-btn" onClick={handlePercentile} style={{ background: "#1e3a5f", border: "1px solid #1a4a7a", color: "#7dd3fc" }}>
              📊 Percentile
            </button>
            <button className="action-btn" onClick={handleSignificance} style={{ background: "#2a1f0a", border: "1px solid #7a4f00", color: "#fbbf24" }}>
              ⚡ Significance
            </button>
            <button className="action-btn" onClick={handleGenerateReport} disabled={reportLoading} style={{ background: "#14532d", border: "1px solid #166534", color: "#4ade80" }}>
              {reportLoading ? <><Spinner /> Generating…</> : "📄 Generate Report"}
            </button>
          </div>
        </Card>

        {/* Tabs */}
        <div style={{ display: "flex", gap: 4, marginBottom: 24, borderBottom: "1px solid #1f2937", paddingBottom: 8 }}>
          {["stats", "comparison"].map(t => (
            <button key={t} className={`tab-btn${activeTab === t ? " active" : ""}`} onClick={() => setActiveTab(t)}>
              {{ stats: "Regional Stats", comparison: "My Comparison" }[t]}
            </button>
          ))}
        </div>

        {loading && (
          <div style={{ textAlign: "center", padding: 60, color: "#6b7280" }}>
            <Spinner /> <span style={{ marginLeft: 10 }}>Loading…</span>
          </div>
        )}

        {!loading && activeTab === "stats" && (
          <div style={{ animation: "fadeIn 0.25s ease" }}>
            <RegionalHeatmap cohorts={stats?.cohorts} />
          </div>
        )}

        {!loading && activeTab === "comparison" && (
          <div style={{ animation: "fadeIn 0.25s ease" }}>
            {percentile && (
              <>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(170px,1fr))", gap: 12, marginBottom: 16 }}>
                  <Stat label="Your Yield" value={`₹${percentile.farmer_yield?.toFixed(0)}`} color="#e5e7eb" />
                  <Stat label="Cohort Size" value={percentile.cohort_size} color="#38bdf8" />
                  <Stat label="Cohort Mean" value={`₹${percentile.cohort_mean?.toFixed(0)}`} color="#fbbf24" />
                  <Stat label="Cohort Median" value={`₹${percentile.cohort_median?.toFixed(0)}`} color="#a78bfa" />
                </div>
                <PercentileGauge percentile={percentile.percentile} />
              </>
            )}
            <SignificanceBadge result={significance} />
            {reportResult && (
              <Card style={{ marginTop: 16, borderColor: "#1a5c30", background: "#0f2a1a" }}>
                <SectionTitle>Report Generated</SectionTitle>
                <div style={{ fontSize: 13, color: "#4ade80", marginBottom: 8 }}>
                  ✓ Report ID: {reportResult.report_id}
                </div>
                <a
                  href={`${API_BASE}${reportResult.download_url}`}
                  target="_blank"
                  rel="noreferrer"
                  style={{
                    display: "inline-block", padding: "8px 16px", borderRadius: 6,
                    background: "#166534", color: "#4ade80", textDecoration: "none",
                    fontSize: 12, fontWeight: 700,
                  }}
                >
                  ↓ Download PDF
                </a>
              </Card>
            )}
          </div>
        )}
      </div>
    </div>
  );
}