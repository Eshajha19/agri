import { useState, useEffect, useCallback } from "react";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

// ── colour tokens ─────────────────────────────────────────────────────────────
const PRIORITY_COLORS = {
  high:   { bg: "#2a0a0a", border: "#7a1a1a", text: "#f87171", dot: "#ef4444" },
  medium: { bg: "#2a1f0a", border: "#7a4f00", text: "#fbbf24", dot: "#f59e0b" },
  low:    { bg: "#0f2a1a", border: "#1a5c30", text: "#4ade80", dot: "#22c55e" },
};
const pc = (p) => PRIORITY_COLORS[p?.toLowerCase()] || PRIORITY_COLORS.low;

const CATEGORY_ICONS = {
  crop: "🌾",
  performance: "📈",
  seasonal: "🌤️",
  general: "💡",
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

const Badge = ({ priority }) => {
  const c = pc(priority);
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 5,
      padding: "2px 10px", borderRadius: 999,
      background: c.bg, border: `1px solid ${c.border}`,
      color: c.text, fontSize: 11, fontWeight: 700,
      letterSpacing: "0.06em", textTransform: "uppercase",
    }}>
      <span style={{ width: 6, height: 6, borderRadius: "50%", background: c.dot }} />
      {priority}
    </span>
  );
};

const Stat = ({ label, value, color = "#e5e7eb", sub }) => (
  <Card>
    <div style={{ fontSize: 11, color: "#6b7280", marginBottom: 8, letterSpacing: "0.08em", textTransform: "uppercase" }}>
      {label}
    </div>
    <div style={{ fontSize: 22, fontWeight: 700, color }}>{value ?? "—"}</div>
    {sub && <div style={{ fontSize: 11, color: "#4b5563", marginTop: 4 }}>{sub}</div>}
  </Card>
);

// ── cluster donut chart ───────────────────────────────────────────────────────
const ClusterDonut = ({ clusters }) => {
  if (!clusters || clusters.length === 0) return null;
  const total = clusters.reduce((s, c) => s + (c.size || 0), 0);
  const colors = ["#4ade80", "#38bdf8", "#fbbf24", "#f87171", "#a78bfa", "#2dd4bf"];
  let cumulative = 0;

  return (
    <Card style={{ marginTop: 16 }}>
      <SectionTitle>Cluster Distribution</SectionTitle>
      <div style={{ display: "flex", alignItems: "center", gap: 24 }}>
        <svg width={140} height={140} viewBox="0 0 100 100">
          <circle cx={50} cy={50} r={40} fill="#0d1117" />
          {clusters.map((c, i) => {
            const pct = (c.size / total) * 100;
            const offset = cumulative;
            cumulative += pct;
            return (
              <circle
                key={c.cluster_id}
                cx={50} cy={50} r={40}
                fill="none"
                stroke={colors[i % colors.length]}
                strokeWidth={12}
                strokeDasharray={`${pct * 2.513} ${(100 - pct) * 2.513}`}
                strokeDashoffset={-offset * 2.513}
                transform="rotate(-90 50 50)"
              />
            );
          })}
          <text x={50} y={48} textAnchor="middle" fill="#e5e7eb" fontSize="14" fontWeight="700" fontFamily="monospace">
            {total}
          </text>
          <text x={50} y={58} textAnchor="middle" fill="#6b7280" fontSize="6" fontFamily="monospace">
            FARMERS
          </text>
        </svg>
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {clusters.map((c, i) => (
            <div key={c.cluster_id} style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12, color: "#9ca3af" }}>
              <span style={{ width: 10, height: 10, borderRadius: "50%", background: colors[i % colors.length] }} />
              Cluster {c.cluster_id}: {c.size} farmers ({c.top_crop})
            </div>
          ))}
        </div>
      </div>
    </Card>
  );
};

// ── peer benchmark table ──────────────────────────────────────────────────────
const PeerBenchmark = ({ benchmark }) => {
  if (!benchmark) return null;
  const { peers, my_rank } = benchmark;

  return (
    <Card style={{ marginTop: 16 }}>
      <SectionTitle>Peer Benchmark</SectionTitle>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(140px,1fr))", gap: 12, marginBottom: 16 }}>
        <Stat label="Cluster Size" value={benchmark.cluster_size} color="#38bdf8" />
        <Stat label="Top Crop" value={benchmark.top_crop} color="#4ade80" />
        <Stat label="Cluster Avg Yield" value={benchmark.cluster_mean_yield?.toFixed(1)} color="#fbbf24" />
        <Stat label="Your Rank" value={my_rank ? `${my_rank.rank}/${my_rank.total}` : "—"} color="#a78bfa" />
      </div>
      {peers && peers.length > 0 && (
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <div style={{ fontSize: 10, color: "#6b7280", marginBottom: 4, textTransform: "uppercase", letterSpacing: "0.08em" }}>
            Top Peers
          </div>
          {peers.slice(0, 5).map((peer, i) => (
            <div key={peer.uid} style={{
              display: "flex", justifyContent: "space-between", alignItems: "center",
              background: "#0d1117", borderRadius: 6, padding: "8px 12px",
              fontSize: 12, color: "#9ca3af",
            }}>
              <span>{peer.display_name || `Peer ${i + 1}`} <span style={{ color: "#4b5563" }}>({peer.crop_type})</span></span>
              <span style={{ color: "#e5e7eb", fontWeight: 700 }}>{peer.mean_yield_proxy?.toFixed(1)}</span>
            </div>
          ))}
        </div>
      )}
    </Card>
  );
};

// ── gap analysis alert ────────────────────────────────────────────────────────
const GapAlert = ({ gap }) => {
  if (!gap || !gap.significant) return null;
  return (
    <Card style={{ marginTop: 16, borderColor: "#7a4f00", background: "#2a1f0a" }}>
      <SectionTitle>⚠ Performance Gap Detected</SectionTitle>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12, marginBottom: 12 }}>
        <div>
          <div style={{ fontSize: 10, color: "#6b7280", marginBottom: 4, textTransform: "uppercase" }}>Your Yield</div>
          <div style={{ fontSize: 16, fontWeight: 700, color: "#e5e7eb" }}>{gap.farmer_yield?.toFixed(1)}</div>
        </div>
        <div>
          <div style={{ fontSize: 10, color: "#6b7280", marginBottom: 4, textTransform: "uppercase" }}>Top 20% Avg</div>
          <div style={{ fontSize: 16, fontWeight: 700, color: "#4ade80" }}>{gap.cluster_top_20_mean?.toFixed(1)}</div>
        </div>
        <div>
          <div style={{ fontSize: 10, color: "#6b7280", marginBottom: 4, textTransform: "uppercase" }}>Gap</div>
          <div style={{ fontSize: 16, fontWeight: 700, color: "#f87171" }}>↓ {gap.gap?.toFixed(1)}</div>
        </div>
      </div>
      {gap.actions?.map((a, i) => (
        <div key={i} style={{
          background: "#0d1117", borderRadius: 6, padding: "10px 14px", marginTop: 8,
          borderLeft: `3px solid ${a.priority === 'high' ? '#ef4444' : '#f59e0b'}`,
        }}>
          <div style={{ fontSize: 12, color: "#fbbf24", fontWeight: 700, marginBottom: 4 }}>
            {a.priority.toUpperCase()}: {a.action}
          </div>
          <div style={{ fontSize: 11, color: "#6b7280" }}>Impact: {a.impact}</div>
        </div>
      ))}
    </Card>
  );
};

// ── recommendation card ─────────────────────────────────────────────────────
const RecommendationCard = ({ rec, index }) => {
  const c = pc(rec.priority);
  return (
    <div style={{
      background: "#0d1117",
      border: `1px solid ${c.border}`,
      borderRadius: 8, padding: "14px 16px",
      animation: `fadeIn 0.18s ease ${Math.min(index, 10) * 0.05}s both`,
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <span style={{ fontSize: 16 }}>{CATEGORY_ICONS[rec.category] || "💡"}</span>
          <Badge priority={rec.priority} />
          <span style={{ fontSize: 10, color: "#4b5563", textTransform: "uppercase", letterSpacing: "0.05em" }}>
            {rec.category}
          </span>
        </div>
        {rec.predicted_yield_impact && (
          <span style={{ fontSize: 11, color: "#4ade80", fontWeight: 700 }}>
            +{rec.predicted_yield_impact} kg/acre
          </span>
        )}
      </div>
      <h4 style={{ margin: "0 0 6px", fontSize: 14, color: "#e5e7eb", fontWeight: 600 }}>
        {rec.title}
      </h4>
      <p style={{ margin: 0, fontSize: 12, color: "#9ca3af", lineHeight: 1.5 }}>
        {rec.text}
      </p>
      <div style={{ display: "flex", gap: 12, marginTop: 8, fontSize: 11, color: "#4b5563" }}>
        <span>Impact: {rec.impact}</span>
        <span>Confidence: {rec.confidence}</span>
      </div>
    </div>
  );
};

// ── main component ────────────────────────────────────────────────────────────
export default function PersonalizedAdvisory() {
  const [profile, setProfile]       = useState(null);
  const [segments, setSegments]     = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [benchmark, setBenchmark]   = useState(null);
  const [gap, setGap]               = useState(null);
  const [activeTab, setActiveTab]   = useState("my_advisory");
  const [loading, setLoading]       = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError]           = useState(null);

  const fetchMyAdvisory = useCallback(async () => {
    try {
      const r = await fetch(`${API_BASE}/api/advisory/personalized`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const d = await r.json();
      setRecommendations(d.recommendations || []);
      setBenchmark(d.benchmark);
      setGap(d.gap_analysis);
      setProfile({ uid: d.uid, cluster_id: d.cluster_id });
    } catch (e) {
      setError(e.message);
    }
  }, []);

  const fetchSegments = useCallback(async () => {
    try {
      const r = await fetch(`${API_BASE}/api/advisory/segments`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const d = await r.json();
      setSegments(d.clusters || []);
    } catch (e) {
      // non-fatal
    }
  }, []);

  const loadAll = useCallback(async () => {
    setLoading(true);
    setError(null);
    await Promise.all([fetchMyAdvisory(), fetchSegments()]);
    setLoading(false);
  }, [fetchMyAdvisory, fetchSegments]);

  useEffect(() => { loadAll(); }, [loadAll]);

  const handleRefresh = async () => {
    setRefreshing(true);
    await Promise.all([fetchMyAdvisory(), fetchSegments()]);
    setRefreshing(false);
  };

  const handleSegmentRefresh = async () => {
    try {
      const r = await fetch(`${API_BASE}/api/advisory/segments/refresh`, {
        method: "POST",
        credentials: "include",
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail || `HTTP ${r.status}`);
      alert(`Refresh queued. Task ID: ${d.task_id}`);
    } catch (e) {
      alert(`Failed: ${e.message}`);
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
              <span style={{ fontSize: 20, color: "#4ade80" }}>🎯</span>
              <h1 style={{ margin: 0, fontSize: 20, fontWeight: 700, color: "#f9fafb" }}>
                Personalized Advisory
              </h1>
            </div>
            <p style={{ margin: 0, fontSize: 12, color: "#6b7280" }}>
              Cluster-based segmentation · Peer benchmarking · Dynamic recommendations
            </p>
          </div>
          <div style={{ display: "flex", gap: 8 }}>
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
            <button
              onClick={handleSegmentRefresh}
              style={{
                display: "flex", alignItems: "center", gap: 7,
                background: "#1e3a5f", border: "1px solid #1a4a7a",
                color: "#7dd3fc", padding: "8px 16px", borderRadius: 8,
                cursor: "pointer", fontFamily: "inherit", fontSize: 12,
              }}
            >
              ⟳ Recompute Clusters
            </button>
          </div>
        </div>

        {/* Tabs */}
        <div style={{ display: "flex", gap: 4, marginBottom: 24, borderBottom: "1px solid #1f2937", paddingBottom: 8 }}>
          {["my_advisory", "segments"].map(t => (
            <button key={t} className={`tab-btn${activeTab === t ? " active" : ""}`} onClick={() => setActiveTab(t)}>
              {{ my_advisory: "My Advisory", segments: "All Segments" }[t]}
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

        {!loading && !error && (
          <>
            {/* ── MY ADVISORY TAB ── */}
            {activeTab === "my_advisory" && (
              <div style={{ animation: "fadeIn 0.25s ease" }}>
                {profile && (
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(170px,1fr))", gap: 12, marginBottom: 16 }}>
                    <Stat label="Cluster ID" value={profile.cluster_id} color="#a78bfa" />
                    <Stat label="Recommendations" value={recommendations.length} color="#4ade80" />
                    <Stat label="High Priority" value={recommendations.filter(r => r.priority === "high").length} color="#f87171" />
                    <Stat label="Yield Impact" value={recommendations.filter(r => r.predicted_yield_impact).length} color="#38bdf8" sub="scored" />
                  </div>
                )}

                <PeerBenchmark benchmark={benchmark} />
                <GapAlert gap={gap} />

                <Card style={{ marginTop: 16 }}>
                  <SectionTitle>Recommendations ({recommendations.length})</SectionTitle>
                  {recommendations.length === 0 ? (
                    <p style={{ color: "#4b5563", fontSize: 13 }}>No personalized recommendations yet. Complete your profile and run farm intelligence to generate insights.</p>
                  ) : (
                    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                      {recommendations.map((rec, i) => (
                        <RecommendationCard key={i} rec={rec} index={i} />
                      ))}
                    </div>
                  )}
                </Card>
              </div>
            )}

            {/* ── SEGMENTS TAB ── */}
            {activeTab === "segments" && (
              <div style={{ animation: "fadeIn 0.25s ease" }}>
                <ClusterDonut clusters={segments} />
                <Card style={{ marginTop: 16 }}>
                  <SectionTitle>Cluster Details</SectionTitle>
                  <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                    {segments.map((seg, i) => (
                      <div key={seg.cluster_id} style={{
                        background: "#0d1117", borderRadius: 8, padding: "12px 16px",
                        display: "grid", gridTemplateColumns: "60px 1fr 1fr 1fr 1fr", gap: 12,
                        alignItems: "center", fontSize: 12, color: "#9ca3af",
                      }}>
                        <span style={{ fontWeight: 700, color: "#e5e7eb" }}>C{seg.cluster_id}</span>
                        <span>{seg.size} farmers</span>
                        <span style={{ color: "#4ade80" }}>Top: {seg.top_crop}</span>
                        <span>Avg Yield: {seg.mean_yield_proxy?.toFixed(1)}</span>
                        <span>Trend: {seg.mean_yield_trend > 0 ? "↗" : "↘"} {seg.mean_yield_trend?.toFixed(1)}</span>
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