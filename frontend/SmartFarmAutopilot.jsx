import React, { useState } from "react";
import {
  Cpu, MapPin, Layers, Droplets, Wallet, Calendar,
  Sprout, BarChart2, FlaskConical, Zap, AlertTriangle,
  CheckCircle2, ChevronDown, ChevronUp, RefreshCw,
  TrendingUp, CloudRain, Leaf, ArrowRight, Info
} from "lucide-react";
import "./SmartFarmAutopilot.css";
import apiClient from "./services/api";

// apiClient (from services/api.js) automatically injects the Firebase auth
// token via its Axios request interceptor. Raw axios has no interceptor and
// would send requests without an Authorization header, causing 401 rejections
// from the authenticated /api/autopilot/generate-plan endpoint.
// API_BASE is no longer needed — apiClient uses relative paths so it works
// correctly in both local development (via Vite proxy) and production.

const STATES = [
  "Andhra Pradesh","Bihar","Gujarat","Haryana","Karnataka",
  "Madhya Pradesh","Maharashtra","Punjab","Rajasthan","Tamil Nadu",
  "Telangana","Uttar Pradesh","West Bengal",
];
const SOIL_TYPES = ["Alluvial","Black","Clay","Laterite","Loamy","Red","Sandy"];
const WATER_SOURCES = ["Canal","Borewell","Rainwater","River","Drip","Pond"];
const SEASONS = [
  { id: "Kharif", label: "Kharif (Jun–Oct)", icon: <CloudRain size={16}/> },
  { id: "Rabi",   label: "Rabi (Nov–Mar)",   icon: <Leaf size={16}/> },
  { id: "Zaid",   label: "Zaid (Feb–May)",   icon: <Zap size={16}/> },
];

// ── Small helpers ──────────────────────────────────────────────────────────

const Badge = ({ children, color = "green" }) => (
  <span className={`sfa-badge sfa-badge-${color}`}>{children}</span>
);

const SectionCard = ({ icon, title, children, defaultOpen = true }) => {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="sfa-section-card">
      <button className="sfa-section-header" onClick={() => setOpen(o => !o)}>
        <span className="sfa-section-title">{icon} {title}</span>
        {open ? <ChevronUp size={18}/> : <ChevronDown size={18}/>}
      </button>
      {open && <div className="sfa-section-body">{children}</div>}
    </div>
  );
};

const StatBox = ({ label, value, sub, color }) => (
  <div className={`sfa-stat-box sfa-stat-${color}`}>
    <div className="sfa-stat-value">{value}</div>
    <div className="sfa-stat-label">{label}</div>
    {sub && <div className="sfa-stat-sub">{sub}</div>}
  </div>
);

// ── Main component ─────────────────────────────────────────────────────────

export default function SmartFarmAutopilot() {
  const [form, setForm] = useState({
    farm_name: "", state: "Maharashtra", district: "",
    area_acres: "", soil_type: "Black", season: "Kharif",
    water_source: "Canal", budget_inr: "",
  });
  const [plan, setPlan]     = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]   = useState("");
  const [activeTab, setActiveTab] = useState("sowing");
  const mountedRef = React.useRef(true);
  const requestIdRef = React.useRef(0);
  const submitControllerRef = React.useRef(null);

  React.useEffect(() => {
    mountedRef.current = true;

    return () => {
      mountedRef.current = false;
      submitControllerRef.current?.abort();
    };
  }, []);

  const handleChange = (e) => {
    const { name, value } = e.target;

    requestIdRef.current++;

    submitControllerRef.current?.abort();

    setForm((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    const requestId = ++requestIdRef.current;

    submitControllerRef.current?.abort();

    const controller = new AbortController();
    submitControllerRef.current = controller;

    setError("");
    setLoading(true);
    setPlan(null);

    try {
      const payload = {
        ...form,
        area_acres: parseFloat(form.area_acres),
        budget_inr: form.budget_inr
          ? parseFloat(form.budget_inr)
          : null,
      };

      const { data } = await apiClient.post(
        "/api/autopilot/generate-plan",
        payload,
        {
          signal: controller.signal,
        }
      );

      if (
        !mountedRef.current ||
        requestId !== requestIdRef.current
      ) {
        return;
      }

      setPlan(data.plan);
      setActiveTab("sowing");
    } catch (err) {
      if (
        err.name === "AbortError" ||
        err.code === "ERR_CANCELED"
      ) {
        return;
      }

      if (
        mountedRef.current &&
        requestId === requestIdRef.current
      ) {
        setError(
          err.response?.data?.detail ||
          "Failed to generate plan. Please try again."
        );
      }
    } finally {
      if (
        mountedRef.current &&
        requestId === requestIdRef.current
      ) {
        setLoading(false);
      }
    }
  };

  const proj = plan?.yield_projection;

  return (
    <div className="sfa-page">
      {/* ── Header ── */}
      <div className="sfa-header">
        <div className="sfa-header-icon"><Cpu size={32}/></div>
        <div>
          <h1 className="sfa-title">Smart Farm Autopilot</h1>
          <p className="sfa-subtitle">
            Enter your farm details once — get a complete seasonal plan covering
            crop selection, sowing, irrigation, fertilizers, and profit projection.
          </p>
        </div>
      </div>

      <div className="sfa-layout">
        {/* ── Input Form ── */}
        <aside className="sfa-sidebar">
          <div className="sfa-form-card">
            <h3 className="sfa-form-title"><Info size={18}/> Farm Details</h3>
            <form onSubmit={handleSubmit} noValidate>

              <div className="sfa-field">
                <label>Farm Name (optional)</label>
                <input name="farm_name" value={form.farm_name}
                  onChange={handleChange} placeholder="e.g. Ramesh's Farm" />
              </div>

              <div className="sfa-field">
                <label><MapPin size={14}/> State *</label>
                <select name="state" value={form.state} onChange={handleChange} required>
                  {STATES.map(s => <option key={s}>{s}</option>)}
                </select>
              </div>

              <div className="sfa-field">
                <label><MapPin size={14}/> District</label>
                <input name="district" value={form.district}
                  onChange={handleChange} placeholder="e.g. Nashik" />
              </div>

              <div className="sfa-field">
                <label><Layers size={14}/> Farm Area (acres) *</label>
                <input name="area_acres" type="number" min="0.1" step="0.1"
                  value={form.area_acres} onChange={handleChange}
                  placeholder="e.g. 2.5" required />
              </div>

              <div className="sfa-field">
                <label><Layers size={14}/> Soil Type *</label>
                <select name="soil_type" value={form.soil_type} onChange={handleChange}>
                  {SOIL_TYPES.map(s => <option key={s}>{s}</option>)}
                </select>
              </div>

              <div className="sfa-field">
                <label><Calendar size={14}/> Season *</label>
                <div className="sfa-season-toggle">
                  {SEASONS.map(s => (
                    <button key={s.id} type="button"
                      className={`sfa-season-btn ${form.season === s.id ? "active" : ""}`}
                      onClick={() => {
                        requestIdRef.current++;

                        submitControllerRef.current?.abort();

                        setForm((p) => ({
                          ...p,
                          season: s.id,
                        }));
                      }}>
                      {s.icon} {s.id}
                    </button>
                  ))}
                </div>
              </div>

              <div className="sfa-field">
                <label><Droplets size={14}/> Water Source</label>
                <select name="water_source" value={form.water_source} onChange={handleChange}>
                  {WATER_SOURCES.map(w => <option key={w}>{w}</option>)}
                </select>
              </div>

              <div className="sfa-field">
                <label><Wallet size={14}/> Budget (₹, optional)</label>
                <input name="budget_inr" type="number" min="0"
                  value={form.budget_inr} onChange={handleChange}
                  placeholder="e.g. 50000" />
              </div>

              {error && (
                <div className="sfa-error">
                  <AlertTriangle size={15}/> {error}
                </div>
              )}

              <button type="submit" className="sfa-submit-btn" disabled={loading || !form.area_acres}>
                {loading
                  ? <><RefreshCw size={16} className="sfa-spin"/> Generating Plan…</>
                  : <><Cpu size={16}/> Generate Season Plan</>
                }
              </button>
            </form>
          </div>
        </aside>

        {/* ── Results ── */}
        <main className="sfa-main">
          {!plan && !loading && (
            <div className="sfa-empty">
              <Sprout size={64} className="sfa-empty-icon"/>
              <h2>Ready to plan your season?</h2>
              <p>Fill in your farm details on the left and click <strong>Generate Season Plan</strong> to get a complete end-to-end farming plan in seconds.</p>
              <div className="sfa-feature-pills">
                {["Crop Selection","Sowing Schedule","Irrigation Plan",
                  "Fertilizer Timeline","Yield Projection","Risk Analysis"].map(f => (
                  <span key={f} className="sfa-pill"><CheckCircle2 size={13}/> {f}</span>
                ))}
              </div>
            </div>
          )}

          {loading && (
            <div className="sfa-loading">
              <div className="sfa-loader-ring"/>
              <p>Analyzing soil, weather, and market data…</p>
            </div>
          )}

          {plan && !loading && (
            <div className="sfa-results">

              {/* Plan header */}
              <div className="sfa-plan-header">
                <div>
                  <div className="sfa-plan-id">Plan #{plan.plan_id}</div>
                  <h2 className="sfa-plan-crop">{plan.primary_crop}
                    <Badge color="green">{plan.season}</Badge>
                  </h2>
                  <p className="sfa-plan-summary">{plan.summary}</p>
                </div>
                <div className="sfa-crop-alts">
                  <span className="sfa-crop-alts-label">Also suitable:</span>
                  {plan.recommended_crops.slice(1).map(c => (
                    <span key={c} className="sfa-crop-chip">{c}</span>
                  ))}
                </div>
              </div>

              {/* Profit stats */}
              {proj && (
                <div className="sfa-stats-row">
                  <StatBox label="Expected Yield" color="green"
                    value={`${proj.expected_yield_kg.toLocaleString()} kg`}
                    sub={`${proj.min_yield_kg.toLocaleString()}–${proj.max_yield_kg.toLocaleString()} kg range`}/>
                  <StatBox label="Gross Revenue" color="blue"
                    value={`₹${proj.gross_revenue_inr.toLocaleString()}`}
                    sub={`@ ₹${proj.market_price_per_kg}/kg`}/>
                  <StatBox label="Net Profit" color="purple"
                    value={`₹${proj.net_profit_inr.toLocaleString()}`}
                    sub={`ROI: ${proj.roi_percent}%`}/>
                  <StatBox label="Break-even" color="amber"
                    value={`${proj.break_even_yield_kg.toLocaleString()} kg`}
                    sub="Minimum to cover costs"/>
                </div>
              )}

              {/* Tabs */}
              <div className="sfa-tabs">
                {[
                  { id: "sowing",    label: "Sowing",      icon: <Calendar size={15}/> },
                  { id: "irrigation",label: "Irrigation",  icon: <Droplets size={15}/> },
                  { id: "agrochem",  label: "Fertilizers & Pesticides", icon: <FlaskConical size={15}/> },
                  { id: "risks",     label: "Risks & Tips",icon: <AlertTriangle size={15}/> },
                ].map(t => (
                  <button key={t.id}
                    className={`sfa-tab ${activeTab === t.id ? "active" : ""}`}
                    onClick={() => setActiveTab(t.id)}>
                    {t.icon} <span>{t.label}</span>
                  </button>
                ))}
              </div>

              {/* Tab: Sowing */}
              {activeTab === "sowing" && (
                <div className="sfa-tab-content">
                  {plan.sowing_schedule.map((s, i) => (
                    <div key={i} className="sfa-sowing-card">
                      <div className="sfa-sowing-timeline">
                        <div className="sfa-timeline-item">
                          <div className="sfa-tl-dot green"/>
                          <div>
                            <div className="sfa-tl-label">Sowing Window</div>
                            <div className="sfa-tl-value">{s.sowing_start} → {s.sowing_end}</div>
                          </div>
                        </div>
                        {s.transplant_date && (
                          <div className="sfa-timeline-item">
                            <div className="sfa-tl-dot blue"/>
                            <div>
                              <div className="sfa-tl-label">Transplanting</div>
                              <div className="sfa-tl-value">{s.transplant_date}</div>
                            </div>
                          </div>
                        )}
                        <div className="sfa-timeline-item">
                          <div className="sfa-tl-dot amber"/>
                          <div>
                            <div className="sfa-tl-label">Expected Harvest</div>
                            <div className="sfa-tl-value">{s.harvest_date}</div>
                          </div>
                        </div>
                      </div>
                      <div className="sfa-sowing-meta">
                        <span><strong>Germination:</strong> {s.germination_days} days</span>
                        <span><strong>Total Duration:</strong> {s.total_days} days</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* Tab: Irrigation */}
              {activeTab === "irrigation" && (
                <div className="sfa-tab-content">
                  <div className="sfa-table-wrap">
                    <table className="sfa-table">
                      <thead>
                        <tr>
                          <th>Week</th><th>Crop Stage</th><th>Method</th>
                          <th>Freq/Week</th><th>Water (mm)</th><th>Notes</th>
                        </tr>
                      </thead>
                      <tbody>
                        {plan.irrigation_plan.map((row, i) => (
                          <tr key={i}>
                            <td><Badge color="blue">Wk {row.week}</Badge></td>
                            <td>{row.stage}</td>
                            <td>{row.method}</td>
                            <td>{row.frequency_per_week}×</td>
                            <td>{row.water_mm} mm</td>
                            <td className="sfa-notes">{row.notes}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* Tab: Agrochemicals */}
              {activeTab === "agrochem" && (
                <div className="sfa-tab-content">
                  <div className="sfa-table-wrap">
                    <table className="sfa-table">
                      <thead>
                        <tr>
                          <th>Week</th><th>Type</th><th>Product</th>
                          <th>Dose/Acre</th><th>Method</th><th>Notes</th>
                        </tr>
                      </thead>
                      <tbody>
                        {plan.agrochemical_timeline.map((row, i) => (
                          <tr key={i}>
                            <td><Badge color="blue">Wk {row.week}</Badge></td>
                            <td>
                              <Badge color={
                                row.type === "fertilizer" ? "green"
                                : row.type === "pesticide" ? "red" : "amber"
                              }>
                                {row.type}
                              </Badge>
                            </td>
                            <td>{row.product}</td>
                            <td>{row.dose_per_acre}</td>
                            <td>{row.application_method}</td>
                            <td className="sfa-notes">{row.notes}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* Tab: Risks & Tips */}
              {activeTab === "risks" && (
                <div className="sfa-tab-content">
                  {plan.risk_factors.length > 0 && (
                    <div className="sfa-risk-section">
                      <h4><AlertTriangle size={16}/> Risk Factors</h4>
                      <ul className="sfa-risk-list">
                        {plan.risk_factors.map((r, i) => (
                          <li key={i}><AlertTriangle size={13} className="sfa-risk-icon"/> {r}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                  <div className="sfa-tips-section">
                    <h4><CheckCircle2 size={16}/> Advisory Notes</h4>
                    <ul className="sfa-tips-list">
                      {plan.advisory_notes.map((n, i) => (
                        <li key={i}><CheckCircle2 size={13} className="sfa-tip-icon"/> {n}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}

            </div>
          )}
        </main>
      </div>
    </div>
  );
}
