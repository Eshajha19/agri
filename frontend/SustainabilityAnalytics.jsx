import React, { useCallback, useEffect, useMemo, useState, lazy, Suspense } from "react";
// import {
//   BarChart,
//   Bar,
//   XAxis,
//   YAxis,
//   CartesianGrid,
//   Tooltip,
//   Legend,
//   ResponsiveContainer,
//   LineChart,
//   Line,
// } from "recharts";
import {
  Droplets,
  Cloud,
  Leaf,
  FileDown,
  BarChart3,
  Loader2,
} from "lucide-react";
// import jsPDF from "jspdf";
// import autoTable from "jspdf-autotable";
import {
  analyzeSustainability,
  fetchSustainabilityHistory,
} from "./services/sustainabilityApi";
import "./SustainabilityAnalytics.css";

// Lazy load recharts components
const ResponsiveContainer = lazy(() => import("recharts").then(m => ({ default: m.ResponsiveContainer })));
const BarChart = lazy(() => import("recharts").then(m => ({ default: m.BarChart })));
const Bar = lazy(() => import("recharts").then(m => ({ default: m.Bar })));
const XAxis = lazy(() => import("recharts").then(m => ({ default: m.XAxis })));
const YAxis = lazy(() => import("recharts").then(m => ({ default: m.YAxis })));
const CartesianGrid = lazy(() => import("recharts").then(m => ({ default: m.CartesianGrid })));
const Tooltip = lazy(() => import("recharts").then(m => ({ default: m.Tooltip })));
const Legend = lazy(() => import("recharts").then(m => ({ default: m.Legend })));
const LineChart = lazy(() => import("recharts").then(m => ({ default: m.LineChart })));
const Line = lazy(() => import("recharts").then(m => ({ default: m.Line })));

const STORAGE_KEY = "agri:sustainabilityHistory";

const CROPS = ["Rice", "Wheat", "Maize", "Cotton", "Sugarcane", "Pulses", "Vegetables"];
const SEASONS = ["Kharif", "Rabi", "Zaid"];
const IRRIGATION_TYPES = [
  { value: "drip", label: "Drip" },
  { value: "sprinkler", label: "Sprinkler" },
  { value: "flood", label: "Flood" },
  { value: "rainfed", label: "Rainfed" },
];

function deriveSeason() {
  const m = new Date().getMonth() + 1;
  if (m >= 6 && m <= 10) return "Kharif";
  if (m >= 3 && m <= 5) return "Zaid";
  return "Rabi";
}

function loadLocalHistory(userId) {
  try {
    const raw = localStorage.getItem(`${STORAGE_KEY}:${userId || "anonymous"}`);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveLocalHistory(userId, entry) {
  const key = `${STORAGE_KEY}:${userId || "anonymous"}`;
  const prev = loadLocalHistory(userId);
  const next = [...prev, entry].slice(-24);
  localStorage.setItem(key, JSON.stringify(next));
}

export default function SustainabilityAnalytics({ userData, onClose }) {
  const userId = userData?.uid || userData?.id || "anonymous";

  const [form, setForm] = useState({
    crop_type: userData?.cropType || "Rice",
    season: userData?.season || deriveSeason(),
    acreage: userData?.landArea || "1",
    irrigation_type: "drip",
    irrigation_events: "12",
    fertilizer_n_kg: "",
    fertilizer_p_kg: "",
    fertilizer_k_kg: "",
    machinery_hours: "",
    diesel_liters: "",
    organic_practices: false,
  });

  const [analysis, setAnalysis] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [error, setError] = useState("");

  const loadHistory = useCallback(async () => {
    try {
      const remote = await fetchSustainabilityHistory(userId, 12);
      const local = loadLocalHistory(userId);
      const merged = [...remote, ...local].reduce((acc, row) => {
        if (!acc.find((r) => r.record_id === row.record_id)) acc.push(row);
        return acc;
      }, []);
      merged.sort((a, b) => (a.created_at < b.created_at ? -1 : 1));
      setHistory(merged.slice(-12));
    } catch {
      setHistory(loadLocalHistory(userId).slice(-12));
    }
  }, [userId]);

  useEffect(() => {
    loadHistory();
  }, [loadHistory]);

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setForm((prev) => ({
      ...prev,
      [name]: type === "checkbox" ? checked : value,
    }));
  };

  const buildPayload = () => {
    const num = (v) => (v === "" || v == null ? undefined : Number(v));
    return {
      crop_type: form.crop_type,
      season: form.season,
      acreage: Number(form.acreage) || 1,
      irrigation_type: form.irrigation_type,
      irrigation_events: Number(form.irrigation_events) || 0,
      fertilizer_n_kg: num(form.fertilizer_n_kg),
      fertilizer_p_kg: num(form.fertilizer_p_kg),
      fertilizer_k_kg: num(form.fertilizer_k_kg),
      machinery_hours: num(form.machinery_hours),
      diesel_liters: num(form.diesel_liters),
      organic_practices: form.organic_practices,
      user_id: userId,
    };
  };

  const handleAnalyze = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    try {
      const data = await analyzeSustainability(buildPayload());
      setAnalysis(data);
      saveLocalHistory(userId, {
        record_id: data.record_id,
        crop_type: data.crop_type,
        season: data.season,
        acreage: data.acreage,
        created_at: data.created_at,
        water_footprint_m3: data.water_footprint_m3,
        carbon_emissions_kg_co2e: data.carbon_emissions_kg_co2e,
        sustainability_score: data.sustainability_score,
      });
      await loadHistory();
    } catch (err) {
      setError(err.message || "Failed to run sustainability analysis.");
    } finally {
      setLoading(false);
    }
  };

  const comparisonChartData = useMemo(() => {
    if (!analysis?.comparison_chart) return [];
    return analysis.comparison_chart;
  }, [analysis]);

  const historyChartData = useMemo(() => {
    return history.map((h, i) => ({
      name: `${h.crop_type?.slice(0, 6) || "Crop"} ${i + 1}`,
      water: h.water_footprint_m3,
      carbon: h.carbon_emissions_kg_co2e,
      score: h.sustainability_score,
    }));
  }, [history]);

  const handleExportPdf = async () => {
    if (!analysis) return;
    setExporting(true);
    try {
      const { default: jsPDF } = await import("jspdf");
      const { default: autoTable } = await import("jspdf-autotable");
      const doc = new jsPDF();
      const farmer = userData?.displayName || "Farmer";
      doc.setFontSize(16);
      doc.text("Fasal Saathi — Sustainability Report", 14, 20);
      doc.setFontSize(10);
      doc.text(`Farmer: ${farmer}`, 14, 28);
      doc.text(`Crop: ${analysis.crop_type} | Season: ${analysis.season} | Area: ${analysis.acreage} acres`, 14, 34);
      doc.text(`Generated: ${new Date(analysis.created_at).toLocaleString()}`, 14, 40);

      autoTable(doc, {
        startY: 46,
        head: [["Metric", "Value"]],
        body: [
          ["Water footprint (m³)", String(analysis.water_footprint_m3)],
          ["Carbon emissions (kg CO₂e)", String(analysis.carbon_emissions_kg_co2e)],
          ["Sustainability score", `${analysis.sustainability_score}/100`],
          ["Blue water (m³)", String(analysis.breakdown?.water?.blue_water_m3 ?? "—")],
          ["Green water (m³)", String(analysis.breakdown?.water?.green_water_m3 ?? "—")],
        ],
      });

      const recs = analysis.recommendations || [];
      let y = doc.lastAutoTable.finalY + 12;
      doc.setFontSize(12);
      doc.text("Recommendations", 14, y);
      y += 8;
      doc.setFontSize(10);
      recs.forEach((r) => {
        doc.text(`• ${r}`, 14, y, { maxWidth: 180 });
        y += 8;
      });

      doc.save(`FasalSaathi_Sustainability_${analysis.crop_type}_${Date.now()}.pdf`);
    } finally {
      setExporting(false);
    }
  };

  return (
    <div className="sustainability-analytics">
      <button type="button" className="sus-close-top" onClick={onClose} aria-label="Close">
        ×
      </button>

      <header className="sus-header">
        <h2>
          <Leaf size={26} />
          Crop Sustainability Analytics
        </h2>
        <p>
          Per-season water footprint and carbon emission estimates using LCA-style formulas.
          Use results to compare crops and improve eco-friendly practices.
        </p>
      </header>

      <div className="sus-body">
        <form onSubmit={handleAnalyze}>
          <div className="sus-form-grid">
            <label>
              Crop
              <select name="crop_type" value={form.crop_type} onChange={handleChange}>
                {CROPS.map((c) => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </select>
            </label>
            <label>
              Season
              <select name="season" value={form.season} onChange={handleChange}>
                {SEASONS.map((s) => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </select>
            </label>
            <label>
              Area (acres)
              <input name="acreage" type="number" min="0.1" step="0.1" value={form.acreage} onChange={handleChange} required />
            </label>
            <label>
              Irrigation
              <select name="irrigation_type" value={form.irrigation_type} onChange={handleChange}>
                {IRRIGATION_TYPES.map((t) => (
                  <option key={t.value} value={t.value}>{t.label}</option>
                ))}
              </select>
            </label>
            <label>
              Irrigation events
              <input name="irrigation_events" type="number" min="0" value={form.irrigation_events} onChange={handleChange} />
            </label>
            <label>
              N fertilizer (kg)
              <input name="fertilizer_n_kg" type="number" min="0" placeholder="Auto" value={form.fertilizer_n_kg} onChange={handleChange} />
            </label>
            <label>
              P fertilizer (kg)
              <input name="fertilizer_p_kg" type="number" min="0" placeholder="Auto" value={form.fertilizer_p_kg} onChange={handleChange} />
            </label>
            <label>
              K fertilizer (kg)
              <input name="fertilizer_k_kg" type="number" min="0" placeholder="Auto" value={form.fertilizer_k_kg} onChange={handleChange} />
            </label>
            <label>
              Machinery (hours)
              <input name="machinery_hours" type="number" min="0" placeholder="Auto" value={form.machinery_hours} onChange={handleChange} />
            </label>
            <label>
              Diesel (liters)
              <input name="diesel_liters" type="number" min="0" placeholder="Auto" value={form.diesel_liters} onChange={handleChange} />
            </label>
            <label className="sus-checkbox">
              <input name="organic_practices" type="checkbox" checked={form.organic_practices} onChange={handleChange} />
              Organic / reduced-input practices
            </label>
          </div>

          <div className="sus-actions" style={{ marginTop: 16 }}>
            <button type="submit" className="sus-btn sus-btn-primary" disabled={loading}>
              {loading ? <Loader2 size={18} className="spin" /> : <BarChart3 size={18} />}
              {loading ? "Analyzing…" : "Analyze season"}
            </button>
            {analysis && (
              <button type="button" className="sus-btn sus-btn-secondary" onClick={handleExportPdf} disabled={exporting}>
                <FileDown size={18} />
                {exporting ? "Exporting…" : "Export PDF report"}
              </button>
            )}
          </div>
        </form>

        {error && <div className="sus-error">{error}</div>}

        {analysis && (
          <>
            <div className="sus-metrics">
              <div className="sus-metric-card water">
                <h4><Droplets size={14} style={{ verticalAlign: "middle" }} /> Water footprint</h4>
                <div className="value">{analysis.water_footprint_m3.toLocaleString()} m³</div>
                <small>{analysis.breakdown?.water?.per_acre_m3} m³ / acre</small>
              </div>
              <div className="sus-metric-card carbon">
                <h4><Cloud size={14} style={{ verticalAlign: "middle" }} /> Carbon emissions</h4>
                <div className="value">{analysis.carbon_emissions_kg_co2e.toLocaleString()} kg</div>
                <small>CO₂e equivalent (season total)</small>
              </div>
              <div className="sus-metric-card score">
                <h4>Sustainability score</h4>
                <div className="value">{analysis.sustainability_score}/100</div>
                <small>vs regional crop benchmark</small>
              </div>
            </div>

            <div className="sus-charts">
              <div className="sus-chart-box">
                <h3>Current vs benchmark</h3>
                <Suspense fallback={<div className="sus-chart-placeholder" />}>
                  <ResponsiveContainer width="100%" height={240}>
                    <BarChart data={comparisonChartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="metric" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="current" fill="#059669" name="Your farm" />
                      <Bar dataKey="benchmark" fill="#94a3b8" name="Benchmark" />
                    </BarChart>
                  </ResponsiveContainer>
                </Suspense>
              </div>

              {historyChartData.length > 1 && (
                <div className="sus-chart-box">
                  <h3>Historical trend</h3>
                  <Suspense fallback={<div className="sus-chart-placeholder" />}>
                    <ResponsiveContainer width="100%" height={240}>
                      <LineChart data={historyChartData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" />
                        <YAxis yAxisId="left" />
                        <YAxis yAxisId="right" orientation="right" />
                        <Tooltip />
                        <Legend />
                        <Line yAxisId="left" type="monotone" dataKey="water" stroke="#0284c7" name="Water (m³)" />
                        <Line yAxisId="right" type="monotone" dataKey="carbon" stroke="#b45309" name="Carbon (kg)" />
                      </LineChart>
                    </ResponsiveContainer>
                  </Suspense>
                </div>
              )}
            </div>

            <div className="sus-recommendations">
              <h3>Recommendations</h3>
              <ul>
                {(analysis.recommendations || []).map((tip) => (
                  <li key={tip}>{tip}</li>
                ))}
              </ul>
            </div>
          </>
        )}

        {history.length > 0 && (
          <div className="sus-chart-box">
            <h3>Seasonal history</h3>
            <table className="sus-history-table">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Crop</th>
                  <th>Season</th>
                  <th>Water (m³)</th>
                  <th>Carbon (kg)</th>
                  <th>Score</th>
                </tr>
              </thead>
              <tbody>
                {[...history].reverse().map((row) => (
                  <tr key={row.record_id || row.created_at}>
                    <td>{row.created_at ? new Date(row.created_at).toLocaleDateString() : "—"}</td>
                    <td>{row.crop_type}</td>
                    <td>{row.season}</td>
                    <td>{row.water_footprint_m3?.toLocaleString?.() ?? row.water_footprint_m3}</td>
                    <td>{row.carbon_emissions_kg_co2e?.toLocaleString?.() ?? row.carbon_emissions_kg_co2e}</td>
                    <td>{row.sustainability_score}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
