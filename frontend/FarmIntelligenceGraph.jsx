import React, { useEffect, useMemo, useState } from "react";
import { GitBranch, Loader, RefreshCw, Sparkles, X, ChevronRight, History, MapPin, BrainCircuit, Bug, Droplets, IndianRupee } from "lucide-react";
import apiClient from "./services/api";
import "./FarmIntelligenceGraph.css";

const buildInitialForm = (userData, weatherData) => {
  const currentWeather = weatherData?.current || {};
  const dailyWeather = weatherData?.daily || {};
  return {
    crop_type: userData?.cropType || userData?.crop_type || "rice",
    location: weatherData?.location?.name || weatherData?.location?.city || userData?.location || "",
    weather: {
      temperature: currentWeather.temperature_2m ?? 30,
      humidity: currentWeather.relative_humidity_2m ?? 70,
      rainfall_next_24h: dailyWeather.precipitation_sum?.[0] ?? 4,
    },
    soil: {
      ph: 6.5,
      moisture: 35,
      nitrogen: "medium",
      phosphorus: "medium",
      potassium: "medium",
    },
    pest: {
      pressure: "medium",
      observed: "",
    },
    market: {
      commodity: userData?.cropType || "Rice",
      price: 3200,
      trend: "stable",
    },
  };
};

const scoreClass = (score) => {
  if (score >= 75) return "score-high";
  if (score >= 45) return "score-medium";
  return "score-low";
};

function NodeCard({ node }) {
  return (
    <div className={`graph-node ${node.type}`}>
      <div className="node-top">
        <div>
          <p className="node-kicker">{node.type}</p>
          <h4>{node.label}</h4>
        </div>
        {typeof node.score === "number" && (
          <div className={`node-score ${scoreClass(node.score)}`}>{node.score}%</div>
        )}
      </div>
      <div className="node-summary">
        {Object.entries(node.summary || {}).map(([key, value]) => (
          <div key={key} className="summary-row">
            <span>{key}</span>
            <strong>{String(value ?? "-")}</strong>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function FarmIntelligenceGraph({ userData, weatherData, onClose }) {
  const [formData, setFormData] = useState(() => buildInitialForm(userData, weatherData));
  const [loading, setLoading] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);

  useEffect(() => {
    setFormData(buildInitialForm(userData, weatherData));
  }, [userData, weatherData]);

  const payload = useMemo(() => ({
    crop_type: formData.crop_type,
    location: formData.location,
    weather: formData.weather,
    soil: formData.soil,
    pest: formData.pest,
    market: formData.market,
    store_history: true,
  }), [formData]);

  const loadHistory = async () => {
    setHistoryLoading(true);
    try {
      const response = await apiClient.get("/api/farm-intelligence/me", { skipGlobalLoader: true });
      if (response.data?.success) {
        setHistory(response.data.data || []);
      }
    } catch {
      setHistory([]);
    } finally {
      setHistoryLoading(false);
    }
  };

  useEffect(() => {
    loadHistory();
  }, []);

  const handleChange = (section, key, value) => {
    setFormData((previous) => ({
      ...previous,
      [section]: {
        ...previous[section],
        [key]: value,
      },
    }));
  };

  const handleGenerate = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError("");

    try {
      const response = await apiClient.post("/api/farm-intelligence/recommend", payload, {
        retries: 0,
        errorContext: "farm-intelligence-graph",
      });

      if (response.data?.success) {
        setResult(response.data);
        await loadHistory();
      } else {
        setError("Unable to generate a farm intelligence graph right now.");
      }
    } catch (err) {
      setError(err.response?.data?.detail || err.message || "Unable to generate a farm intelligence graph right now.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="farm-intelligence-modal">
      <div className="farm-intelligence-shell">
        <header className="farm-intelligence-header">
          <div>
            <div className="header-chip"><GitBranch size={14} /> Graph intelligence</div>
            <h2>Farm Intelligence Graph System</h2>
            <p>Connect soil, weather, crop, pest, and market data into one reasoning graph.</p>
          </div>
          <button className="close-btn" onClick={onClose} aria-label="Close farm intelligence graph">
            <X size={20} />
          </button>
        </header>

        <form className="farm-intelligence-form" onSubmit={handleGenerate}>
          <div className="form-grid">
            <label>
              <span><Sparkles size={14} /> Crop</span>
              <input
                value={formData.crop_type}
                onChange={(event) => setFormData((previous) => ({ ...previous, crop_type: event.target.value }))}
                placeholder="Rice, wheat, cotton..."
              />
            </label>
            <label>
              <span><MapPin size={14} /> Location</span>
              <input
                value={formData.location}
                onChange={(event) => setFormData((previous) => ({ ...previous, location: event.target.value }))}
                placeholder="Farm location"
              />
            </label>
            <label>
              <span><Droplets size={14} /> Temperature</span>
              <input
                type="number"
                value={formData.weather.temperature}
                onChange={(event) => handleChange("weather", "temperature", Number(event.target.value))}
              />
            </label>
            <label>
              <span><Droplets size={14} /> Humidity</span>
              <input
                type="number"
                value={formData.weather.humidity}
                onChange={(event) => handleChange("weather", "humidity", Number(event.target.value))}
              />
            </label>
            <label>
              <span><Droplets size={14} /> Rainfall next 24h</span>
              <input
                type="number"
                value={formData.weather.rainfall_next_24h}
                onChange={(event) => handleChange("weather", "rainfall_next_24h", Number(event.target.value))}
              />
            </label>
            <label>
              <span><Droplets size={14} /> Soil moisture</span>
              <input
                type="number"
                value={formData.soil.moisture}
                onChange={(event) => handleChange("soil", "moisture", Number(event.target.value))}
              />
            </label>
            <label>
              <span><Bug size={14} /> Pest pressure</span>
              <select value={formData.pest.pressure} onChange={(event) => handleChange("pest", "pressure", event.target.value)}>
                <option value="low">Low</option>
                <option value="medium">Medium</option>
                <option value="high">High</option>
                <option value="critical">Critical</option>
              </select>
            </label>
            <label>
              <span><IndianRupee size={14} /> Market price</span>
              <input
                type="number"
                value={formData.market.price}
                onChange={(event) => handleChange("market", "price", Number(event.target.value))}
              />
            </label>
          </div>

          <div className="market-row">
            <label>
              <span>Observed pest</span>
              <input
                value={formData.pest.observed}
                onChange={(event) => handleChange("pest", "observed", event.target.value)}
                placeholder="Whitefly, hopper, blight..."
              />
            </label>
            <label>
              <span>Market trend</span>
              <select value={formData.market.trend} onChange={(event) => handleChange("market", "trend", event.target.value)}>
                <option value="stable">Stable</option>
                <option value="up">Rising</option>
                <option value="down">Falling</option>
              </select>
            </label>
            <label>
              <span>Commodity</span>
              <input
                value={formData.market.commodity}
                onChange={(event) => handleChange("market", "commodity", event.target.value)}
                placeholder="Wheat, paddy, cotton..."
              />
            </label>
          </div>

          <div className="action-row">
            <button type="submit" className="generate-btn" disabled={loading}>
              {loading ? <Loader size={16} className="spin" /> : <BrainCircuit size={16} />}
              Generate graph insight
            </button>
            <button type="button" className="history-btn" onClick={loadHistory} disabled={historyLoading}>
              <RefreshCw size={16} className={historyLoading ? "spin" : ""} /> Refresh history
            </button>
          </div>
        </form>

        {error && <div className="error-banner">{error}</div>}

        {result && (
          <section className="result-panel">
            <div className="result-header">
              <div>
                <p className="section-tag">AI recommendations</p>
                <h3>Cross-factor reasoning</h3>
              </div>
              <div className="score-pill">Pest {result.scores.pest_risk}% · Irrigation {result.scores.irrigation}% · Market {result.scores.market}%</div>
            </div>

            <div className="reasoning-list">
              {result.reasoning.map((line) => (
                <div key={line} className="reasoning-item">
                  <ChevronRight size={16} />
                  <span>{line}</span>
                </div>
              ))}
            </div>

            <div className="graph-grid">
              {result.graph.nodes.map((node) => <NodeCard key={node.id} node={node} />)}
            </div>

            <div className="edge-list">
              {result.graph.edges.map((edge) => (
                <div key={`${edge.from}-${edge.to}-${edge.reason}`} className="edge-item">
                  <strong>{edge.from}</strong>
                  <ChevronRight size={14} />
                  <strong>{edge.to}</strong>
                  <span>{edge.reason}</span>
                </div>
              ))}
            </div>

            <div className="recommendation-list">
              {result.recommendations.map((item) => (
                <article key={item.title} className={`recommendation-card ${item.priority}`}>
                  <div className="recommendation-title-row">
                    <h4>{item.title}</h4>
                    <span>{item.priority}</span>
                  </div>
                  <p>{item.action}</p>
                  <small>{item.why}</small>
                </article>
              ))}
            </div>
          </section>
        )}

        <section className="history-panel">
          <div className="result-header">
            <div>
              <p className="section-tag">Farm history</p>
              <h3>Saved relational graph history</h3>
            </div>
            <div className="history-meta"><History size={14} /> {history.length} entries</div>
          </div>

          {historyLoading ? (
            <div className="history-empty">Loading saved graph history...</div>
          ) : history.length === 0 ? (
            <div className="history-empty">No farm intelligence graph has been saved yet.</div>
          ) : (
            <div className="history-list">
              {history.slice(0, 5).map((entry) => (
                <article key={entry.history_id} className="history-item">
                  <div className="history-top">
                    <strong>{entry.crop_type}</strong>
                    <span>{entry.location || "Unknown location"}</span>
                  </div>
                  <p>{entry.summary}</p>
                </article>
              ))}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
