import React, { useState, useEffect, useRef } from "react";
import { ClipLoader } from "react-spinners";
import "./DamageHeatmap.css";

const API = process.env.REACT_APP_API_URL || "";

const SEV_COLORS = {
  0: "#e8f5e9",
  1: "#fff9c4",
  2: "#ffe082",
  3: "#ff8a65",
  4: "#e53935",
};
const SEV_LABELS = { 0: "None", 1: "Minor", 2: "Moderate", 3: "Severe", 4: "Critical" };

export default function DamageHeatmap() {
  const [fieldId, setFieldId] = useState("");
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [selectedCell, setSelectedCell] = useState(null);
  const inputRef = useRef(null);

  useEffect(() => {
    if (inputRef.current) inputRef.current.focus();
  }, []);

  const handleLoad = async () => {
    const id = fieldId.trim() || "default-field";
    setLoading(true);
    setError("");
    setSelectedCell(null);
    try {
      const res = await fetch(`${API}/api/damage-heatmap/${encodeURIComponent(id)}`);
      if (!res.ok) { setError("Failed to load heatmap"); setLoading(false); return; }
      setData(await res.json());
    } catch {
      setError("Network error");
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => { if (e.key === "Enter") handleLoad(); };

  return (
    <div className="dh-container">
      <div className="dh-card">
        <h2>Crop Damage Heatmap</h2>
        <p className="dh-subtitle">Visualize damage severity across farm zones.</p>

        <div className="dh-input-row">
          <input ref={inputRef} type="text" placeholder="Field ID (e.g. field-42)" value={fieldId} onChange={(e) => setFieldId(e.target.value)} onKeyDown={handleKeyDown} />
          <button onClick={handleLoad} disabled={loading}>
            {loading ? <ClipLoader size={16} color="#fff" /> : "Load Heatmap"}
          </button>
        </div>

        {error && <div className="dh-error">{error}</div>}

        {data && (
          <>
            <div className="dh-summary">
              <div className="dh-summary-row">
                <span>Crop: <strong>{data.crop_type}</strong></span>
                <span>Field: <strong>{data.field_size_acres} acres</strong></span>
                <span>Affected: <strong>{data.summary.estimated_area_affected_acres} acres</strong></span>
              </div>
              <div className="dh-summary-stats">
                <div className="dh-stat">
                  <span className="dh-stat-value">{data.summary.avg_damage_pct}%</span>
                  <span className="dh-stat-label">Avg Damage</span>
                </div>
                <div className="dh-stat">
                  <span className="dh-stat-value">{data.summary.damaged_cells}/{data.summary.total_cells}</span>
                  <span className="dh-stat-label">Damaged Zones</span>
                </div>
                <div className="dh-stat">
                  <span className="dh-stat-value">{data.summary.yield_loss_projection_pct}%</span>
                  <span className="dh-stat-label">Yield Loss Projection</span>
                </div>
                <div className="dh-stat">
                  <span className="dh-stat-value">{data.summary.estimated_yield_loss_qtl} qtl</span>
                  <span className="dh-stat-label">Est. Yield Loss</span>
                </div>
              </div>
            </div>

            <div className="dh-grid-wrapper">
              <div className="dh-grid" style={{ gridTemplateColumns: `repeat(${data.grid[0].length}, 1fr)` }}>
                {data.grid.flat().map((cell, i) => (
                  <div
                    key={i}
                    className="dh-cell"
                    style={{ background: SEV_COLORS[cell.severity] }}
                    onClick={() => setSelectedCell(cell)}
                    title={`Zone (${cell.row},${cell.col}) - ${cell.label}: ${cell.damage_pct}%`}
                  >
                    <span className="dh-cell-pct">{cell.damage_pct > 0 ? `${cell.damage_pct}%` : ""}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="dh-legend">
              {Object.entries(SEV_LABELS).map(([k, v]) => (
                <div key={k} className="dh-legend-item">
                  <span className="dh-legend-swatch" style={{ background: SEV_COLORS[k] }} />
                  <span>{v}</span>
                </div>
              ))}
            </div>

            {selectedCell && (
              <div className="dh-cell-detail">
                <strong>Zone ({selectedCell.row}, {selectedCell.col})</strong>
                <span>Severity: {selectedCell.label}</span>
                <span>Damage: {selectedCell.damage_pct}%</span>
              </div>
            )}

            <div className="dh-filter-hint">
              <span>Enter a different Field ID above to load another field.</span>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
