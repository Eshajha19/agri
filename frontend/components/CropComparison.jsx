import React, { useState } from "react";
import "./CropComparison.css";

const cropsData = {
  wheat: { profitability: "High", water: "Medium", risk: "Low", duration: "120 days" },
  rice: { profitability: "Medium", water: "High", risk: "Medium", duration: "150 days" },
  maize: { profitability: "Medium", water: "Low", risk: "Low", duration: "90 days" },
  cotton: { profitability: "High", water: "High", risk: "High", duration: "180 days" },
};

export default function CropComparison() {
  const [crop1, setCrop1] = useState("wheat");
  const [crop2, setCrop2] = useState("rice");

  const c1 = cropsData[crop1];
  const c2 = cropsData[crop2];

  const metricRows = [
    { label: "Profitability", k: "profitability" },
    { label: "Water Usage", k: "water" },
    { label: "Risk Level", k: "risk" },
    { label: "Growth Duration", k: "duration" },
  ];

  const badgeClassFor = (value) => {
    const v = String(value).toLowerCase();
    if (v === "high") return "badge-high";
    if (v === "medium") return "badge-medium";
    if (v === "low") return "badge-low";
    return "";
  };

  const normalizeProgressValue = (value) => {
    const v = String(value).toLowerCase();
    if (v === "high") return 92;
    if (v === "medium") return 60;
    if (v === "low") return 28;
    return 50;
  };

  const renderCropCard = (cropKey, cropData) => {
    const profitability = cropData.profitability;
    const risk = cropData.risk;

    return (
      <section className="crop-card" aria-label={`${cropKey} crop card`}>
        <header className="crop-card-header">
          <div className="crop-icon" aria-hidden="true">
            {cropKey === "wheat" && "🌾"}
            {cropKey === "rice" && "🌾"}
            {cropKey === "maize" && "🌽"}
            {cropKey === "cotton" && "🧵"}
          </div>
          <h2>{cropKey.charAt(0).toUpperCase() + cropKey.slice(1)}</h2>
        </header>

        <div className="crop-card-body">
          <div className="metric-group">
            <h3>
              <span>Performance snapshot</span>
            </h3>

            <div className="metric-row">
              <div className="metric-top">
                <span className="metric-label">Profitability</span>
                <span className={`badge ${badgeClassFor(profitability)}`}>{profitability}</span>
              </div>
              <div className="progress-bar" aria-hidden="true">
                <div className="progress-fill" style={{ width: `${normalizeProgressValue(profitability)}%` }} />
              </div>
            </div>

            <div className="metric-row">
              <div className="metric-top">
                <span className="metric-label">Risk Level</span>
                <span className={`badge ${badgeClassFor(risk)}`}>{risk}</span>
              </div>
              <div className="progress-bar" aria-hidden="true">
                <div className="progress-fill" style={{ width: `${normalizeProgressValue(risk)}%` }} />
              </div>
            </div>
          </div>

          <div className="metric-group" aria-label="Other metrics">
            {metricRows.map((row) => (
              <div key={row.k} className="metric-row">
                <div className="metric-top">
                  <span className="metric-label">{row.label}</span>
                  <span className="metric-value">{cropData[row.k]}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
    );
  };

  return (
    <div className="comparison-container">
      <header className="comparison-header">
        <h1>🌾 Crop Comparison</h1>
        <p className="header-subtitle">
          Compare key agronomic factors side-by-side to choose the crop that fits your farm conditions.
        </p>
      </header>

      <div className="selection-panel" aria-label="Crop selectors">
        <div className="selection-group">
          <label htmlFor="crop1">Crop A</label>
          <select id="crop1" value={crop1} onChange={(e) => setCrop1(e.target.value)}>
            {Object.keys(cropsData).map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </div>

        <div className="selection-group">
          <label htmlFor="crop2">Crop B</label>
          <select id="crop2" value={crop2} onChange={(e) => setCrop2(e.target.value)}>
            {Object.keys(cropsData).map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="comparison-cards">
        {renderCropCard(crop1, c1)}
        {renderCropCard(crop2, c2)}
      </div>

      <section className="comparison-summary" aria-label="Side-by-side table">
        <h2>Side-by-side metrics</h2>
        <div className="table-wrapper">
          <table className="comparison-table">
            <thead>
              <tr>
                <th>Metric</th>
                <th>{crop1}</th>
                <th>{crop2}</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Profitability</td>
                <td>
                  <span className={`badge ${badgeClassFor(c1.profitability)}`}>{c1.profitability}</span>
                </td>
                <td>
                  <span className={`badge ${badgeClassFor(c2.profitability)}`}>{c2.profitability}</span>
                </td>
              </tr>
              <tr>
                <td>Water Usage</td>
                <td>
                  <span className={`badge ${badgeClassFor(c1.water)}`}>{c1.water}</span>
                </td>
                <td>
                  <span className={`badge ${badgeClassFor(c2.water)}`}>{c2.water}</span>
                </td>
              </tr>
              <tr>
                <td>Risk Level</td>
                <td>
                  <span className={`badge ${badgeClassFor(c1.risk)}`}>{c1.risk}</span>
                </td>
                <td>
                  <span className={`badge ${badgeClassFor(c2.risk)}`}>{c2.risk}</span>
                </td>
              </tr>
              <tr>
                <td>Growth Duration</td>
                <td>{c1.duration}</td>
                <td>{c2.duration}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
