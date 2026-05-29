import React, { useState } from "react";
import "./FertilizerOveruseGuide.css";

const TOPICS = [
  {
    id: "soil-degradation",
    title: "Soil Degradation",
    icon: "🌍",
    summary: "Excess fertilizers alter soil chemistry, reduce microbial diversity, and increase salinity or acidity over time.",
    details: [
      "Nutrient imbalances (excess N or P) can lock out micronutrients.",
      "Soil structure degradation from salt buildup reduces infiltration.",
      "Beneficial microbes decline leading to poorer nutrient cycling.",
    ],
  },
  {
    id: "crop-damage",
    title: "Crop Damage Symptoms",
    icon: "🍂",
    summary: "Symptoms include leaf burn, stunted roots, wilting despite moist soil, and increased pest/disease susceptibility.",
    details: [
      "Leaf tip burn or chlorosis from excess soluble salts.",
      "Poor root development and reduced uptake leading to stunting.",
      "Greater disease incidence due to imbalanced nutrition.",
    ],
  },
  {
    id: "recovery-methods",
    title: "Recovery Methods",
    icon: "♻️",
    summary: "Steps to recover degraded soil and mitigate plant damage.",
    details: [
      "Leaching salts with clean irrigation where drainage permits.",
      "Organic matter additions (compost, green manures) to rebuild biology.",
      "Use gypsum where sodicity is diagnosed; correct pH with lime or sulfur as needed.",
      "Adopt crop rotations and cover crops to restore soil structure.",
    ],
  },
];

export default function FertilizerOveruseGuide({ onClose }) {
  const [active, setActive] = useState(TOPICS[0]);

  return (
    <div className="fo-overlay" onClick={onClose} role="dialog" aria-modal="true">
      <div className="fo-modal" onClick={(e) => e.stopPropagation()}>
        <header className="fo-header">
          <h2>Fertilizer Overuse Damage Awareness</h2>
          <button className="fo-close" aria-label="Close" onClick={onClose}>✕</button>
        </header>

        <div className="fo-body">
          <nav className="fo-nav">
            {TOPICS.map((t) => (
              <button key={t.id} className={`fo-nav-btn ${active.id === t.id ? 'active' : ''}`} onClick={() => setActive(t)}>
                <span className="fo-icon">{t.icon}</span>
                <span>{t.title}</span>
              </button>
            ))}
          </nav>

          <main className="fo-main">
            <div className="fo-visual">
              <span className="fo-detail-icon">{active.icon}</span>
              <p className="fo-summary">{active.summary}</p>
            </div>

            <div className="fo-info">
              <h3>{active.title}</h3>
              <ul>
                {active.details.map((d, i) => (<li key={i}>{d}</li>))}
              </ul>

              <div className="fo-actions">
                <button className="fo-primary" onClick={onClose}>Close</button>
              </div>
            </div>
          </main>
        </div>
      </div>
    </div>
  );
}
