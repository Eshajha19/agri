import React, { useState } from "react";
import "./FertilizerOveruseGuide.css";

const TOPICS = [
  {
    id: "soil-degradation",
    title: "Soil Degradation",
    image: "https://images.unsplash.com/photo-1501004318641-b39e6451bec6?auto=format&fit=crop&w=1000&q=60",
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
    image: "https://images.unsplash.com/photo-1498887964941-9c5b0a2d5f79?auto=format&fit=crop&w=1000&q=60",
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
    image: "https://images.unsplash.com/photo-1501004318641-4d1b8f9f2d12?auto=format&fit=crop&w=1000&q=60",
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
  const [showLightbox, setShowLightbox] = useState(false);

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
                <img src={`${t.image}&w=160&q=40`} alt={t.title} />
                <span>{t.title}</span>
              </button>
            ))}
          </nav>

          <main className="fo-main">
            <div className="fo-visual">
              <img src={`${active.image}&w=900&q=70`} alt={active.title} onClick={() => setShowLightbox(true)} />
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

        {showLightbox && (
          <div className="fo-lightbox" onClick={() => setShowLightbox(false)}>
            <img src={`${active.image}&w=1400&q=80`} alt="Large" />
          </div>
        )}
      </div>
    </div>
  );
}
