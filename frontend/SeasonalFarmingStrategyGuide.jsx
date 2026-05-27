import React, { useState } from "react";
import "./SeasonalFarmingStrategyGuide.css";

const SEASONS = [
  {
    id: 'kharif',
    name: 'Kharif',
    summary: 'Monsoon-season crops (June–Oct): rice, maize, millets, cotton.',
    strategy: [
      'Sow with onset of reliable monsoon; select short-duration varieties if rainfall is uncertain.',
      'Prioritize water-conserving practices: raised beds, SRI for paddy where suitable.',
      'Apply basal fertilizers based on soil test; split N applications to match crop uptake.',
      'Use mulches and cover crops to retain moisture and suppress weeds.',
      'Monitor pests (stem borer, leaf folders) frequently after heavy rains.',
    ],
  },
  {
    id: 'rabi',
    name: 'Rabi',
    summary: 'Cool-season crops (Oct–Mar): wheat, mustard, chickpea, barley.',
    strategy: [
      'Prepare seedbed in early autumn — aim for good tilth and seed-soil contact.',
      'Use residual soil moisture and timely irrigation (lighter but frequent for some crops).',
      'Apply phosphorus and potassium as recommended; apply nitrogen in splits tied to tillering/early growth.',
      'Consider relay cropping or cover crops to protect soil after harvest.',
      'Watch for fungal diseases (rusts, blights) and manage with resistant varieties and timely sprays.',
    ],
  },
  {
    id: 'zaid',
    name: 'Zaid',
    summary: 'Short summer season (Mar–Jun): vegetables, fodder, some pulses.',
    strategy: [
      'Target fast-maturing, heat-tolerant crops and vegetables for high-value returns.',
      'Irrigate carefully — focus on deficit scheduling and water-efficient methods like drip.',
      'Use light nutrient dressings; avoid heavy nitrogen that promotes excessive vegetative growth under heat.',
      'Implement pest monitoring for sucking pests and thrips—use biocontrol where possible.',
      'Use shading nets or mulches for temperature-sensitive seedlings to reduce heat stress.',
    ],
  },
];

export default function SeasonalFarmingStrategyGuide({ onClose }) {
  const [active, setActive] = useState(SEASONS[0]);

  return (
    <div className="sfs-overlay" onClick={onClose} role="dialog" aria-modal="true">
      <div className="sfs-modal" onClick={(e) => e.stopPropagation()}>
        <header className="sfs-header">
          <h2>Seasonal Farming Strategy Guide</h2>
          <button className="sfs-close" aria-label="Close" onClick={onClose}>✕</button>
        </header>

        <div className="sfs-body">
          <aside className="sfs-nav">
            {SEASONS.map((s) => (
              <button key={s.id} className={`sfs-nav-btn ${active.id === s.id ? 'active' : ''}`} onClick={() => setActive(s)}>
                {s.name}
              </button>
            ))}
          </aside>

          <main className="sfs-main">
            <h3>{active.name}</h3>
            <p className="sfs-summary">{active.summary}</p>

            <h4>Key Strategy Points</h4>
            <ul>
              {active.strategy.map((pt, idx) => <li key={idx}>{pt}</li>)}
            </ul>

            <div className="sfs-actions">
              <button className="sfs-primary" onClick={onClose}>Close</button>
            </div>
          </main>
        </div>
      </div>
    </div>
  );
}
