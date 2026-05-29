import React from "react";
import "./SeasonalFarmingStrategyGuide.css";

export default function SeasonalFarmingStrategyGuide({ onClose }) {
  return (
    <div className="seasonal-guide-shell">
      <header className="seasonal-guide-header">
        <h2>Seasonal Farming Strategy Guide</h2>
        <p className="lead">Practical, crop-agnostic strategies for Kharif, Rabi and Zaid seasons.</p>
      </header>

      <div className="seasonal-grid">
        <section className="season-card kharif">
          <h3>Kharif (Monsoon)</h3>
          <ul>
            <li><strong>Sowing:</strong> Start with timely sowing aligned to monsoon onset; use short-duration varieties if rains are uncertain.</li>
            <li><strong>Irrigation:</strong> Prefer rain-fed sowing; establish drainage and avoid waterlogging during heavy rains.</li>
            <li><strong>Soil & Nutrition:</strong> Apply basal NPK and consider green manures; monitor for leaching after heavy showers.</li>
            <li><strong>Pests & Diseases:</strong> Watch for fungal diseases and stem borers; schedule early scouting after rains.</li>
            <li><strong>Harvest:</strong> Plan harvest windows avoiding peak monsoon to reduce quality loss.</li>
          </ul>
        </section>

        <section className="season-card rabi">
          <h3>Rabi (Winter)</h3>
          <ul>
            <li><strong>Sowing:</strong> Sow after soil moisture is retained from monsoon or after irrigation; choose varieties for cool season growth.</li>
            <li><strong>Irrigation:</strong> Use supplemental irrigation at critical stages (tillering, flowering); conserve soil moisture with mulches.</li>
            <li><strong>Soil & Nutrition:</strong> Apply phosphorus and potassium at sowing; nitrogen top-ups timed to growth stages.</li>
            <li><strong>Pests & Diseases:</strong> Monitor aphids and rusts; dry, cool conditions favour some pests — scout regularly.</li>
            <li><strong>Harvest:</strong> Harvest in dry weather; manage drying and storage to avoid post-harvest losses.</li>
          </ul>
        </section>

        <section className="season-card zaid">
          <h3>Zaid (Summer interim)</h3>
          <ul>
            <li><strong>Sowing:</strong> Short-duration vegetables and fodder crops fit Zaid; select tolerant varieties for heat and water stress.</li>
            <li><strong>Irrigation:</strong> Plan light, frequent irrigations and adopt micro-irrigation where possible.</li>
            <li><strong>Soil & Nutrition:</strong> Ensure balanced N and timely fertiliser splits; consider organic mulches to reduce evaporation.</li>
            <li><strong>Pests & Diseases:</strong> High temperatures can increase pest pressure — apply IPM and biological controls early.</li>
            <li><strong>Harvest:</strong> Fast harvest cycles; use post-harvest cooling where feasible for perishables.</li>
          </ul>
        </section>
      </div>

      <div className="seasonal-actions">
        <button className="btn close" onClick={onClose}>Close</button>
        <a className="btn print" href="#" onClick={(e) => { e.preventDefault(); window.print(); }}>Print Guide</a>
      </div>
    </div>
  );
}
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
