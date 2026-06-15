import React, { useState } from "react";
import "./FarmingMistakesGuide.css";

const MISTAKES = [
  {
    id: "over-fertilization",
    title: "Over-fertilization",
    icon: "🌱",
    problem: "Applying too much fertilizer leading to nutrient burn and runoff.",
    avoid: [
      "Perform a soil test before applying fertilizers.",
      "Follow recommended doses for your crop and soil type.",
      "Use split applications and consider slow-release formulations.",
    ],
  },
  {
    id: "wrong-irrigation",
    title: "Wrong irrigation timing",
    icon: "💧",
    problem: "Irrigating at suboptimal times causing stress or disease.",
    avoid: [
      "Irrigate during early morning or late evening to reduce evaporation.",
      "Match irrigation to crop stage and soil moisture (avoid waterlogging).",
      "Use soil moisture sensors or simple manual checks to guide decisions.",
    ],
  },
  {
    id: "poor-seed-selection",
    title: "Poor seed selection",
    icon: "🌾",
    problem: "Using low-quality or inappropriate varieties for the region.",
    avoid: [
      "Choose certified seed suited to your agro-climatic zone.",
      "Consider disease-resistant or drought-tolerant varieties if needed.",
      "Check seed viability and expiry; store seeds properly.",
    ],
  },
  {
    id: "late-harvest",
    title: "Late harvest & poor post-harvest handling",
    icon: "⏰",
    problem: "Delaying harvest reduces quality and increases losses.",
    avoid: [
      "Monitor maturity indices and harvest at recommended windows.",
      "Plan logistics and drying/storage ahead of harvest.",
      "Use recommended drying and storage practices to reduce spoilage.",
    ],
  },
];

export default function FarmingMistakesGuide({ onClose }) {
  const [active, setActive] = useState(MISTAKES[0]);

  return (
    <div className="fm-overlay" onClick={onClose} role="dialog" aria-modal="true">
      <div className="fm-modal" onClick={(e) => e.stopPropagation()}>
        <header className="fm-header">
          <h2>Farming Mistakes Awareness</h2>
          <button className="fm-close" aria-label="Close" onClick={onClose}>✕</button>
        </header>

        <div className="fm-body">
          <aside className="fm-list">
            {MISTAKES.map((m) => (
              <button
                key={m.id}
                className={`fm-list-item ${active.id === m.id ? "active" : ""}`}
                onClick={() => setActive(m)}
              >
                <span className="fm-icon">{m.icon}</span>
                <span>{m.title}</span>
              </button>
            ))}
          </aside>

          <main className="fm-detail">
            <div className="fm-visual">
              <span className="fm-detail-icon">{active.icon}</span>
            </div>

            <div className="fm-info">
              <h3>{active.title}</h3>
              <p className="fm-problem"><strong>Problem:</strong> {active.problem}</p>

              <h4>How to avoid</h4>
              <ul>
                {active.avoid.map((a, i) => (
                  <li key={i}>{a}</li>
                ))}
              </ul>

              <div className="fm-footer-actions">
                <button className="fm-primary" onClick={onClose}>Got it</button>
              </div>
            </div>
          </main>
        </div>
      </div>
    </div>
  );
}
