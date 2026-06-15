import React, { useState } from "react";
import "./SoilImprovementPath.css";

const STEPS = [
  {
    id: "season-1",
    title: "Season 1 — Baseline & Organic Matter",
    summary: "Soil testing and building organic matter.",
    actions: [
      "Conduct a full soil test (pH, EC, NPK, organic carbon).",
      "Add compost at 5–10 t/ha or use farmyard manure during land prep.",
      "Establish cover crops/green manure after harvest to add biomass.",
    ],
  },
  {
    id: "season-2",
    title: "Season 2 — Structure & Rotation",
    summary: "Improve structure and break pest/disease cycles.",
    actions: [
      "Introduce a legume into rotation (e.g., chickpea, mung) to fix nitrogen.",
      "Use reduced tillage and mulching to protect soil structure.",
      "Apply gypsum or lime only if soil test indicates need.",
    ],
  },
  {
    id: "season-3",
    title: "Season 3 — Fertilizer Correction Cycles",
    summary: "Move from blanket NPK to corrective, split applications.",
    actions: [
      "Apply fertilizers based on soil test recommendations and crop uptake.",
      "Use split dosing and slow-release formulations where possible.",
      "Monitor micronutrient needs and correct deficiencies early.",
    ],
  },
  {
    id: "season-4",
    title: "Season 4 — Monitoring & Continuous Improvement",
    summary: "Measure impact and iterate the plan.",
    actions: [
      "Repeat soil testing post-season to track changes in organic carbon and pH.",
      "Record yields and correlate with soil indicators; adjust plan.",
      "Adopt best practices like contouring, drainage fixes, and integrated pest management.",
    ],
  },
];

export default function SoilImprovementPath({ onClose }) {
  const [active, setActive] = useState(STEPS[0]);

  return (
    <div className="sip-overlay" onClick={onClose} role="dialog" aria-modal="true">
      <div className="sip-modal" onClick={(e) => e.stopPropagation()}>
        <header className="sip-header">
          <h2>Soil Improvement Learning Path</h2>
          <button className="sip-close" aria-label="Close" onClick={onClose}>✕</button>
        </header>

        <div className="sip-body">
          <nav className="sip-nav" aria-label="Season navigation">
            {STEPS.map((s, index) => (
              <button
                key={s.id}
                className={`sip-nav-btn ${active.id === s.id ? "active" : ""}`}
                onClick={() => setActive(s)}
              >
                <span className="sip-step-number">{index + 1}</span>
                <span>{s.title}</span>
              </button>
            ))}
          </nav>

          <main className="sip-main">
            <h3>{active.title}</h3>
            <p className="sip-summary">{active.summary}</p>

            <section className="sip-section">
              <h4>Practical Actions</h4>
              <ul className="sip-actions-list">
                {active.actions.map((a, i) => (
                  <li key={i} className="sip-action-item">{a}</li>
                ))}
              </ul>
            </section>
          </main>
        </div>

        <footer className="sip-footer">
          <button className="sip-primary" onClick={onClose}>
            Done
          </button>
        </footer>
      </div>
    </div>
  );
}
