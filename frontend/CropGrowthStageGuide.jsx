import React, { useState } from "react";
import { Sprout, Leaf, Wheat } from "lucide-react";
import "./CropGrowthStageGuide.css";

const STAGES = [
  {
    id: "seed",
    title: "Seed",
    Icon: Sprout,
    color: "#10b981",
    description: "Selecting high-quality seed and proper bed preparation.",
    care: [
      "Use certified seed and follow recommended spacing.",
      "Treat seed with recommended bio/chemical treatments if required.",
      "Prepare seedbed to ensure good soil-to-seed contact and drainage.",
    ],
    timeline: "Days 0-7",
  },
  {
    id: "sprout",
    title: "Sprout",
    Icon: Sprout,
    color: "#22c55e",
    description: "Germination and early root development.",
    care: [
      "Maintain gentle moisture — avoid waterlogging.",
      "Protect tender sprouts from birds and pests.",
      "Apply starter nutrients if soil test recommends.",
    ],
    timeline: "Days 7-21",
  },
  {
    id: "growth",
    title: "Growth",
    Icon: Leaf,
    color: "#0ea5e9",
    description: "Vegetative growth, canopy development, and nutrient uptake.",
    care: [
      "Follow a split-application fertilizer schedule based on NPK needs.",
      "Irrigate based on crop water requirements and soil moisture.",
      "Scout regularly for pest, disease, and nutrient deficiency signs.",
    ],
    timeline: "Days 21-60",
  },
  {
    id: "harvest",
    title: "Harvest",
    Icon: Wheat,
    color: "#f59e0b",
    description: "Physiological maturity and harvest operations.",
    care: [
      "Monitor grain/fruit maturity indices for optimal harvest time.",
      "Minimize field losses by planning harvest logistics in advance.",
      "Perform post-harvest drying and storage best practices.",
    ],
    timeline: "Days 60+",
  },
];

export default function CropGrowthStageGuide({ onClose }) {
  const [activeStage, setActiveStage] = useState(STAGES[0]);

  return (
    <div className="cgs-overlay" onClick={onClose} role="dialog" aria-modal="true">
      <div className="cgs-modal" onClick={(e) => e.stopPropagation()}>
        <header className="cgs-header">
          <div className="cgs-header-content">
            <h2>Crop Growth Stage Visual Guide</h2>
            <p className="cgs-subtitle">Seed → Sprout → Growth → Harvest lifecycle</p>
          </div>
          <button className="cgs-close" aria-label="Close" onClick={onClose}>✕</button>
        </header>

        <div className="cgs-body">
          <aside className="cgs-sidebar" aria-label="Growth stages">
            <nav className="cgs-timeline">
              {STAGES.map((s) => (
                <button
                  key={s.id}
                  className={`cgs-stage-item ${activeStage.id === s.id ? "active" : ""}`}
                  onClick={() => setActiveStage(s)}
                  style={{ "--stage-color": s.color }}
                >
                  <span className="cgs-stage-icon">
                    <s.Icon size={24} />
                  </span>
                  <div className="cgs-stage-info">
                    <span className="cgs-stage-title">{s.title}</span>
                    <span className="cgs-stage-time">{s.timeline}</span>
                  </div>
                </button>
              ))}
            </nav>
          </aside>

          <main className="cgs-main">
            <div className="cgs-visual-card">
              <div className="cgs-stage-illustration" style={{ backgroundColor: `${activeStage.color}15` }}>
                <activeStage.Icon size={120} style={{ color: activeStage.color }} />
              </div>
            </div>

            <div className="cgs-content">
              <h3 className="cgs-stage-heading">{activeStage.title}</h3>
              <p className="cgs-description">{activeStage.description}</p>

              <div className="cgs-care-section">
                <h4>Stage-wise Care</h4>
                <ul className="cgs-care-list">
                  {activeStage.care.map((c, i) => (
                    <li key={i}>
                      <span className="cgs-check">✓</span>
                      <span>{c}</span>
                    </li>
                  ))}
                </ul>
              </div>

              <div className="cgs-thumbnails">
                <p className="cgs-thumbs-label">Quick View:</p>
                <div className="cgs-thumbs-grid">
                  {STAGES.map((s) => (
                    <button
                      key={s.id}
                      className={`cgs-thumb ${activeStage.id === s.id ? "active" : ""}`}
                      onClick={() => setActiveStage(s)}
                      aria-label={`View ${s.title} stage`}
                    >
                      <div className="cgs-thumb-icon" style={{ backgroundColor: `${s.color}15` }}>
                        <s.Icon size={40} style={{ color: s.color }} />
                      </div>
                      <span>{s.title}</span>
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </main>
        </div>

        <footer className="cgs-footer">
          <button className="cgs-primary-btn" onClick={onClose}>Got it</button>
        </footer>
      </div>
    </div>
  );
}
