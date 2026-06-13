import React, { useState } from "react";
import "./CropDiseaseLifecycleExplorer.css";

const CROPS = ["Wheat", "Rice", "Maize", "Potato", "Cotton"];

const DEFAULT_DATA = {
  Wheat: {
    disease: "Leaf Rust",
    stages: [
      { key: "early", title: "Early", icon: "🌱", note: "Small pustules on lower leaves; minimal yield impact if treated." },
      { key: "mid", title: "Mid", icon: "⚠️", note: "Larger lesions across canopy; spread accelerating." },
      { key: "severe", title: "Severe", icon: "💀", note: "Extensive pustulation, heavy defoliation and large yield loss." },
    ],
    prevention: [
      { when: "Pre-sowing / seedling", tips: ["Use resistant varieties", "Ensure good seed health"] },
      { when: "Vegetative", tips: ["Scout weekly", "Apply targeted foliar fungicide on first signs"] },
      { when: "Pre-harvest", tips: ["Avoid late heavy nitrogen that promotes susceptibility", "Monitor humidity and avoid irrigation that increases leaf wetness"] },
    ],
  },
  Rice: {
    disease: "Blast",
    stages: [
      { key: "early", title: "Early", icon: "🌱", note: "Small diamond-shaped spots on leaves." },
      { key: "mid", title: "Mid", icon: "⚠️", note: "Lesions enlarge and coalesce; panicle infection possible." },
      { key: "severe", title: "Severe", icon: "💀", note: "Neck rot and spikelet sterility; heavy losses." },
    ],
    prevention: [
      { when: "Nursery", tips: ["Avoid dense nursery sowing", "Treat seedlings if needed"] },
      { when: "Tillering", tips: ["Balanced N application", "Improve drainage to reduce standing water"] },
      { when: "Heading", tips: ["Apply protectant fungicide if risk is high", "Avoid high humidity around heading"] },
    ],
  },
};

export default function CropDiseaseLifecycleExplorer({ onClose }) {
  const [crop, setCrop] = useState(CROPS[0]);
  const data = DEFAULT_DATA[crop] || DEFAULT_DATA["Wheat"];
  const [activeStage, setActiveStage] = useState(data.stages[0]);

  const handleCropChange = (e) => {
    const next = e.target.value;
    setCrop(next);
    setActiveStage((DEFAULT_DATA[next] || DEFAULT_DATA["Wheat"]).stages[0]);
  };

  return (
    <div className="cdl-overlay" onClick={onClose} role="dialog" aria-modal="true">
      <div className="cdl-modal" onClick={(e) => e.stopPropagation()}>
        <header className="cdl-header">
          <h2>Crop Disease Lifecycle Explorer</h2>
          <button className="cdl-close" aria-label="Close" onClick={onClose}>✕</button>
        </header>

        <div className="cdl-controls">
          <label>
            Crop:
            <select value={crop} onChange={handleCropChange} aria-label="Select crop">
              {CROPS.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          </label>
        </div>

        <div className="cdl-body">
          <aside className="cdl-stages">
            {data.stages.map((s) => (
              <button
                key={s.key}
                className={`cdl-stage-btn ${activeStage.key === s.key ? 'active' : ''}`}
                onClick={() => setActiveStage(s)}
              >
                <span className="cdl-icon">{s.icon}</span>
                <span>{s.title}</span>
              </button>
            ))}
          </aside>

          <section className="cdl-detail">
            <div className="cdl-visual">
              <span className="cdl-detail-icon">{activeStage.icon}</span>
              <p className="cdl-note">{activeStage.note}</p>
            </div>

            <div className="cdl-info">
              <h3>{data.disease} — {activeStage.title} stage</h3>
              <div className="cdl-progression">
                <h4>Progression</h4>
                <p>Diseases typically start localized and expand under favorable conditions (high humidity, dense canopy, stressed plants). Early detection reduces spread and treatment costs.</p>
              </div>

              <div className="cdl-prevention">
                <h4>Prevention Timing Suggestions</h4>
                <ul>
                  {data.prevention.map((p, idx) => (
                    <li key={idx}><strong>{p.when}:</strong> {p.tips.join(' • ')}</li>
                  ))}
                </ul>
              </div>

              <div className="cdl-actions">
                <button className="cdl-primary" onClick={onClose}>Close</button>
              </div>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
