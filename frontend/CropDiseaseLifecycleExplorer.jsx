import React, { useState } from "react";
import "./CropDiseaseLifecycleExplorer.css";

const CROPS = ["Wheat", "Rice", "Maize", "Potato", "Cotton"];

const DEFAULT_DATA = {
  Wheat: {
    disease: "Leaf Rust",
    stages: [
      { key: "early", title: "Early", image: "https://images.unsplash.com/photo-1501004318641-4d1b8f9f2d12?auto=format&fit=crop&w=1000&q=60", note: "Small pustules on lower leaves; minimal yield impact if treated." },
      { key: "mid", title: "Mid", image: "https://images.unsplash.com/photo-1501004318641-b39e6451bec6?auto=format&fit=crop&w=1000&q=60", note: "Larger lesions across canopy; spread accelerating." },
      { key: "severe", title: "Severe", image: "https://images.unsplash.com/photo-1501004318641-1f6f8c9d3b0d?auto=format&fit=crop&w=1000&q=60", note: "Extensive pustulation, heavy defoliation and large yield loss." },
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
      { key: "early", title: "Early", image: "https://images.unsplash.com/photo-1498887964941-9c5b0a2d5f79?auto=format&fit=crop&w=1000&q=60", note: "Small diamond-shaped spots on leaves." },
      { key: "mid", title: "Mid", image: "https://images.unsplash.com/photo-1501004318641-4d1b8f9f2d12?auto=format&fit=crop&w=1000&q=60", note: "Lesions enlarge and coalesce; panicle infection possible." },
      { key: "severe", title: "Severe", image: "https://images.unsplash.com/photo-1501004318641-b39e6451bec6?auto=format&fit=crop&w=1000&q=60", note: "Neck rot and spikelet sterility; heavy losses." },
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
  const [lightbox, setLightbox] = useState(false);

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
                <img src={`${s.image}&w=240&q=40`} alt={s.title} />
                <span>{s.title}</span>
              </button>
            ))}
          </aside>

          <section className="cdl-detail">
            <div className="cdl-visual">
              <img src={`${activeStage.image}&w=1000&q=70`} alt={`${activeStage.title}`} onClick={() => setLightbox(true)} />
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

        {lightbox && (
          <div className="cdl-lightbox" onClick={() => setLightbox(false)}>
            <img src={`${activeStage.image}&w=1400&q=80`} alt="Large stage" />
          </div>
        )}
      </div>
    </div>
  );
}
