import React, { useState } from "react";
import "./CropGrowthStageGuide.css";

const STAGES = [
  {
    id: "seed",
    title: "Seed",
    image: "https://images.unsplash.com/photo-1501004318641-b39e6451bec6?auto=format&fit=crop&w=800&q=60",
    description: "Selecting high-quality seed and proper bed preparation.",
    care: [
      "Use certified seed and follow recommended spacing.",
      "Treat seed with recommended bio/chemical treatments if required.",
      "Prepare seedbed to ensure good soil-to-seed contact and drainage.",
    ],
  },
  {
    id: "sprout",
    title: "Sprout",
    image: "https://images.unsplash.com/photo-1498887964941-9c5b0a2d5f79?auto=format&fit=crop&w=800&q=60",
    description: "Germination and early root development.",
    care: [
      "Maintain gentle moisture — avoid waterlogging.",
      "Protect tender sprouts from birds and pests.",
      "Apply starter nutrients if soil test recommends.",
    ],
  },
  {
    id: "growth",
    title: "Growth",
    image: "https://images.unsplash.com/photo-1501004318641-4d1b8f9f2d12?auto=format&fit=crop&w=800&q=60",
    description: "Vegetative growth, canopy development, and nutrient uptake.",
    care: [
      "Follow a split-application fertilizer schedule based on NPK needs.",
      "Irrigate based on crop water requirements and soil moisture.",
      "Scout regularly for pest, disease, and nutrient deficiency signs.",
    ],
  },
  {
    id: "harvest",
    title: "Harvest",
    image: "https://images.unsplash.com/photo-1501004318641-1f6f8c9d3b0d?auto=format&fit=crop&w=800&q=60",
    description: "Physiological maturity and harvest operations.",
    care: [
      "Monitor grain/fruit maturity indices for optimal harvest time.",
      "Minimize field losses by planning harvest logistics in advance.",
      "Perform post-harvest drying and storage best practices.",
    ],
  },
];

export default function CropGrowthStageGuide({ onClose }) {
  const [activeStage, setActiveStage] = useState(STAGES[0]);
  const [showLightbox, setShowLightbox] = useState(false);

  return (
    <div className="cg-overlay" onClick={onClose} role="dialog" aria-modal="true">
      <div className="cg-modal" onClick={(e) => e.stopPropagation()}>
        <header className="cg-header">
          <h2>Crop Growth Stage Visual Guide</h2>
          <button className="cg-close" aria-label="Close" onClick={onClose}>✕</button>
        </header>

        <div className="cg-body">
          <nav className="cg-stages-nav" aria-label="Stages">
            {STAGES.map((s) => (
              <button
                key={s.id}
                className={`cg-stage-btn ${activeStage.id === s.id ? "active" : ""}`}
                onClick={() => setActiveStage(s)}
              >
                <img alt={s.title} src={`${s.image}&w=160&q=40`} />
                <span>{s.title}</span>
              </button>
            ))}
          </nav>

          <section className="cg-stage-detail">
            <div className="cg-stage-visual">
              <img
                src={`${activeStage.image}&w=900&q=70`}
                alt={activeStage.title}
                onClick={() => setShowLightbox(true)}
              />
              <button className="cg-enlarge" onClick={() => setShowLightbox(true)}>View Image</button>
            </div>

            <div className="cg-stage-info">
              <h3>{activeStage.title}</h3>
              <p className="cg-desc">{activeStage.description}</p>

              <h4>Stage-wise Care</h4>
              <ul>
                {activeStage.care.map((c, i) => (
                  <li key={i}>{c}</li>
                ))}
              </ul>

              <div className="cg-learning">
                <h4>Image-based Learning</h4>
                <p className="cg-learning-note">Tap a thumbnail to inspect real examples from the field.</p>
                <div className="cg-thumbs">
                  {STAGES.map((s) => (
                    <button key={s.id} className="cg-thumb" onClick={() => setActiveStage(s)}>
                      <img alt={s.title} src={`${s.image}&w=280&q=40`} />
                      <span>{s.title}</span>
                    </button>
                  ))}
                </div>
                <p className="cg-note">Tip: Use these images to train your eye on healthy vs stressed plants.</p>
              </div>
            </div>
          </section>
        </div>

        <footer className="cg-footer">
          <button className="cg-primary" onClick={onClose}>Done</button>
        </footer>

        {showLightbox && (
          <div className="cg-lightbox" onClick={() => setShowLightbox(false)}>
            <img src={`${activeStage.image}&w=1400&q=80`} alt={`${activeStage.title} large`} />
          </div>
        )}
      </div>
    </div>
  );
}
