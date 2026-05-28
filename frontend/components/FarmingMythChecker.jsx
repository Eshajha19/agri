import React from "react";
import "./FarmingMythChecker.css";

const myths = [
  {
    myth: "More fertilizer = more yield",
    fact: "Excess fertilizer harms soil and reduces yield long-term.",
    verdict: "false",
    icon: "⚠️",
    color: "#dc2626"
  },
  {
    myth: "Drip irrigation always increases yield",
    fact: "It depends on crop type, soil, and water quality.",
    verdict: "depends",
    icon: "💧",
    color: "#d97706"
  },
  {
    myth: "Organic farming cannot feed the world",
    fact: "Studies show organic methods can be productive with sustainable practices.",
    verdict: "false",
    icon: "🌱",
    color: "#dc2626"
  },
  {
    myth: "Farmers don't need to rotate crops",
    fact: "Crop rotation prevents soil depletion and breaks pest cycles.",
    verdict: "false",
    icon: "🔄",
    color: "#dc2626"
  },
  {
    myth: "All pesticides are harmful to the environment",
    fact: "Modern integrated pest management uses targeted, eco-friendly solutions.",
    verdict: "false",
    icon: "🐞",
    color: "#dc2626"
  },
  {
    myth: "Higher seed density always means higher yield",
    fact: "Overcrowding leads to competition for resources and lower yields.",
    verdict: "false",
    icon: "🌾",
    color: "#dc2626"
  }
];

export default function FarmingMythChecker() {
  return (
    <div className="myth-checker-container">
      <h2>🌱 Farming Myth vs Fact Checker</h2>
      <p className="subtitle">
        Separating agricultural truth from tradition
      </p>
      <div className="myths-grid">
        {myths.map((item, idx) => (
          <div key={idx} className="myth-card">
            <div className="myth-header">
              <span className="myth-icon">{item.icon}</span>
              <h3>Myth #{idx + 1}</h3>
            </div>
            <div className="myth-body">
              <p className="myth-statement">
                <strong>Myth:</strong> {item.myth}
              </p>
              <p className="fact-statement">
                <strong>Fact:</strong> {item.fact}
              </p>
            </div>
            <div className={`myth-footer verdict-${item.verdict}`}>
              <span className="verdict-badge">
                {item.verdict === "false" ? "❌ Myth" : 
                 item.verdict === "true" ? "✅ Fact" : 
                 "⚠️ Depends"}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}