import React, { useState } from "react";
import "./FarmingMythChecker.css";

export default function FarmingMythChecker() {
  const [search, setSearch] = useState("");
  const [expanded, setExpanded] = useState(null);

  const myths = [
    {
      myth: "More fertilizer = more yield",
      fact: "Excess fertilizer harms soil and reduces yield long-term.",
      verdict: "false",
      icon: "⚠️",
    },
    {
      myth: "Drip irrigation always increases yield",
      fact: "It depends on crop type, soil, and water quality.",
      verdict: "depends",
      icon: "💧",
    },
    {
      myth: "Organic farming cannot feed the world",
      fact: "Studies show organic methods can be productive with sustainable practices.",
      verdict: "false",
      icon: "🌱",
    },
    {
      myth: "Farmers don't need to rotate crops",
      fact: "Crop rotation prevents soil depletion and breaks pest cycles.",
      verdict: "false",
      icon: "🔄",
    },
  ];

  const filteredMyths = myths.filter(
    (item) =>
      item.myth.toLowerCase().includes(search.toLowerCase()) ||
      item.fact.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="myth-checker-container">
      <h2>🌱 Farming Myth vs Fact Checker</h2>
      <p className="subtitle">Separating agricultural truth from tradition</p>

      {/* Search bar */}
      <input
        type="text"
        placeholder="Search myths..."
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        className="search-bar"
      />

      <div className="myths-grid">
        {filteredMyths.map((item, idx) => (
          <div key={idx} className={`myth-card verdict-${item.verdict}`}>
            <div className="myth-header">
              <span className="myth-icon">{item.icon}</span>
              <h3>Myth #{idx + 1}</h3>
            </div>
            <div className="myth-body">
              <p className="myth-statement">
                <strong>Myth:</strong> {item.myth}
              </p>
              <p className="fact-statement">
                <strong>Fact:</strong>{" "}
                {expanded === idx ? item.fact : item.fact.slice(0, 60) + "..."}
              </p>
              <button
                className="expand-btn"
                onClick={() =>
                  setExpanded(expanded === idx ? null : idx)
                }
              >
                {expanded === idx ? "Show Less" : "Read More"}
              </button>
            </div>
            <div className={`myth-footer verdict-${item.verdict}`}>
              <span className="verdict-badge">
                {item.verdict === "false"
                  ? "❌ Myth"
                  : item.verdict === "true"
                  ? "✅ Fact"
                  : "⚠️ Depends"}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
