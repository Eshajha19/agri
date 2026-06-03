import React from "react";
import "./FarmingMythChecker.css";

const myths = [
  {
    myth: "More fertilizer = more yield",
    fact: "Excess fertilizer harms soil and reduces yield long-term.",
    verdict: "false",
    icon: "⚠️",
    color: "#dc2626",
    reference: "FAO - Fertilizer use and soil health"
  },
  {
    myth: "Drip irrigation always increases yield",
    fact: "It depends on crop type, soil, and water quality.",
    verdict: "depends",
    icon: "💧",
    color: "#d97706",
    reference: "International Water Management Institute"
  },
  {
    myth: "Organic farming cannot feed the world",
    fact: "Studies show organic methods can be productive with sustainable practices.",
    verdict: "false",
    icon: "🌱",
    color: "#dc2626",
    reference: "Nature Communications - Organic agriculture in the 21st century"
  },
  {
    myth: "Farmers don't need to rotate crops",
    fact: "Crop rotation prevents soil depletion and breaks pest cycles.",
    verdict: "false",
    icon: "🔄",
    color: "#dc2626",
    reference: "USDA - Benefits of Crop Rotation"
  },
  {
    myth: "All pesticides are harmful to the environment",
    fact: "Modern integrated pest management uses targeted, eco-friendly solutions.",
    verdict: "false",
    icon: "🐞",
    color: "#dc2626",
    reference: "EPA - Integrated Pest Management (IPM) Principles"
  },
  {
    myth: "Higher seed density always means higher yield",
    fact: "Overcrowding leads to competition for resources and lower yields.",
    verdict: "false",
    icon: "🌾",
    color: "#dc2626",
    reference: "Agronomy Journal - Plant Density and Yield"
  },
  {
    myth: "Bees only pollinate flowers, not crops",
    fact: "Many essential crops like almonds, apples, and blueberries rely heavily on bees for pollination.",
    verdict: "false",
    icon: "🐝",
    color: "#dc2626",
    reference: "FAO - The role of pollinators in agriculture"
  },
  {
    myth: "GMOs are inherently dangerous to human health",
    fact: "Scientific consensus from major health organizations states that GMOs on the market are safe.",
    verdict: "false",
    icon: "🧬",
    color: "#dc2626",
    reference: "WHO - Food safety: Genetically modified foods"
  },
  {
    myth: "Farming uses up all the world's fresh water",
    fact: "Agriculture uses 70% of global freshwater, but precision irrigation reduces waste.",
    verdict: "depends",
    icon: "💧",
    color: "#d97706",
    reference: "World Bank - Water in Agriculture"
  },
  {
    myth: "Raw milk is always healthier than pasteurized milk",
    fact: "Pasteurization kills harmful bacteria without significantly altering nutritional value.",
    verdict: "false",
    icon: "🥛",
    color: "#dc2626",
    reference: "CDC - Raw Milk Questions and Answers"
  },
  {
    myth: "Hydroponics produces less nutritious food than soil-grown crops",
    fact: "Nutrients in hydroponic systems can be precisely controlled, often matching soil-grown quality.",
    verdict: "false",
    icon: "🥗",
    color: "#dc2626",
    reference: "Harvard Health - Should you go hydroponic?"
  },
  {
    myth: "All modern farming is corporate-owned",
    fact: "Over 90% of farms in many countries (like the US) are still family-owned and operated.",
    verdict: "false",
    icon: "🏠",
    color: "#dc2626",
    reference: "USDA - Family Farms"
  },
  {
    myth: "Tilling the soil is always necessary for a good harvest",
    fact: "No-till farming preserves soil structure, reduces erosion, and improves water retention.",
    verdict: "false",
    icon: "🚜",
    color: "#dc2626",
    reference: "USDA NRCS - No-Till Farming"
  },
  {
    myth: "Pesticides are only used in conventional farming",
    fact: "Organic farming also uses pesticides, but they must be from natural sources.",
    verdict: "false",
    icon: "🌿",
    color: "#dc2626",
    reference: "USDA - Organic Standards"
  },
  {
    myth: "Brown eggs are more nutritious than white eggs",
    fact: "Egg color is determined by hen breed and does not affect nutritional value.",
    verdict: "false",
    icon: "🥚",
    color: "#dc2626",
    reference: "Egg Nutrition Center"
  },
  {
    myth: "Livestock production is the primary cause of climate change",
    fact: "While livestock emits GHGs, energy and transport are larger global contributors.",
    verdict: "false",
    icon: "🐄",
    color: "#dc2626",
    reference: "EPA - Global Greenhouse Gas Emissions Data"
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
              {item.reference && (
                <p className="reference-statement">
                  <strong>Source:</strong> <span className="reference-text">{item.reference}</span>
                </p>
              )}
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
