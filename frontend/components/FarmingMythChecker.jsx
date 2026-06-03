import React, { useMemo, useState } from "react";
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

function verdictToLabel(verdict) {
  if (verdict === "true") return "✅ Fact";
  if (verdict === "false") return "❌ Myth";
  return "⚠️ Depends";
}

export default function FarmingMythChecker() {
  const [query, setQuery] = useState("");
  const [verdictFilter, setVerdictFilter] = useState("all");
  const [revealFacts, setRevealFacts] = useState(true);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();

    return myths
      .map((m, i) => ({ ...m, _idx: i }))
      .filter((m) => {
        if (verdictFilter !== "all" && m.verdict !== verdictFilter) return false;
        if (!q) return true;
        return (
          m.myth.toLowerCase().includes(q) ||
          m.fact.toLowerCase().includes(q)
        );
      });
  }, [query, verdictFilter]);

  return (
    <div className="myth-page">
      <div className="myth-hero">
        <div className="myth-hero__badge" aria-hidden="true">
          🌾
        </div>
        <div>
          <h2>Farming Myth vs Fact Checker</h2>
          <p className="myth-hero__subtitle">
            Separate agricultural truth from tradition—quickly.
          </p>
        </div>
      </div>

      <section className="myth-controls" aria-label="Myth checker controls">
        <div className="myth-control">
          <label htmlFor="myth-search" className="myth-control__label">
            Search
          </label>
          <input
            id="myth-search"
            className="myth-control__input"
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Try: fertilizer, drip, organic..."
            aria-label="Search myths"
          />
        </div>

        <div className="myth-control">
          <label htmlFor="myth-verdict" className="myth-control__label">
            Verdict
          </label>
          <select
            id="myth-verdict"
            className="myth-control__input"
            value={verdictFilter}
            onChange={(e) => setVerdictFilter(e.target.value)}
            aria-label="Filter by verdict"
          >
            <option value="all">All</option>
            <option value="false">❌ Myth</option>
            <option value="true">✅ Fact</option>
            <option value="depends">⚠️ Depends</option>
          </select>
        </div>

        <div className="myth-toggle" role="group" aria-label="Reveal facts">
          <button
            type="button"
            className={`myth-toggle__btn ${revealFacts ? "is-on" : ""}`}
            aria-pressed={revealFacts}
            onClick={() => setRevealFacts((v) => !v)}
          >
            <span className="myth-toggle__dot" aria-hidden="true" />
            <span>{revealFacts ? "Facts shown" : "Facts hidden"}</span>
          </button>
        </div>
      </section>

      <section className="myths-grid" aria-label="Myths list">
        {filtered.length === 0 ? (
          <div className="myth-empty" role="status" aria-live="polite">
            <div className="myth-empty__icon" aria-hidden="true">
              🔎
            </div>
            <h3>No matches</h3>
            <p>Try adjusting the search or verdict filter.</p>
          </div>
        ) : (
          filtered.map((item) => (
            <article key={item._idx} className="myth-card">
              <header className="myth-header">
                <span className="myth-icon" aria-hidden="true">
                  {item.icon}
                </span>
                <h3>Myth #{item._idx + 1}</h3>
              </header>

              <div className="myth-body">
                <p className="myth-statement">
                  <strong>Myth:</strong> {item.myth}
                </p>

                {revealFacts ? (
                  <p className="fact-statement">
                    <strong>Fact:</strong> {item.fact}
                  </p>
                ) : (
                  <p className="fact-statement myth-fact-hidden" aria-hidden="true">
                    <strong>Fact:</strong> (hidden)
                  </p>
                )}
              </div>

              <footer className={`myth-footer verdict-${item.verdict}`}>
                <span className="verdict-badge">{verdictToLabel(item.verdict)}</span>
              </footer>
            </article>
          ))
        )}
      </section>
    </div>
  );
}

