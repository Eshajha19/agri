import React, { useMemo, useState } from "react";
import "./FarmingMythChecker.css";

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
  {
    myth: "All pesticides are harmful to the environment",
    fact: "Modern integrated pest management uses targeted, eco-friendly solutions.",
    verdict: "false",
    icon: "🐞",
  },
  {
    myth: "Higher seed density always means higher yield",
    fact: "Overcrowding leads to competition for resources and lower yields.",
    verdict: "false",
    icon: "🌾",
  },
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

