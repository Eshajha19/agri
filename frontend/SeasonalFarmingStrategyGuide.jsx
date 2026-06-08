import React, { useMemo, useState } from "react";
import {
  CalendarDays,
  CloudRain,
  Droplets,
  Leaf,
  Sun,
  Sprout,
  ShieldAlert,
  CheckCircle2,
  ArrowRight,
  X,
} from "lucide-react";
import "./SeasonalFarmingStrategyGuide.css";

const SEASONS = [
  {
    id: "Kharif",
    label: "Kharif",
    timing: "June - October",
    icon: CloudRain,
    accent: "#2563eb",
    gradient: "linear-gradient(135deg, rgba(37, 99, 235, 0.14), rgba(14, 165, 233, 0.08))",
    summary:
      "Monsoon-driven farming season. Focus on timely sowing, drainage, and disease prevention while rainfall is active.",
    crops: ["Rice", "Maize", "Cotton", "Soybean", "Groundnut"],
    priorities: [
      "Prepare field drainage before the first heavy showers.",
      "Sow quickly after monsoon onset for better germination windows.",
      "Use split fertilizer doses and monitor fungal pressure closely.",
    ],
    irrigation: "Rely on rainfall first, then support only during dry spells or germination gaps.",
    risks: ["Waterlogging", "Fungal outbreaks", "Nutrient leaching"],
    checklist: [
      "Clear bunds and channels before sowing.",
      "Keep short-duration seed backups ready for re-sowing.",
      "Scout for pests after every major rain event.",
    ],
  },
  {
    id: "Rabi",
    label: "Rabi",
    timing: "November - March",
    icon: Sun,
    accent: "#ca8a04",
    gradient: "linear-gradient(135deg, rgba(202, 138, 4, 0.14), rgba(245, 158, 11, 0.08))",
    summary:
      "Cool-season farming period. Protect crops from frost, conserve soil moisture, and time irrigation carefully.",
    crops: ["Wheat", "Mustard", "Chickpea", "Barley", "Pea"],
    priorities: [
      "Keep soil moisture steady without over-irrigating.",
      "Delay sprays until dew lifts and winds are calm.",
      "Watch for frost nights and cold stress in tender crops.",
    ],
    irrigation: "Irrigate in planned intervals because cool weather slows evaporation and extends wet periods.",
    risks: ["Frost injury", "Water stress during jointing", "Late-season heat"],
    checklist: [
      "Use protective measures on cold mornings where frost is common.",
      "Plan critical irrigations around crown root and flowering stages.",
      "Store inputs early so harvest operations are not delayed.",
    ],
  },
  {
    id: "Zaid",
    label: "Zaid",
    timing: "March - June",
    icon: Sprout,
    accent: "#0f766e",
    gradient: "linear-gradient(135deg, rgba(15, 118, 110, 0.14), rgba(20, 184, 166, 0.08))",
    summary:
      "Short summer window between rabi and kharif. Choose quick varieties, conserve moisture, and avoid peak heat.",
    crops: ["Cucumber", "Watermelon", "Moong", "Fodder crops", "Vegetables"],
    priorities: [
      "Use early-morning irrigation to reduce evaporation loss.",
      "Prefer short-duration crops that can finish before monsoon onset.",
      "Mulch or shade where possible to retain moisture and lower stress.",
    ],
    irrigation: "Water is usually the limiting factor, so schedule light but timely irrigations.",
    risks: ["Heat stress", "Rapid moisture loss", "Short planning window"],
    checklist: [
      "Prepare land early so sowing is not delayed by heat.",
      "Avoid midday field work when temperatures peak.",
      "Harvest on time because late crops can be hit by early rains.",
    ],
  },
];

function SeasonIcon({ icon: Icon, accent }) {
  return (
    <span className="sfsg-season-icon" style={{ backgroundColor: `${accent}1a`, color: accent }}>
      <Icon size={22} />
    </span>
  );
}

export default function SeasonalFarmingStrategyGuide({ onClose }) {
  const [activeSeasonId, setActiveSeasonId] = useState(SEASONS[0].id);

  const activeSeason = useMemo(
    () => SEASONS.find((season) => season.id === activeSeasonId) || SEASONS[0],
    [activeSeasonId],
  );

  return (
    <div className="sfsg-modal" role="dialog" aria-modal="true" aria-labelledby="seasonal-farming-strategy-title">
      <header className="sfsg-header">
        <div className="sfsg-header-copy">
          <div className="sfsg-badge">
            <CalendarDays size={14} /> Seasonal strategy guide
          </div>
          <h2 id="seasonal-farming-strategy-title">Seasonal Farming Strategy Guide</h2>
          <p>
            Switch your field plan as the season changes. This guide highlights what matters most in Kharif, Rabi,
            and Zaid so you can match crops, irrigation, and risk controls to the calendar.
          </p>
        </div>
        <button className="sfsg-close" type="button" onClick={onClose} aria-label="Close seasonal farming strategy guide">
          <X size={20} />
        </button>
      </header>

      <div className="sfsg-layout">
        <aside className="sfsg-nav" aria-label="Season selection">
          {SEASONS.map((season) => {
            const isActive = season.id === activeSeason.id;
            return (
              <button
                key={season.id}
                type="button"
                className={`sfsg-nav-item ${isActive ? "active" : ""}`}
                onClick={() => setActiveSeasonId(season.id)}
                style={{ "--season-accent": season.accent }}
              >
                <SeasonIcon icon={season.icon} accent={season.accent} />
                <span className="sfsg-nav-copy">
                  <strong>{season.label}</strong>
                  <span>{season.timing}</span>
                </span>
              </button>
            );
          })}
        </aside>

        <main className="sfsg-content">
          <section className="sfsg-hero" style={{ background: activeSeason.gradient }}>
            <div className="sfsg-hero-copy">
              <div className="sfsg-hero-kicker">
                <activeSeason.icon size={16} /> {activeSeason.timing}
              </div>
              <h3>{activeSeason.label} strategy</h3>
              <p>{activeSeason.summary}</p>
            </div>
            <div className="sfsg-hero-panel">
              <div>
                <span className="sfsg-panel-label">Priority crops</span>
                <strong>{activeSeason.crops.join(" • ")}</strong>
              </div>
              <div>
                <span className="sfsg-panel-label">Irrigation posture</span>
                <strong>{activeSeason.irrigation}</strong>
              </div>
            </div>
          </section>

          <section className="sfsg-grid">
            <article className="sfsg-card">
              <h4><Leaf size={18} /> Core field priorities</h4>
              <ul className="sfsg-list">
                {activeSeason.priorities.map((item) => (
                  <li key={item}>
                    <CheckCircle2 size={16} />
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
            </article>

            <article className="sfsg-card">
              <h4><Droplets size={18} /> Water and timing</h4>
              <div className="sfsg-highlight">
                <p>{activeSeason.irrigation}</p>
              </div>
              <div className="sfsg-risks">
                <span className="sfsg-mini-label">Main risks</span>
                <div className="sfsg-chip-row">
                  {activeSeason.risks.map((risk) => (
                    <span key={risk} className="sfsg-chip sfsg-chip-warning">
                      <ShieldAlert size={13} /> {risk}
                    </span>
                  ))}
                </div>
              </div>
            </article>
          </section>

          <section className="sfsg-card sfsg-checklist-card">
            <h4><ArrowRight size={18} /> Quick checklist</h4>
            <div className="sfsg-checklist">
              {activeSeason.checklist.map((item, index) => (
                <div className="sfsg-checklist-item" key={item}>
                  <span className="sfsg-step">0{index + 1}</span>
                  <span>{item}</span>
                </div>
              ))}
            </div>
          </section>
        </main>
      </div>
    </div>
  );
}
