import React, { useMemo } from "react";
import { CloudRain, ThermometerSun, Wind, CalendarDays, Sprout, AlertTriangle, CheckCircle2, X } from "lucide-react";
import "./WeatherFarmingImpactGuide.css";

const RAIN_RULES = [
  {
    label: "0-2 mm",
    tone: "caution",
    title: "Dry spell stress",
    text: "Very low rain often means moisture stress for seedlings, leafy vegetables, and shallow-rooted crops. Increase monitoring and consider irrigation sooner.",
  },
  {
    label: "2-10 mm",
    tone: "good",
    title: "Useful rainfall",
    text: "Light to moderate rain usually supports soil moisture without major runoff. Good for germination, vegetative growth, and reducing irrigation demand.",
  },
  {
    label: "10-25 mm",
    tone: "caution",
    title: "Watch runoff",
    text: "Heavier rain can help deep-rooted crops but may delay field work, increase disease pressure, and wash away nutrients on exposed soils.",
  },
  {
    label: "25+ mm",
    tone: "critical",
    title: "Flood risk",
    text: "Very heavy rain can waterlog roots, trigger lodging in cereals, and spread fungal disease. Improve drainage and avoid fertilizer or spray applications.",
  },
];

const TEMPERATURE_RULES = [
  { range: "Below 12°C", tone: "caution", title: "Cold injury risk", text: "Slows germination and can damage sensitive crops or seedlings. Protect young plants and delay exposed irrigation where frost is possible." },
  { range: "12-18°C", tone: "caution", title: "Slow growth", text: "Most crops grow slowly in this range. Use it for hardier crops, but avoid transplanting tender seedlings in cold snaps." },
  { range: "18-32°C", tone: "good", title: "Comfort zone", text: "This is the most productive band for many crops. Irrigation, spraying, and nutrient uptake are usually more predictable here." },
  { range: "32-38°C", tone: "caution", title: "Heat stress", text: "Expect faster water loss, flower drop, and wilting. Irrigate earlier in the day and avoid spraying in peak heat." },
  { range: "Above 38°C", tone: "critical", title: "Extreme heat", text: "Can scorch leaves, reduce pollination, and cut yield sharply. Prioritize water, shade where possible, and avoid field operations at midday." },
];

const WIND_RULES = [
  { range: "Below 10 km/h", tone: "good", title: "Good for spraying", text: "Spray drift is usually low, so coverage stays more accurate on target leaves and stems." },
  { range: "10-18 km/h", tone: "caution", title: "Use caution", text: "Drift becomes more likely. Use larger droplets, lower boom height, and spray only when wind is steady." },
  { range: "Above 18 km/h", tone: "critical", title: "Postpone spraying", text: "Wind can carry chemicals off target, reduce efficacy, and increase drift damage to neighboring crops." },
];

function toneClass(tone) {
  return {
    good: "tone-good",
    caution: "tone-caution",
    critical: "tone-critical",
  }[tone] || "tone-caution";
}

function deriveSeasonAdvice(season) {
  const normalized = String(season || "").toLowerCase();
  if (normalized.includes("kharif")) {
    return [
      "Keep drainage channels open because monsoon showers can quickly waterlog fields.",
      "Expect higher fungal pressure and plan spray windows around dry spells.",
      "Use weather breaks to sow or transplant when soil moisture is workable, not saturated.",
    ];
  }
  if (normalized.includes("rabi")) {
    return [
      "Watch frost nights and protect tender crops during clear, cold mornings.",
      "Irrigate carefully because cool weather slows evaporation and can leave soils wet longer.",
      "Spraying is often safer in calm late mornings once dew has lifted.",
    ];
  }
  return [
    "Use summer heat windows to irrigate early and avoid midday field work.",
    "Short-duration crops and moisture conservation matter more during hot, dry spells.",
    "Watch for rapid pest buildup when temperatures rise and humidity remains moderate.",
  ];
}

export default function WeatherFarmingImpactGuide({ onClose, weatherSnapshot, season }) {
  const currentWeather = weatherSnapshot?.current || {};
  const dailyWeather = weatherSnapshot?.daily || {};

  const currentTemp = currentWeather.temperature_2m ?? null;
  const currentWind = currentWeather.wind_speed_10m ?? null;
  const currentHumidity = currentWeather.relative_humidity_2m ?? null;
  const nextRain = dailyWeather.precipitation_sum?.[0] ?? null;

  const currentRisk = useMemo(() => {
    if (currentWind !== null && currentWind > 18) return { label: "Spraying should wait", tone: "critical" };
    if (currentTemp !== null && currentTemp > 38) return { label: "Extreme heat stress", tone: "critical" };
    if (nextRain !== null && nextRain >= 25) return { label: "Flooding risk", tone: "critical" };
    if (currentTemp !== null && currentTemp >= 18 && currentTemp <= 32 && (currentWind === null || currentWind < 10)) {
      return { label: "Favorable field conditions", tone: "good" };
    }
    return { label: "Monitor conditions closely", tone: "caution" };
  }, [currentTemp, currentWind, nextRain]);

  const seasonAdvice = useMemo(() => deriveSeasonAdvice(season || weatherSnapshot?.season), [season, weatherSnapshot?.season]);

  return (
    <div className="weather-impact-modal">
      <div className="weather-impact-shell">
        <header className="weather-impact-header">
          <div>
            <div className={`impact-badge ${toneClass(currentRisk.tone)}`}>
              <Sprout size={14} /> Weather farming impact
            </div>
            <h2>Weather Farming Impact Guide</h2>
            <p>Quick decision rules for rain, temperature, wind, and seasonal farming actions.</p>
          </div>
          <button className="close-btn" onClick={onClose} aria-label="Close weather farming impact guide">
            <X size={20} />
          </button>
        </header>

        <section className={`current-status ${toneClass(currentRisk.tone)}`}>
          <div>
            <p className="section-label">Current read</p>
            <h3>{currentRisk.label}</h3>
          </div>
          <div className="status-pills">
            <span><ThermometerSun size={14} /> {currentTemp !== null ? `${Math.round(currentTemp)}°C` : "Temp n/a"}</span>
            <span><CloudRain size={14} /> {nextRain !== null ? `${Math.round(nextRain)} mm rain` : "Rain n/a"}</span>
            <span><Wind size={14} /> {currentWind !== null ? `${Math.round(currentWind)} km/h wind` : "Wind n/a"}</span>
            <span><CheckCircle2 size={14} /> {currentHumidity !== null ? `${Math.round(currentHumidity)}% humidity` : "Humidity n/a"}</span>
          </div>
        </section>

        <div className="rule-grid">
          <article className="rule-card">
            <h3><CloudRain size={18} /> Rain Impact on Crops</h3>
            <div className="rule-list">
              {RAIN_RULES.map((rule) => (
                <div key={rule.label} className={`rule-item ${toneClass(rule.tone)}`}>
                  <div className="rule-kicker">{rule.label}</div>
                  <div>
                    <strong>{rule.title}</strong>
                    <p>{rule.text}</p>
                  </div>
                </div>
              ))}
            </div>
          </article>

          <article className="rule-card">
            <h3><ThermometerSun size={18} /> Temperature Thresholds</h3>
            <div className="rule-list">
              {TEMPERATURE_RULES.map((rule) => (
                <div key={rule.range} className={`rule-item ${toneClass(rule.tone)}`}>
                  <div className="rule-kicker">{rule.range}</div>
                  <div>
                    <strong>{rule.title}</strong>
                    <p>{rule.text}</p>
                  </div>
                </div>
              ))}
            </div>
          </article>
        </div>

        <div className="rule-grid bottom-grid">
          <article className="rule-card">
            <h3><Wind size={18} /> Wind Effect on Spraying</h3>
            <div className="rule-list">
              {WIND_RULES.map((rule) => (
                <div key={rule.range} className={`rule-item ${toneClass(rule.tone)}`}>
                  <div className="rule-kicker">{rule.range}</div>
                  <div>
                    <strong>{rule.title}</strong>
                    <p>{rule.text}</p>
                  </div>
                </div>
              ))}
            </div>
          </article>

          <article className="rule-card">
            <h3><CalendarDays size={18} /> Seasonal Advisory Tips</h3>
            <div className="season-tips">
              {seasonAdvice.map((tip) => (
                <div key={tip} className="season-tip">
                  <AlertTriangle size={16} />
                  <span>{tip}</span>
                </div>
              ))}
            </div>
          </article>
        </div>
      </div>
    </div>
  );
}