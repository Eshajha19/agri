/**
 * Recommendation Engine
 *
 * Generates prioritised agricultural advisory tips based on:
 *   - Upcoming weather conditions (rain, heat, frost)
 *   - Crop type selected in the user's profile
 *   - Current farming season (Kharif / Rabi / Zaid)
 *
 * Tips are sorted by priority so the most urgent alerts appear first.
 */

/** Priority order: lower number = shown first */
const PRIORITY = {
  warning: 1,
  heat:    2,
  frost:   2,
  crop:    3,
  season:  4,
};

export const generateRecommendations = ({ weatherData, cropType, season }) => {
  const tips = [];

  if (!weatherData?.daily) return tips;

  // Normalize daily data to array format.
  // The weather snapshot from open-meteo uses an object with arrays (daily.time, etc.)
  // Some callers may pass an array already; handle both.
  let dailyArray;
  if (Array.isArray(weatherData.daily)) {
    dailyArray = weatherData.daily;
  } else if (weatherData.daily?.time && Array.isArray(weatherData.daily.time)) {
    const d = weatherData.daily;
    dailyArray = d.time.slice(0, 3).map((date, i) => ({
      date,
      maxTemp: d.temperature_2m_max?.[i] ?? null,
      minTemp: d.temperature_2m_min?.[i] ?? null,
      rain:    d.precipitation_sum?.[i] ?? 0,
    }));
  } else {
    return tips;
  }

  const nextDays = dailyArray.slice(0, 3);

  const heavyRain = nextDays.some(day => day.rain > 0);
  const highTemp  = nextDays.some(day => day.maxTemp > 35);
  const coldTemp  = nextDays.some(day => day.minTemp < 5);

  // ── Weather-based tips ────────────────────────────────────────────────────

  if (heavyRain) {
    tips.push({
      type: "warning",
      icon: "🌧️",
      title: "Heavy Rain Alert",
      text: "Avoid fertilizer spraying and ensure field drainage channels are clear.",
    });
  }

  if (highTemp) {
    tips.push({
      type: "heat",
      icon: "☀️",
      title: "Heat Stress Risk",
      text: "Irrigate early morning, apply mulching to retain soil moisture, and avoid transplanting at midday.",
    });
  }

  if (coldTemp) {
    tips.push({
      type: "frost",
      icon: "❄️",
      title: "Frost Warning",
      text: "Cover tender crops overnight and consider a light protective irrigation before dawn.",
    });
  }

  // ── Crop-specific tips ────────────────────────────────────────────────────

  if (cropType === "Paddy") {
    tips.push({
      type: "crop",
      icon: "🌾",
      title: "Paddy Care",
      text: "Maintain 2–5 cm standing water and monitor for blast disease during humid spells.",
    });
  }

  if (cropType === "Wheat") {
    tips.push({
      type: "crop",
      icon: "🌱",
      title: "Wheat Care",
      text: "Monitor for yellow rust and irrigate at the tillering and grain-filling stages for best yield.",
    });
  }

  if (cropType === "Cotton") {
    tips.push({
      type: "crop",
      icon: "🌿",
      title: "Cotton Care",
      text: "Scout for bollworm and whitefly regularly. Avoid excess nitrogen to reduce pest pressure.",
    });
  }

  if (cropType === "Maize") {
    tips.push({
      type: "crop",
      icon: "🌽",
      title: "Maize Care",
      text: "Ensure adequate moisture at silking stage and watch for fall armyworm in early growth.",
    });
  }

  if (cropType === "Soybean") {
    tips.push({
      type: "crop",
      icon: "🫘",
      title: "Soybean Care",
      text: "Apply Rhizobium seed treatment for nitrogen fixation and maintain weed-free conditions up to 30 days.",
    });
  }

  // ── Season-specific tips ──────────────────────────────────────────────────

  if (season === "Kharif") {
    tips.push({
      type: "season",
      icon: "🌧️",
      title: "Kharif Strategy",
      text: "Ensure proper field drainage, control weeds early, and monitor for fungal diseases during high humidity.",
    });
  }

  if (season === "Rabi") {
    tips.push({
      type: "season",
      icon: "❄️",
      title: "Rabi Strategy",
      text: "Focus on timely irrigation at critical growth stages and protect crops from frost during cold nights.",
    });
  }

  if (season === "Zaid") {
    tips.push({
      type: "season",
      icon: "☀️",
      title: "Zaid Strategy",
      text: "Prioritize water conservation with drip or sprinkler irrigation and choose short-duration heat-tolerant varieties.",
    });
  }

  // ── Sort by priority and return ───────────────────────────────────────────

  return tips.sort((a, b) => (PRIORITY[a.type] ?? 99) - (PRIORITY[b.type] ?? 99));
};
