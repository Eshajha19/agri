export const generateRecommendations = ({ weatherData, cropType, season }) => {
  const tips = [];

  if (!weatherData?.daily) return tips;

  const nextDays = weatherData.daily.slice(0, 3);

  const heavyRain = nextDays.some(day => day.pop > 0.7);
  const highTemp = nextDays.some(day => day.temp.max > 35);
  const coldTemp = nextDays.some(day => day.temp.min < 5);

  if (heavyRain) {
    tips.push({
      type: "warning",
      icon: "🌧️",
      title: "Heavy Rain Alert",
      text: "Avoid fertilizer spraying and ensure drainage.",
    });
  }

  if (highTemp) {
    tips.push({
      type: "heat",
      icon: "☀️",
      title: "Heat Stress Risk",
      text: "Irrigate early morning and use mulching.",
    });
  }

  if (coldTemp) {
    tips.push({
      type: "frost",
      icon: "❄️",
      title: "Frost Warning",
      text: "Cover crops overnight to prevent damage.",
    });
  }

  // Crop logic
  if (cropType === "Paddy") {
    tips.push({
      type: "crop",
      icon: "🌾",
      title: "Paddy Care",
      text: "Maintain 2–5 cm standing water.",
    });
  }

  if (cropType === "Wheat") {
    tips.push({
      type: "crop",
      icon: "🌱",
      title: "Wheat Care",
      text: "Monitor rust disease and irrigate at tillering stage.",
    });
  }

  // Season logic
  if (season === "Kharif") {
    tips.push({
      type: "season",
      icon: "🌧️",
      title: "Kharif Strategy",
      text: "Ensure proper drainage and weed control.",
    });
  }

  if (season === "Rabi") {
    tips.push({
      type: "season",
      icon: "❄️",
      title: "Rabi Strategy",
      text: "Focus on irrigation and frost protection.",
    });
  }

  const priority = {
    warning: 1,
    heat: 2,
    frost: 2,
    crop: 3,
    season: 4,
  };

  return tips.sort((a, b) => priority[a.type] - priority[b.type]);
};