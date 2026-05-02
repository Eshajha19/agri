/**
 * Seasonal Crop Planning Logic
 * Provides detailed crop recommendations based on Location, Season, and Soil Type.
 */

const CROP_DATABASE = [
  // KHARIF CROPS
  {
    name: "Paddy (Rice)",
    seasons: ["Kharif"],
    soilTypes: ["Alluvial", "Clay", "Loamy"],
    regions: ["West Bengal", "Punjab", "Uttar Pradesh", "Andhra Pradesh", "Tamil Nadu", "Bihar", "Chhattisgarh", "Odisha"],
    benefits: "Staple food with stable market demand and government MSP support.",
    riskMitigation: "Use SRI (System of Rice Intensification) for water saving. Monitor for Stem Borer.",
    duration: "120-150 days"
  },
  {
    name: "Cotton",
    seasons: ["Kharif"],
    soilTypes: ["Black", "Alluvial", "Laterite"],
    regions: ["Gujarat", "Maharashtra", "Telangana", "Karnataka", "Punjab", "Haryana", "Rajasthan"],
    benefits: "High-value commercial fiber crop with excellent export potential.",
    riskMitigation: "Avoid sowing in waterlogged areas. Use pheromone traps for Pink Bollworm.",
    duration: "160-210 days"
  },
  {
    name: "Soybean",
    seasons: ["Kharif"],
    soilTypes: ["Black", "Loamy"],
    regions: ["Madhya Pradesh", "Maharashtra", "Rajasthan", "Karnataka", "Telangana"],
    benefits: "Short duration crop that improves soil nitrogen levels naturally.",
    riskMitigation: "Ensure seed treatment to prevent root rot. Harvest promptly to avoid pod shattering.",
    duration: "90-110 days"
  },
  {
    name: "Maize (Corn)",
    seasons: ["Kharif", "Rabi"],
    soilTypes: ["Alluvial", "Red", "Loamy", "Black"],
    regions: ["Karnataka", "Madhya Pradesh", "Bihar", "Tamil Nadu", "Maharashtra", "Rajasthan"],
    benefits: "Versatile usage in poultry feed, starch industry, and human consumption.",
    riskMitigation: "Apply balanced NPK. Use resistant varieties against Fall Armyworm.",
    duration: "90-120 days"
  },
  {
    name: "Bajra (Pearl Millet)",
    seasons: ["Kharif"],
    soilTypes: ["Sandy", "Red", "Alluvial"],
    regions: ["Rajasthan", "Maharashtra", "Gujarat", "Uttar Pradesh", "Haryana"],
    benefits: "Extremely drought-tolerant; performs well even in poor soil conditions.",
    riskMitigation: "Ensure proper spacing. Protect from birds during the grain-filling stage.",
    duration: "75-90 days"
  },
  {
    name: "Arhar (Pigeon Pea)",
    seasons: ["Kharif"],
    soilTypes: ["Black", "Alluvial", "Red", "Loamy"],
    regions: ["Maharashtra", "Karnataka", "Madhya Pradesh", "Uttar Pradesh", "Gujarat"],
    benefits: "Deep-rooted crop that breaks soil compaction; high protein value.",
    riskMitigation: "Manage Pod Borer at flowering stage. Avoid fields prone to wilting.",
    duration: "150-180 days"
  },

  // RABI CROPS
  {
    name: "Wheat",
    seasons: ["Rabi"],
    soilTypes: ["Alluvial", "Loamy", "Clay", "Black"],
    regions: ["Punjab", "Haryana", "Uttar Pradesh", "Madhya Pradesh", "Rajasthan", "Bihar"],
    benefits: "Primary winter staple; high productivity with irrigation support.",
    riskMitigation: "Ensure 'CRI' stage irrigation. Monitor for Yellow Rust in cooler regions.",
    duration: "120-140 days"
  },
  {
    name: "Mustard",
    seasons: ["Rabi"],
    soilTypes: ["Alluvial", "Sandy", "Loamy"],
    regions: ["Rajasthan", "Haryana", "Madhya Pradesh", "Uttar Pradesh", "West Bengal"],
    benefits: "Low water requirement; high oil content value in local markets.",
    riskMitigation: "Sow timely to avoid Aphid infestation during peak winter.",
    duration: "110-130 days"
  },
  {
    name: "Chickpea (Gram)",
    seasons: ["Rabi"],
    soilTypes: ["Black", "Loamy", "Alluvial", "Clay"],
    regions: ["Madhya Pradesh", "Maharashtra", "Rajasthan", "Uttar Pradesh", "Karnataka"],
    benefits: "Highest-selling pulse; thrives on residual soil moisture.",
    riskMitigation: "Use Wilt-resistant varieties (like JG-11). Avoid over-irrigation.",
    duration: "110-140 days"
  },
  {
    name: "Potato",
    seasons: ["Rabi"],
    soilTypes: ["Alluvial", "Sandy Loam", "Loamy"],
    regions: ["Uttar Pradesh", "West Bengal", "Bihar", "Gujarat", "Punjab"],
    benefits: "High yield potential per unit area; high commercial demand.",
    riskMitigation: "Monitor for Late Blight during foggy weather. Use certified seed tubers.",
    duration: "90-120 days"
  },
  {
    name: "Barley",
    seasons: ["Rabi"],
    soilTypes: ["Alluvial", "Sandy", "Loamy"],
    regions: ["Rajasthan", "Uttar Pradesh", "Haryana", "Madhya Pradesh"],
    benefits: "Can tolerate soil salinity better than wheat; used in malt industry.",
    riskMitigation: "Avoid excessive nitrogen application to prevent lodging.",
    duration: "110-130 days"
  },

  // ZAID CROPS
  {
    name: "Moong Dal (Green Gram)",
    seasons: ["Zaid", "Kharif"],
    soilTypes: ["Loamy", "Alluvial", "Red"],
    regions: ["Rajasthan", "Madhya Pradesh", "Maharashtra", "Punjab", "Tamil Nadu"],
    benefits: "Quick harvest; enriches soil with nitrogen before the next Kharif.",
    riskMitigation: "Manage Yellow Mosaic Virus through resistant seeds.",
    duration: "60-70 days"
  },
  {
    name: "Watermelon",
    seasons: ["Zaid"],
    soilTypes: ["Sandy", "Alluvial", "Sandy Loam"],
    regions: ["Uttar Pradesh", "Karnataka", "Tamil Nadu", "Maharashtra", "Andhra Pradesh"],
    benefits: "High cash return in summer; takes advantage of high temperatures.",
    riskMitigation: "Monitor for Fruit Fly. Ensure consistent but controlled watering.",
    duration: "80-90 days"
  },
  {
    name: "Cucumber",
    seasons: ["Zaid"],
    soilTypes: ["Sandy", "Loamy", "Alluvial"],
    regions: ["Haryana", "Uttar Pradesh", "Karnataka", "Delhi", "Punjab"],
    benefits: "Short duration vegetable crop with steady summer demand.",
    riskMitigation: "Provide mulch to keep fruits off the hot soil. Watch for Powdery Mildew.",
    duration: "50-60 days"
  },
  {
    name: "Sunflower",
    seasons: ["Zaid", "Rabi"],
    soilTypes: ["Black", "Alluvial", "Loamy"],
    regions: ["Karnataka", "Maharashtra", "Andhra Pradesh", "Punjab", "Haryana"],
    benefits: "High oil quality; can be grown in diverse soil conditions.",
    riskMitigation: "Protect from parrot damage during seed ripening. Ensure boron application.",
    duration: "90-105 days"
  },

  // ANNUAL / LONG DURATION
  {
    name: "Sugarcane",
    seasons: ["Annual"],
    soilTypes: ["Black", "Alluvial", "Red", "Clay"],
    regions: ["Uttar Pradesh", "Maharashtra", "Karnataka", "Tamil Nadu", "Gujarat", "Bihar"],
    benefits: "Industrial cash crop; provides long-term income and multiple ratoon crops.",
    riskMitigation: "Manage Red Rot and Early Shoot Borer. Requires high water security.",
    duration: "300-360 days"
  },
  {
    name: "Turmeric",
    seasons: ["Annual"],
    soilTypes: ["Red", "Loamy", "Laterite", "Alluvial"],
    regions: ["Andhra Pradesh", "Tamil Nadu", "Maharashtra", "Odisha", "Karnataka"],
    benefits: "High export value; medicinal and culinary demand is worldwide.",
    riskMitigation: "Ensure good drainage to prevent rhizome rot. Use raised bed planting.",
    duration: "210-270 days"
  },
  {
    name: "Ginger",
    seasons: ["Annual"],
    soilTypes: ["Loamy", "Red", "Laterite"],
    regions: ["Kerala", "Sikkim", "Meghalaya", "Karnataka", "West Bengal"],
    benefits: "High profitability; suitable for intercropping in young orchards.",
    riskMitigation: "Protect from Soft Rot. Requires shaded environment in early stages.",
    duration: "240-300 days"
  }
];

export const getCropRecommendations = (location, season, soilType) => {
  // Matching logic:
  // 1. Must match season (or be Annual)
  // 2. Score based on Soil and Region match
  const recommendations = CROP_DATABASE.filter(crop => {
    const seasonMatch = crop.seasons.includes(season) || crop.seasons.includes("Annual");
    return seasonMatch;
  });

  return recommendations.map(crop => {
    let score = 0;
    if (crop.soilTypes.includes(soilType)) score += 2;
    if (crop.regions.includes(location)) score += 2;
    
    // Bonus for common matches
    if (soilType === "Black" && crop.soilTypes.includes("Black")) score += 1;
    if (soilType === "Alluvial" && crop.soilTypes.includes("Alluvial")) score += 1;

    return { ...crop, score };
  })
  .filter(crop => crop.score > 0) // Only show relevant ones
  .sort((a, b) => b.score - a.score); // Sort by highest relevance
};

export const getYearlyCycle = (location, soilType) => {
  // Pick best for each season
  const kharif = getCropRecommendations(location, "Kharif", soilType).slice(0, 3);
  const rabi = getCropRecommendations(location, "Rabi", soilType).slice(0, 3);
  const zaid = getCropRecommendations(location, "Zaid", soilType).slice(0, 2);

  return {
    kharif,
    rabi,
    zaid
  };
};
