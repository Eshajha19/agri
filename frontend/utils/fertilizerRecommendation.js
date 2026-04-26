const LEVEL_SCORE = {
  low: 1,
  medium: 2,
  high: 3,
};

const CROP_PROFILES = {
  Paddy: {
    nutrientTargets: { nitrogen: 3, phosphorus: 2, potassium: 2 },
    label: "Paddy",
    focus: "Nitrogen for vegetative growth and split application to reduce losses.",
    products: {
      nitrogen: "Urea",
      phosphorus: "DAP",
      potassium: "MOP",
    },
    organic: "Well-decomposed FYM or compost before puddling helps improve nutrient retention.",
  },
  Wheat: {
    nutrientTargets: { nitrogen: 3, phosphorus: 2, potassium: 1 },
    label: "Wheat",
    focus: "A balanced nitrogen and phosphorus plan with early top dressing.",
    products: {
      nitrogen: "Urea",
      phosphorus: "DAP",
      potassium: "MOP",
    },
    organic: "Add compost or farmyard manure at land preparation to reduce fertilizer stress.",
  },
  Maize: {
    nutrientTargets: { nitrogen: 3, phosphorus: 3, potassium: 2 },
    label: "Maize",
    focus: "Maize responds strongly to nitrogen, phosphorus, and early potassium support.",
    products: {
      nitrogen: "Urea",
      phosphorus: "DAP",
      potassium: "MOP",
    },
    organic: "Use FYM or vermicompost as a basal amendment for steady nutrient release.",
  },
  Cotton: {
    nutrientTargets: { nitrogen: 2, phosphorus: 2, potassium: 2 },
    label: "Cotton",
    focus: "Keep nitrogen moderate and prioritize potassium for boll development.",
    products: {
      nitrogen: "Urea",
      phosphorus: "DAP",
      potassium: "MOP",
    },
    organic: "Organic matter improves fiber quality and helps the crop handle dry spells.",
  },
  Groundnut: {
    nutrientTargets: { nitrogen: 1, phosphorus: 3, potassium: 2 },
    label: "Groundnut",
    focus: "Groundnut prefers low nitrogen and stronger phosphorus support for pod formation.",
    products: {
      nitrogen: "Rhizobium inoculant",
      phosphorus: "SSP",
      potassium: "MOP",
    },
    organic: "Gypsum and compost are useful for pod filling and soil structure.",
  },
  Vegetables: {
    nutrientTargets: { nitrogen: 3, phosphorus: 3, potassium: 3 },
    label: "Vegetables",
    focus: "Vegetable crops need a balanced, split nutrient plan with close monitoring.",
    products: {
      nitrogen: "Urea",
      phosphorus: "DAP",
      potassium: "MOP",
    },
    organic: "Frequent compost additions and mulching help keep nutrient availability stable.",
  },
};

const LEGUME_CROPS = new Set(["Groundnut", "Pea", "Lentil", "Bengal Gram", "Red Gram", "Soybean"]);

const BASE_DOSAGE_PER_ACRE = {
  nitrogen: 22,
  phosphorus: 18,
  potassium: 15,
};

function normaliseLevel(value) {
  const key = String(value || "").trim().toLowerCase();
  return LEVEL_SCORE[key] || LEVEL_SCORE.medium;
}

function roundDose(value) {
  return Math.max(0, Math.round(value / 5) * 5);
}

function getCropProfile(crop) {
  return CROP_PROFILES[crop] || CROP_PROFILES.Wheat;
}

export function generateFertilizerRecommendation(input = {}) {
  const crop = input.crop || "Wheat";
  const cropProfile = getCropProfile(crop);
  const acreage = Math.max(0.1, Number(input.acreage) || 1);
  const soilPH = Number(input.soilPH);
  const soilType = String(input.soilType || "Loamy");
  const season = String(input.season || "Rabi");
  const moisture = String(input.moisture || "Medium");

  const currentLevels = {
    nitrogen: normaliseLevel(input.nitrogen),
    phosphorus: normaliseLevel(input.phosphorus),
    potassium: normaliseLevel(input.potassium),
  };

  const products = [];

  Object.entries(cropProfile.nutrientTargets).forEach(([nutrient, targetLevel]) => {
    const gap = targetLevel - currentLevels[nutrient];
    if (gap <= 0) {
      return;
    }

    const productName = cropProfile.products[nutrient];
    const dose = roundDose(BASE_DOSAGE_PER_ACRE[nutrient] * gap * acreage);

    products.push({
      nutrient,
      productName,
      dose: `${dose} kg`,
      timing:
        nutrient === "nitrogen"
          ? "Split between basal dose and top dressing"
          : "Apply as a basal dose before the main irrigation",
      reason: `${nutrient.charAt(0).toUpperCase() + nutrient.slice(1)} is below the crop target.`,
    });
  });

  if (crop === "Groundnut" || LEGUME_CROPS.has(crop)) {
    const nitrogenEntry = products.find((item) => item.nutrient === "nitrogen");
    if (nitrogenEntry) {
      nitrogenEntry.productName = "Rhizobium inoculant";
      nitrogenEntry.dose = "Treat seed before sowing";
      nitrogenEntry.timing = "Seed inoculation before planting";
      nitrogenEntry.reason = "Legume crops fix nitrogen better when inoculated instead of over-fertilized.";
    }
  }

  const phAdvice = [];
  if (Number.isFinite(soilPH) && soilPH > 0) {
    if (soilPH < 5.8) {
      phAdvice.push("Apply agricultural lime or dolomite at 80-120 kg/acre to reduce acidity.");
    } else if (soilPH > 7.5) {
      phAdvice.push("Use gypsum and organic matter to improve nutrient uptake in alkaline soil.");
    } else {
      phAdvice.push("Soil pH is in a workable range, so nutrient uptake should be stable.");
    }
  }

  if (moisture.toLowerCase() === "low") {
    phAdvice.push("Split fertilizer doses and irrigate lightly after application to reduce volatilization.");
  }

  const fieldNotes = [];
  if (soilType === "Sandy") {
    fieldNotes.push("Sandy soil loses nutrients quickly, so split nitrogen and potassium into smaller doses.");
  } else if (soilType === "Clay") {
    fieldNotes.push("Clay soil retains nutrients longer, so avoid heavy early doses and watch for waterlogging.");
  } else if (soilType === "Black") {
    fieldNotes.push("Black soil often holds potassium well, but phosphorus still needs a careful basal plan.");
  } else {
    fieldNotes.push("Loamy soil is flexible, so balanced split dosing usually works well.");
  }

  if (season === "Kharif") {
    fieldNotes.push("Kharif rains can wash away nitrogen, so top dress after heavy rainfall has passed.");
  } else if (season === "Rabi") {
    fieldNotes.push("Rabi conditions are usually drier, so place basal nutrients close to the root zone.");
  } else {
    fieldNotes.push("Zaid crops usually grow in warmer months, so keep nutrient and moisture checks frequent.");
  }

  const summary =
    products.length === 0
      ? `Your ${cropProfile.label} field looks nutritionally balanced. Use maintenance nutrition and keep monitoring pH.`
      : `For ${acreage} acre(s) of ${cropProfile.label} in ${soilType.toLowerCase()} soil, focus on ${products.map((item) => item.productName).join(", ")} with split timing where needed.`;

  const priority =
    products.length === 0
      ? "Maintenance"
      : products.some((item) => item.nutrient === "nitrogen")
        ? "Nitrogen first"
        : "Balanced feeding";

  return {
    crop: cropProfile.label,
    summary,
    priority,
    focus: cropProfile.focus,
    products,
    phAdvice,
    fieldNotes,
    organicBoost: cropProfile.organic,
    acreage,
  };
}
