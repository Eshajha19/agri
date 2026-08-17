const RULE_BASED_CROP_BASE_YIELD_KG_PER_HA = {
  rice: 3200,
  paddy: 3200,
  wheat: 3800,
  maize: 2800,
  corn: 2800,
  cotton: 550,
  sugarcane: 72000,
  potato: 22000,
  tomato: 25000,
  onion: 16000,
  soybean: 1800,
  groundnut: 1800,
  mustard: 1200,
  chickpea: 1100,
  pigeonpea: 900,
  barley: 3000,
  millet: 1500,
  bajra: 1200,
  jowar: 1400,
  lentil: 1000,
  apple: 15000,
  banana: 30000,
  mango: 12000,
  grapes: 20000,
  orange: 15000,
};

const RULE_BASED_SEASON_MULTIPLIER = {
  kharif: 1.1,
  rabi: 0.95,
  zaid: 0.85,
};

export function predictYieldRuleBased(inputData = {}) {
  const crop = String(inputData.Crop || inputData.crop || "").trim().toLowerCase();
  const season = String(inputData.Season || inputData.season || "").trim().toLowerCase();
  const area = Number(inputData.CropCoveredArea || inputData.cropCoveredArea || 1);
  const height = Number(inputData.CHeight || inputData.cHeight || 0);
  const irriCount = Number(inputData.IrriCount || inputData.irriCount || 0);
  const waterCov = Number(inputData.WaterCov || inputData.waterCov || 0);
  const cNext = String(inputData.CNext || inputData.cNext || "").trim().toLowerCase();
  const cLast = String(inputData.CLast || inputData.cLast || "").trim().toLowerCase();

  const baseYield = RULE_BASED_CROP_BASE_YIELD_KG_PER_HA[crop] || 2500;
  const seasonMult = RULE_BASED_SEASON_MULTIPLIER[season] || 1.0;
  const irriMult = 1.0 + Math.min(irriCount, 6) * 0.03;
  const waterMult = 1.0 + (waterCov / 100) * 0.15;

  let heightMult = 1.0;
  if (height > 0) {
    if (height < 30) heightMult = 0.85;
    else if (height > 120) heightMult = 1.1;
  }

  let rotationMult = 1.0;
  if (cLast && cNext && cLast !== cNext) {
    rotationMult = 1.05;
  }

  const predicted = baseYield * seasonMult * irriMult * waterMult * heightMult * rotationMult;
  return Math.round(predicted * 100) / 100;
}
