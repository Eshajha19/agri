export function predictYieldRuleBased(inputData = {}) {
  const crop = String(inputData.Crop || inputData.crop || "").trim();
  const season = String(inputData.Season || inputData.season || "").trim();
  const cropHeight = Number(inputData.CHeight || inputData.cHeight || inputData.cropHeight || 0);
  const transplantingMethod = String(inputData.CTransp || inputData.cTransp || inputData.transplantingMethod || "").trim();
  const irrigationType = String(inputData.IrriType || inputData.irriType || inputData.irrigationType || "").trim();
  const irrigationSource = String(inputData.IrriSource || inputData.irriSource || inputData.irrigationSource || "").trim();
  const irrigationCount = Number(inputData.IrriCount || inputData.irriCount || inputData.irrigationCount || 0);
  const waterCoverage = Number(inputData.WaterCov || inputData.waterCov || inputData.waterCoverage || 0);

  let yieldValue = 18;

  const cropAdjustment = {
    Paddy: 0,
    Cotton: -2,
    Maize: 2,
    "Bengal Gram": -3,
    Groundnut: -1,
    Chillies: -2,
    "Red Gram": -3
  };

  yieldValue += cropAdjustment[crop] || 0;

  if (season === "Kharif") {
    yieldValue += 1;
  } else if (season === "Rabi") {
    yieldValue += 0.5;
  }

  if (cropHeight >= 100) yieldValue += 1.5;
  else if (cropHeight >= 80) yieldValue += 1;
  else if (cropHeight >= 60) yieldValue += 0.5;
  else yieldValue -= 1;

  if (transplantingMethod === "Transplanting") {
    yieldValue += 1;
  }

  const irrigationAdjustment = {
    Flood: 1,
    Sprinkler: 1.5,
    Drip: 2,
    Surface: 0.5
  };

  yieldValue += irrigationAdjustment[irrigationType] || 0;

  const sourceAdjustment = {
    Groundwater: 0.5,
    Canal: 1,
    Rainfed: -2,
    Well: 0.5,
    Tubewell: 1
  };

  yieldValue += sourceAdjustment[irrigationSource] || 0;

  if (irrigationCount >= 8) yieldValue += 0.8;
  else if (irrigationCount >= 5) yieldValue += 0.4;
  else if (irrigationCount < 3) yieldValue -= 1;

  if (waterCoverage >= 85) yieldValue += 1;
  else if (waterCoverage >= 70) yieldValue += 0.5;
  else if (waterCoverage < 50) yieldValue -= 2;

  yieldValue = Math.max(5, Math.min(yieldValue, 40));

  return Number(yieldValue.toFixed(2));
}
