export function calculateSingleCropProfit(cost, yieldAmount, price) {
  if (cost < 0 || yieldAmount < 0 || price < 0) {
    throw new Error("Inputs cannot be negative");
  }

  const revenue = yieldAmount * price;
  const profit = revenue - cost;
  
  let profitPercentage = "0.00";
  if (cost > 0) {
    profitPercentage = ((profit / cost) * 100).toFixed(2);
  }

  return {
    cost,
    yield: yieldAmount,
    price,
    revenue,
    profit,
    profitPercentage,
  };
}

export function compareProfits(cropsArray) {
  if (!Array.isArray(cropsArray) || cropsArray.length === 0) {
    return { data: [], bestCropIds: [] };
  }

  const resultsArray = cropsArray.map((crop) => {
    const cost = parseFloat(crop.farmingCost) || 0;
    const yield_ = parseFloat(crop.expectedYield) || 0;
    const price = parseFloat(crop.marketPrice) || 0;
    
    const revenue = yield_ * price;
    const profit = revenue - cost;
    
    let profitPercentage = "0.00";
    if (cost > 0) {
      profitPercentage = ((profit / cost) * 100).toFixed(2);
    }
    
    return {
      id: crop.id,
      name: crop.cropName || "Unknown Crop",
      cost,
      yield: yield_,
      price,
      revenue,
      profit,
      profitPercentage,
    };
  });

  // Calculate the max profit
  let maxProfit = -Infinity;
  resultsArray.forEach(r => {
    if (r.profit > maxProfit) {
      maxProfit = r.profit;
    }
  });

  // Find all crops that match the max profit (if maxProfit > 0)
  const bestCropIds = [];
  if (maxProfit > 0) {
    resultsArray.forEach(r => {
      // Using a small epsilon to handle floating point comparisons
      if (Math.abs(r.profit - maxProfit) < 0.001) {
        bestCropIds.push(r.id);
      }
    });
  }

  return { data: resultsArray, bestCropIds };
}
