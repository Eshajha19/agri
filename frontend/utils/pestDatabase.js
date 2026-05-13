// Comprehensive pest database with treatment recommendations
export const pestDatabase = {
  translations: {
    en: {
      aphids: "Aphids",
      whiteflies: "Whiteflies",
      thrips: "Thrips",
      spider_mites: "Spider Mites",
      armyworms: "Armyworms",
      cutworms: "Cutworms",
      leaf_miners: "Leaf Miners",
      fruit_flies: "Fruit Flies",
      locusts: "Locusts",
      beetles: "Beetles",
      caterpillars: "Caterpillars",
      healthy: "Healthy"
    },
    hi: {
      aphids: "एफिड्स",
      whiteflies: "व्हाइटफ्लाइज",
      thrips: "थ्रिप्स",
      spider_mites: "स्पाइडर माइट्स",
      armyworms: "आर्मीवर्म्स",
      cutworms: "कटवर्म्स",
      leaf_miners: "लीफ माइनर्स",
      fruit_flies: "फ्रूट फ्लाइज",
      locusts: "टिड्डी",
      beetles: "बीटल्स",
      caterpillars: "कैटरपिलर्स",
      healthy: "स्वस्थ"
    },
    te: {
      aphids: "ఎఫిడ్స్",
      whiteflies: "వైట్‌ఫ్లైస్",
      thrips: "థ్రిప్స్",
      spider_mites: "స్పైడర్ మైట్స్",
      armyworms: "ఆర్మీవర్మ్స్",
      cutworms: "కట్‌వర్మ్స్",
      leaf_miners: "ఆకు మైనర్స్",
      fruit_flies: "పండ్ల ఈగలు",
      locusts: "తెల్లపల్లు",
      beetles: "బీటల్స్",
      caterpillars: "కాటర్‌పిల్లర్లు",
      healthy: "ఆరోగ్యంగా"
    }
  },

  treatments: {
    aphids: {
      description: "Small sap-sucking insects that cause leaf curling and stunted growth",
      damage: "Yellowing leaves, honeydew secretion, sooty mold growth",
      treatment: "Apply neem oil spray, introduce ladybugs, use insecticidal soap",
      prevention: "Monitor plants regularly, remove infested leaves, avoid over-fertilizing",
      chemical: ["Imidacloprid", "Pyrethrins", "Malathion"],
      organic: ["Neem oil", "Insecticidal soap", "Ladybug release"],
      lifecycle: "3-4 weeks from egg to adult, multiple generations per season"
    },
    whiteflies: {
      description: "Tiny white insects that feed on plant sap and transmit viruses",
      damage: "Yellowing leaves, stunted growth, honeydew deposits",
      treatment: "Use yellow sticky traps, apply reflective mulch, release beneficial insects",
      prevention: "Inspect new plants, quarantine new arrivals, maintain good air circulation",
      chemical: ["Imidacloprid", "Pyriproxyfen", "Bifenthrin"],
      organic: ["Neem oil", "Insecticidal soap", "Yellow sticky traps"],
      lifecycle: "3-4 weeks, rapid reproduction in warm conditions"
    },
    thrips: {
      description: "Small slender insects that scrape plant surfaces and feed on sap",
      damage: "Silvering of leaves, distorted growth, black fecal spots",
      treatment: "Use blue sticky traps, apply spinosad, maintain humidity",
      prevention: "Remove weeds, use reflective mulch, avoid over-watering",
      chemical: ["Spinosad", "Abamectin", "Methiocarb"],
      organic: ["Neem oil", "Beauveria bassiana", "Sticky traps"],
      lifecycle: "2-3 weeks, multiple generations annually"
    },
    spider_mites: {
      description: "Tiny arachnids that pierce plant cells and suck contents",
      damage: "Yellow stippling on leaves, fine webbing, leaf drop",
      treatment: "Increase humidity, apply miticides, release predatory mites",
      prevention: "Maintain adequate moisture, avoid dusty conditions, regular inspection",
      chemical: ["Abamectin", "Bifenthrin", "Spiromesifen"],
      organic: ["Neem oil", "Insecticidal soap", "Predatory mites"],
      lifecycle: "1-2 weeks, explosive population growth"
    },
    armyworms: {
      description: "Caterpillars that feed voraciously on leaves and fruits",
      damage: "Irregular holes in leaves, skeletonized foliage, fruit damage",
      treatment: "Hand pick larger caterpillars, apply Bt, use pheromone traps",
      prevention: "Till soil in fall, use row covers, monitor moth activity",
      chemical: ["Bacillus thuringiensis", "Spinosad", "Permethrin"],
      organic: ["Bt spray", "Hand removal", "Pheromone traps"],
      lifecycle: "3-4 weeks, multiple generations per season"
    },
    cutworms: {
      description: "Soil-dwelling caterpillars that cut seedlings at ground level",
      damage: "Severed seedlings, missing plants, wilted cut stems",
      treatment: "Create soil barriers, apply beneficial nematodes, use collars",
      prevention: "Till soil before planting, remove debris, use row covers",
      chemical: ["Carbaryl", "Permethrin", "Bifenthrin"],
      organic: ["Beneficial nematodes", "Diatomaceous earth", "Collars"],
      lifecycle: "6-8 weeks, one generation per year in most regions"
    },
    leaf_miners: {
      description: "Larvae that tunnel between leaf layers creating mines",
      damage: "Serpentine mines on leaves, reduced photosynthesis, leaf drop",
      treatment: "Remove affected leaves, apply systemic insecticides, use traps",
      prevention: "Monitor adult flies, use row covers, remove infested leaves",
      chemical: ["Spinosad", "Abamectin", "Imidacloprid"],
      organic: ["Neem oil", "Sticky traps", "Leaf removal"],
      lifecycle: "2-3 weeks, multiple generations per season"
    },
    fruit_flies: {
      description: "Small flies whose larvae feed inside fruits",
      damage: "Larvae in fruit, premature fruit drop, rotting fruits",
      treatment: "Use baited traps, remove infested fruits, apply cover sprays",
      prevention: "Sanitation, bagging fruits, monitoring with traps",
      chemical: ["Spinosad", "Malathion", "Diazinon"],
      organic: ["Baited traps", "Fruit bagging", "Sanitation"],
      lifecycle: "3-4 weeks, multiple generations in warm climates"
    },
    locusts: {
      description: "Large grasshoppers that can form devastating swarms",
      damage: "Complete defoliation, crop destruction, economic losses",
      treatment: "Apply ULV sprays, use barrier treatments, early intervention",
      prevention: "Monitor breeding grounds, soil treatment, community coordination",
      chemical: ["Fenitrothion", "Malathion", "Chlorpyrifos"],
      organic: ["Metarhizium anisopliae", "Beauveria bassiana", "Early detection"],
      lifecycle: "2-3 months, can produce 2-3 generations per year"
    },
    beetles: {
      description: "Hard-bodied insects with chewing mouthparts that damage various plant parts",
      damage: "Holes in leaves, damaged roots, feeding galleries",
      treatment: "Hand pick adults, apply appropriate insecticides, use traps",
      prevention: "Crop rotation, sanitation, monitoring adult activity",
      chemical: ["Carbaryl", "Permethrin", "Imidacloprid"],
      organic: ["Hand removal", "Beneficial nematodes", "Row covers"],
      lifecycle: "Variable, 1-3 months depending on species"
    },
    caterpillars: {
      description: "Larval stage of moths and butterflies that feed on foliage",
      damage: "Chewed leaves, defoliation, fruit damage",
      treatment: "Hand removal, Bt applications, pheromone traps",
      prevention: "Monitor egg masses, use row covers, encourage natural enemies",
      chemical: ["Bacillus thuringiensis", "Spinosad", "Permethrin"],
      organic: ["Bt spray", "Hand removal", "Natural predators"],
      lifecycle: "2-6 weeks, multiple generations per season"
    }
  }
};

export const getPestInfo = (pestKey, language = 'en') => {
  const pestName = pestDatabase.translations[language]?.[pestKey] || 
                 pestDatabase.translations.en[pestKey] || pestKey;
  
  const treatment = pestDatabase.treatments[pestKey] || {
    description: "Pest information not available",
    damage: "Damage patterns not documented",
    treatment: "Consult local agricultural expert",
    prevention: "Follow integrated pest management practices",
    chemical: ["Consult expert"],
    organic: ["Neem oil", "Proper sanitation"],
    lifecycle: "Lifecycle information not available"
  };

  return {
    pest: pestName,
    ...treatment
  };
};

export const savePestHistory = (detectionResult) => {
  try {
    const history = JSON.parse(localStorage.getItem('pestHistory') || '[]');
    const newEntry = {
      id: Date.now(),
      timestamp: new Date().toISOString(),
      ...detectionResult
    };
    history.unshift(newEntry);
    
    // Keep only last 50 entries
    if (history.length > 50) {
      history.splice(50);
    }
    
    localStorage.setItem('pestHistory', JSON.stringify(history));
    return newEntry;
  } catch (error) {
    return null;
  }
};

export const getPestHistory = () => {
  try {
    return JSON.parse(localStorage.getItem('pestHistory') || '[]');
  } catch (error) {
    return [];
  }
};

export const clearPestHistory = () => {
  try {
    localStorage.removeItem('pestHistory');
    return true;
  } catch (error) {
    return false;
  }
};

export const getRegionalAlerts = () => {
  // Simulated regional pest alerts - in real implementation, this would come from API
  return [
    {
      id: 1,
      pest: 'aphids',
      severity: 'medium',
      region: 'North India',
      description: 'Increased aphid activity reported in Punjab region',
      timestamp: new Date().toISOString()
    },
    {
      id: 2,
      pest: 'whiteflies',
      severity: 'high',
      region: 'South India',
      description: 'Whitefly outbreak detected in Karnataka',
      timestamp: new Date().toISOString()
    }
  ];
};
