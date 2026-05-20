// Comprehensive crop disease database with treatments and pesticide recommendations
export const diseaseDatabase = {
  translations: {
    en: {
      healthy: "Healthy",
      leaf_blight: "Leaf Blight",
      powdery_mildew: "Powdery Mildew",
      rust: "Rust",
      bacterial_spot: "Bacterial Spot",
      early_blight: "Early Blight",
      late_blight: "Late Blight",
      septoria_leaf_spot: "Septoria Leaf Spot",
      target_spot: "Target Spot",
      mosaic_virus: "Mosaic Virus",
      downy_mildew: "Downy Mildew",
      anthracnose: "Anthracnose",
      root_rot: "Root Rot",
      leaf_spot: "Leaf Spot"
    },
    hi: {
      healthy: "स्वस्थ",
      leaf_blight: "पत्ती ब्लाइट",
      powdery_mildew: "खसखसी फफूंदी",
      rust: "जंग",
      bacterial_spot: "जीवाणु धब्बा",
      early_blight: "प्रारंभिक ब्लाइट",
      late_blight: "देर से ब्लाइट",
      septoria_leaf_spot: "सेप्टोरिया पत्ती धब्बा",
      target_spot: "लक्ष्य धब्बा",
      mosaic_virus: "मोज़ेक वायरस",
      downy_mildew: "डाउनी फफूंदी",
      anthracnose: "एंथ्रेकनोज",
      root_rot: "जड़ सड़न",
      leaf_spot: "पत्ती धब्बा"
    },
    te: {
      healthy: "ఆరోగ్యంగా",
      leaf_blight: "ఆకుల బ్లైట్",
      powdery_mildew: "పొడి తెగులు",
      rust: "తుప్పు",
      bacterial_spot: "బాక్టీరియా మచ్చ",
      early_blight: "ప్రారంభ బ్లైట్",
      late_blight: "ఆలస్యం బ్లైట్",
      septoria_leaf_spot: "సెప్టోరియా ఆకుల మచ్చ",
      target_spot: "లక్ష్య మచ్చ",
      mosaic_virus: "మొజాయిక్ వైరస్",
      downy_mildew: "డౌనీ తెగులు",
      anthracnose: "ఆంథ్రాక్నోజ్",
      root_rot: "వేరు కుళ్ళు",
      leaf_spot: "ఆకుల మచ్చ"
    }
  },
  
  treatments: {
    leaf_blight: {
      treatment: "Remove infected leaves, apply copper-based fungicide, ensure proper air circulation, avoid overhead watering",
      prevention: "Crop rotation, resistant varieties, balanced fertilization, proper spacing",
      pesticides: ["Copper hydroxide", "Mancozeb", "Chlorothalonil"],
      organic: ["Neem oil", "Baking soda spray", "Copper soap"]
    },
    powdery_mildew: {
      treatment: "Apply sulfur-based fungicide, increase air circulation, remove affected parts, reduce humidity",
      prevention: "Proper spacing, resistant varieties, avoid nitrogen excess, morning watering",
      pesticides: ["Sulfur", "Potassium bicarbonate", "Myclobutanil"],
      organic: ["Neem oil", "Milk spray", "Baking soda solution"]
    },
    rust: {
      treatment: "Remove infected plants, apply fungicide with active ingredient, avoid working with wet plants",
      prevention: "Resistant varieties, proper spacing, avoid overhead irrigation, sanitation",
      pesticides: ["Azoxystrobin", "Tebuconazole", "Mancozeb"],
      organic: ["Sulfur", "Copper sprays", "Neem oil"]
    },
    bacterial_spot: {
      treatment: "Copper-based bactericides, remove infected plant parts, avoid working with wet plants",
      prevention: "Disease-free seeds, crop rotation, proper sanitation, resistant varieties",
      pesticides: ["Copper hydroxide", "Streptomycin", "Fixed copper"],
      organic: ["Copper soap", "Bacillus subtilis", "Compost tea"]
    },
    early_blight: {
      treatment: "Apply fungicide at first sign, remove lower infected leaves, maintain plant vigor",
      prevention: "Crop rotation, resistant varieties, proper irrigation, balanced nutrition",
      pesticides: ["Chlorothalonil", "Mancozeb", "Copper hydroxide"],
      organic: ["Copper soap", "Neem oil", "Baking soda spray"]
    },
    late_blight: {
      treatment: "Apply metalaxyl-based fungicides, destroy infected plants, improve drainage",
      prevention: "Resistant varieties, proper spacing, avoid overhead watering, sanitation",
      pesticides: ["Metalaxyl", "Mefenoxam", "Copper hydroxide"],
      organic: ["Copper sprays", "Compost tea", "Bacillus subtilis"]
    },
    septoria_leaf_spot: {
      treatment: "Apply fungicide, remove infected leaves, improve air circulation",
      prevention: "Crop rotation, resistant varieties, proper spacing, sanitation",
      pesticides: ["Mancozeb", "Chlorothalonil", "Copper hydroxide"],
      organic: ["Neem oil", "Copper soap", "Baking soda spray"]
    },
    target_spot: {
      treatment: "Apply fungicide, remove infected leaves, maintain proper nutrition",
      prevention: "Resistant varieties, proper spacing, balanced fertilization",
      pesticides: ["Chlorothalonil", "Mancozeb", "Tebuconazole"],
      organic: ["Neem oil", "Copper sprays", "Bacillus subtilis"]
    },
    mosaic_virus: {
      treatment: "No cure, remove infected plants, control insect vectors",
      prevention: "Resistant varieties, insect control, sanitation, weed control",
      pesticides: ["Imidacloprid", "Dinotefuran", "Thiamethoxam"],
      organic: ["Neem oil", "Insecticidal soap", "Row covers"]
    },
    downy_mildew: {
      treatment: "Apply fungicide, improve air circulation, reduce humidity",
      prevention: "Resistant varieties, proper spacing, avoid overhead watering",
      pesticides: ["Metalaxyl", "Mefenoxam", "Copper hydroxide"],
      organic: ["Copper sprays", "Neem oil", "Baking soda solution"]
    },
    anthracnose: {
      treatment: "Apply fungicide, remove infected parts, improve sanitation",
      prevention: "Resistant varieties, crop rotation, proper irrigation",
      pesticides: ["Chlorothalonil", "Mancozeb", "Tebuconazole"],
      organic: ["Copper soap", "Neem oil", "Bacillus subtilis"]
    },
    root_rot: {
      treatment: "Improve drainage, apply fungicide, reduce watering, remove affected plants",
      prevention: "Well-draining soil, proper watering, resistant varieties, crop rotation",
      pesticides: ["Mefenoxam", "Metalaxyl", "Fosetyl-Al"],
      organic: ["Copper sprays", "Neem oil", "Beneficial microbes"]
    },
    leaf_spot: {
      treatment: "Apply fungicide, remove infected leaves, improve air circulation",
      prevention: "Resistant varieties, proper spacing, sanitation, balanced nutrition",
      pesticides: ["Mancozeb", "Chlorothalonil", "Copper hydroxide"],
      organic: ["Neem oil", "Copper soap", "Baking soda spray"]
    }
  }
};

export const getDiseaseInfo = (diseaseKey, language = 'en') => {
  const diseaseName = diseaseDatabase.translations[language]?.[diseaseKey] || 
                     diseaseDatabase.translations.en[diseaseKey] || diseaseKey;
  
  const treatment = diseaseDatabase.treatments[diseaseKey] || {
    treatment: "Consult local agricultural expert for specific treatment recommendations",
    prevention: "Follow good agricultural practices, maintain plant health",
    pesticides: ["Consult local expert"],
    organic: ["Neem oil", "Proper sanitation"]
  };

  return {
    disease: diseaseName,
    ...treatment
  };
};

export const saveDiseaseHistory = (detectionResult) => {
  try {
    const history = JSON.parse(localStorage.getItem('diseaseHistory') || '[]');

    // Store only a slim summary — NOT the full AI response.
    //
    // The original code spread the entire detectionResult object, which
    // includes free-text treatment/prevention strings from Gemini that can
    // be several KB each.  With 50 entries that could consume a significant
    // fraction of the 5–10 MB localStorage quota, causing QuotaExceededError
    // that silently broke every other feature using localStorage (farm
    // reports, WhatsApp settings, QR batches, language preferences).
    //
    // The history panel only displays disease name, confidence, method, and
    // date — so those are the only fields we need to persist.
    const newEntry = {
      // Use crypto.randomUUID() for collision-resistant IDs.
      // Date.now() as the id caused duplicate keys when two detections
      // completed within the same millisecond, causing React to silently
      // drop one entry from the rendered list.
      id: (typeof crypto !== 'undefined' && crypto.randomUUID)
        ? crypto.randomUUID()
        : `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
      timestamp: new Date().toISOString(),
      disease:    detectionResult.disease    || 'Unknown',
      confidence: detectionResult.confidence || 'Low',
      method:     detectionResult.method     || 'unknown',
    };

    history.unshift(newEntry);

    // Cap at 20 entries — enough for a useful history without risking
    // quota exhaustion even if disease names are long.
    if (history.length > 20) {
      history.splice(20);
    }

    localStorage.setItem('diseaseHistory', JSON.stringify(history));
    return newEntry;
  } catch (error) {
    // Re-throw QuotaExceededError so the caller can warn the user.
    // Swallowing it silently (the original behaviour) meant the user had
    // no idea their history wasn't saved AND that other localStorage writes
    // were also failing.
    if (
      error instanceof DOMException &&
      (error.name === 'QuotaExceededError' ||
       error.name === 'NS_ERROR_DOM_QUOTA_REACHED')
    ) {
      throw error;
    }
    // For other errors (e.g. JSON parse failure on corrupt data) return null
    // so the caller degrades gracefully.
    return null;
  }
};

export const getDiseaseHistory = () => {
  try {
    return JSON.parse(localStorage.getItem('diseaseHistory') || '[]');
  } catch (error) {
    return [];
  }
};

export const clearDiseaseHistory = () => {
  try {
    localStorage.removeItem('diseaseHistory');
    return true;
  } catch (error) {
    return false;
  }
};
