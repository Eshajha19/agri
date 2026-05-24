import React from "react";
import {
  FaShieldAlt,
  FaStethoscope,
  FaFlask,
  FaLeaf,
  FaBug,
  FaTint,
  FaSeedling,
} from "react-icons/fa";
import "./CropDiseaseAwareness.css";

const diseaseData = [
  {
    id: 1,
    name: "Late Blight",
    crop: "Potato & Tomato",
    icon: "🥔",
    cause: "Caused by the fungus-like pathogen Phytophthora infestans.",
    symptoms:
      "Water-soaked spots on leaves that turn brown or black. White fungal growth appears under leaves in humid weather. Tubers and fruits develop brown decay.",
    prevention:
      "Use disease-free seeds, avoid overhead irrigation, maintain plant spacing, and rotate crops regularly.",
    remedies:
      "Apply copper fungicides and remove infected plants immediately. Use resistant varieties where possible.",
  },
  {
    id: 2,
    name: "Rice Blast",
    crop: "Rice",
    icon: "🌾",
    cause: "Caused by the fungus Magnaporthe oryzae.",
    symptoms:
      "Diamond-shaped lesions with gray centers on leaves. Neck rot leads to grain failure and weak stems.",
    prevention:
      "Avoid excess nitrogen fertilizer, maintain field flooding, and use resistant seed varieties.",
    remedies:
      "Spray Tricyclazole or Carbendazim. Destroy infected crop residues after harvest.",
  },
  {
    id: 3,
    name: "Wheat Rust",
    crop: "Wheat",
    icon: "🍞",
    cause: "Fungal infection spread by airborne spores.",
    symptoms:
      "Orange-red powdery pustules on leaves and stems. Severe infection causes yellowing and shriveled grains.",
    prevention:
      "Use resistant varieties and monitor fields regularly.",
    remedies:
      "Apply systemic fungicides like Propiconazole or Tebuconazole.",
  },
  {
    id: 4,
    name: "Black Rot",
    crop: "Cabbage & Cauliflower",
    icon: "🥬",
    cause: "Bacterial disease caused by Xanthomonas campestris.",
    symptoms:
      "Yellow V-shaped lesions on leaves with blackened veins.",
    prevention:
      "Treat seeds with hot water and practice crop rotation.",
    remedies:
      "Use copper sprays and remove infected debris from fields.",
  },
  {
    id: 5,
    name: "Powdery Mildew",
    crop: "Peas & Cucurbits",
    icon: "🎃",
    cause: "Fungal disease thriving in warm dry climates.",
    symptoms:
      "White powdery growth on leaves and stems. Leaves curl and dry prematurely.",
    prevention:
      "Ensure sunlight exposure and proper airflow between plants.",
    remedies:
      "Apply sulfur fungicides, neem oil, or baking soda spray.",
  },
  {
    id: 6,
    name: "Citrus Canker",
    crop: "Citrus Fruits",
    icon: "🍋",
    cause: "Bacterial disease spread by wind and rain.",
    symptoms:
      "Raised corky lesions on fruits, leaves, and branches with yellow halos.",
    prevention:
      "Use disease-free nursery stock and disinfect tools.",
    remedies:
      "Apply copper bactericides and prune infected branches.",
  },

  /* NEW DISEASE CARDS */

  {
    id: 7,
    name: "Downy Mildew",
    crop: "Grapes & Vegetables",
    icon: "🍇",
    cause: "Fungal-like pathogen favored by cool, humid conditions.",
    symptoms:
      "Yellow patches on upper leaf surfaces with gray mold underneath.",
    prevention:
      "Improve air circulation and avoid excessive watering.",
    remedies:
      "Apply Mancozeb or Metalaxyl-based fungicides.",
  },
  {
    id: 8,
    name: "Bacterial Leaf Blight",
    crop: "Rice",
    icon: "🌱",
    cause: "Bacterial infection caused by Xanthomonas oryzae.",
    symptoms:
      "Leaves turn yellow from tips downward and dry out.",
    prevention:
      "Use resistant varieties and balanced fertilizer application.",
    remedies:
      "Apply copper bactericides and avoid excessive nitrogen.",
  },
  {
    id: 9,
    name: "Anthracnose",
    crop: "Chili & Mango",
    icon: "🌶️",
    cause: "Fungal disease common during rainy weather.",
    symptoms:
      "Dark sunken lesions on fruits, leaves, and stems.",
    prevention:
      "Ensure proper drainage and avoid overhead irrigation.",
    remedies:
      "Use fungicides such as Chlorothalonil or Copper Oxychloride.",
  },
  {
    id: 10,
    name: "Root Rot",
    crop: "Vegetables & Pulses",
    icon: "🥕",
    cause: "Soil-borne fungi thriving in waterlogged soils.",
    symptoms:
      "Roots become brown and mushy. Plants wilt despite watering.",
    prevention:
      "Improve soil drainage and avoid overwatering.",
    remedies:
      "Treat soil with fungicides and remove infected plants.",
  },
  {
    id: 11,
    name: "Mosaic Virus",
    crop: "Tomato & Tobacco",
    icon: "🍅",
    cause: "Virus spread through insects and infected tools.",
    symptoms:
      "Mottled yellow-green leaf patterns with stunted growth.",
    prevention:
      "Control aphids and sanitize gardening tools.",
    remedies:
      "Remove infected plants immediately to stop spread.",
  },
  {
    id: 12,
    name: "Stem Borer",
    crop: "Rice & Maize",
    icon: "🌽",
    cause: "Insect pest larvae boring into stems.",
    symptoms:
      "Dead hearts in young plants and hollow stems.",
    prevention:
      "Use pheromone traps and resistant crop varieties.",
    remedies:
      "Apply suitable insecticides and destroy crop residues.",
  },
  {
    id: 13,
    name: "Leaf Curl Disease",
    crop: "Chili & Tomato",
    icon: "🍃",
    cause: "Virus transmitted by whiteflies.",
    symptoms:
      "Leaves curl upward and plants become stunted.",
    prevention:
      "Control whitefly populations and use healthy seedlings.",
    remedies:
      "Spray neem oil and remove infected plants.",
  },
  {
    id: 14,
    name: "Wilt Disease",
    crop: "Banana & Tomato",
    icon: "🍌",
    cause: "Soil-borne fungal infection blocking water transport.",
    symptoms:
      "Sudden wilting and yellowing despite adequate moisture.",
    prevention:
      "Practice crop rotation and use disease-free planting material.",
    remedies:
      "Apply bio-fungicides and remove infected plants.",
  },
  {
    id: 15,
    name: "Scab Disease",
    crop: "Apple & Potato",
    icon: "🍎",
    cause: "Fungal infection affecting fruits and tubers.",
    symptoms:
      "Rough corky spots appear on fruit or potato skin.",
    prevention:
      "Use resistant varieties and avoid excess soil moisture.",
    remedies:
      "Apply preventive fungicides during early growth stages.",
  },
  {
    id: 16,
    name: "Fruit Rot",
    crop: "Tomato & Papaya",
    icon: "🍍",
    cause: "Fungal pathogens attacking ripening fruits.",
    symptoms:
      "Soft watery decay with foul smell on fruits.",
    prevention:
      "Avoid fruit injury and maintain field sanitation.",
    remedies:
      "Remove infected fruits and apply fungicidal sprays.",
  },
];

const CropDiseaseAwareness = () => {
  return (
    <div className="disease-awareness-container">
      <header className="disease-header">
        <h1>
          <FaLeaf /> Crop Disease Awareness
        </h1>

        <p>
          Learn about common crop diseases, their causes, prevention methods,
          and treatments to improve crop health and productivity.
        </p>
      </header>

      <div className="disease-grid">
        {diseaseData.map((disease) => (
          <div key={disease.id} className="disease-card">
            <div className="disease-image-placeholder">
              {disease.icon}
            </div>

            <div className="disease-content">
              <span className="crop-tag">{disease.crop}</span>

              <h2 className="disease-title">{disease.name}</h2>

              <div className="disease-section">
                <h4>
                  <FaBug /> Cause
                </h4>
                <p>{disease.cause}</p>
              </div>

              <div className="disease-section">
                <h4>
                  <FaStethoscope /> Symptoms
                </h4>
                <p>{disease.symptoms}</p>
              </div>

              <div className="disease-section">
                <h4>
                  <FaShieldAlt /> Prevention
                </h4>
                <p>{disease.prevention}</p>
              </div>
            </div>

            <div className="disease-footer">
              <div className="remedy-badge">
                <FaFlask size={20} style={{ marginTop: "4px" }} />

                <p>
                  <strong>Recommended Treatment:</strong>{" "}
                  {disease.remedies}
                </p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};



export default CropDiseaseAwareness;