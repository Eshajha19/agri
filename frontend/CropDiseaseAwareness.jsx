import React from "react";
import { FaShieldAlt, FaStethoscope, FaFlask, FaLeaf } from "react-icons/fa";
import "./CropDiseaseAwareness.css";

const diseaseData = [
  {
    id: 1,
    name: "Late Blight",
    crop: "Potato & Tomato",
    icon: "🥔",
    symptoms: "Water-soaked spots on leaves that turn brown/black. White fungal growth appears on leaf undersides in humid conditions. Fruits and tubers develop firm, brown decay.",
    prevention: "Use certified disease-free seeds. Avoid overhead irrigation to keep leaves dry. Maintain proper spacing for air circulation and practice crop rotation.",
    remedies: "Apply copper-based fungicides. Remove and destroy infected plants immediately. Use resistant varieties in areas prone to the disease.",
  },
  {
    id: 2,
    name: "Rice Blast",
    crop: "Rice",
    icon: "🌾",
    symptoms: "Diamond-shaped (spindle) lesions with gray or white centers and brown borders on leaves. Infected nodes turn black and break easily. Neck rot causes grain failure.",
    prevention: "Avoid excessive nitrogen fertilization. Maintain continuous flooding in fields. Treat seeds before sowing and use blast-resistant cultivars.",
    remedies: "Spray recommended fungicides like Tricyclazole or Carbendazim. Burn or bury infected crop residues after harvest to prevent spore survival.",
  },
  {
    id: 3,
    name: "Wheat Rust",
    crop: "Wheat",
    icon: "🍞",
    symptoms: "Orange-red or reddish-brown powdery pustules on leaves, leaf sheaths, and stems. Severely infected plants turn yellow and produce shriveled grains.",
    prevention: "Plant resistant varieties. Avoid early or late sowing beyond the optimal window. Monitor fields regularly for the first signs of pustules.",
    remedies: "Apply systemic fungicides such as Propiconazole or Tebuconazole. Ensure balanced nutrient application to maintain plant vigor.",
  },
  {
    id: 4,
    name: "Black Rot",
    crop: "Cabbage & Cauliflower",
    icon: "🥬",
    symptoms: "V-shaped yellow lesions appearing at the margins of the leaves. Veins within the yellowed areas turn black. Stems may show blackening when cut.",
    prevention: "Hot water treatment for seeds. Practice a 3-year crop rotation without cruciferous crops. Control weeds that may harbor the bacteria.",
    remedies: "Apply copper-based sprays during the growing season. Remove infected debris. Avoid working in fields when plants are wet.",
  },
  {
    id: 5,
    name: "Powdery Mildew",
    crop: "Many (Peas, Cucurbits, etc.)",
    icon: "🎃",
    symptoms: "White, powdery fungal growth on the surface of leaves, stems, and fruits. Affected leaves may turn yellow, curl, and drop prematurely.",
    prevention: "Ensure plants are in sunny locations with good air movement. Avoid overcrowding. Use resistant varieties whenever available.",
    remedies: "Spray with sulfur-based fungicides, neem oil, or a mixture of baking soda and water. Prune affected parts to improve light penetration.",
  },
  {
    id: 6,
    name: "Citrus Canker",
    crop: "Citrus Fruits",
    icon: "🍋",
    symptoms: "Raised, corky, brown lesions on leaves, twigs, and fruits, often surrounded by a yellow halo. Severe infection leads to premature leaf and fruit drop.",
    prevention: "Plant windbreaks to reduce bacterial spread. Use disease-free nursery stock. Disinfect pruning tools and equipment regularly.",
    remedies: "Apply copper-based bactericides as a preventive measure. Prune and destroy infected branches during the dry season.",
  }
];

const CropDiseaseAwareness = () => {
  return (
    <div className="disease-awareness-container">
      <header className="disease-header">
        <h1>Crop Disease Awareness 🌱</h1>
        <p>Identify, prevent, and treat common agricultural diseases to protect your harvest.</p>
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
                <h4><FaStethoscope /> Symptoms</h4>
                <p>{disease.symptoms}</p>
              </div>

              <div className="disease-section">
                <h4><FaShieldAlt /> Prevention</h4>
                <p>{disease.prevention}</p>
              </div>
            </div>
            <div className="disease-footer">
              <div className="remedy-badge">
                <FaFlask size={20} style={{ marginTop: '4px' }} />
                <p>
                  <strong>Basic Remedies:</strong>
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
