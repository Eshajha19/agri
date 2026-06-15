import React, { useState } from "react";
import {
  FaShieldAlt,
  FaStethoscope,
  FaFlask,
  FaLeaf,
  FaBug,
  FaSearch,
  FaTint,
  FaSeedling,
} from "react-icons/fa";
import { Wheat, Carrot, Leaf, Banana, Apple, Citrus, Grape, Salad, Bean, Cherry } from "lucide-react";
import "./CropDiseaseAwareness.css";
import Recommendations from "./Recommendations";


const diseaseData = [
  {
    id: "late-blight",
    name: "Late Blight",
    crop: "Potato & Tomato",
    icon: <Carrot size={40} aria-hidden="true" />,
    cause: "Caused by the fungus-like pathogen Phytophthora infestans.",
    symptoms:
      "Water-soaked spots on leaves that turn brown or black. White fungal growth appears under leaves in humid weather. Tubers and fruits develop brown decay.",
    prevention:
      "Use disease-free seeds, avoid overhead irrigation, maintain plant spacing, and rotate crops regularly.",
    remedies:
      "Apply copper fungicides and remove infected plants immediately. Use resistant varieties where possible.",
  },
  {
    id: "rice-blast",
    name: "Rice Blast",
    crop: "Rice",
    icon: <Wheat size={40} aria-hidden="true" />,
    cause: "Caused by the fungus Magnaporthe oryzae.",
    symptoms:
      "Diamond-shaped lesions with gray centers on leaves. Neck rot leads to grain failure and weak stems.",
    prevention:
      "Avoid excess nitrogen fertilizer, maintain field flooding, and use resistant seed varieties.",
    remedies:
      "Spray Tricyclazole or Carbendazim. Destroy infected crop residues after harvest.",
  },
  {
    id: "wheat-rust",
    name: "Wheat Rust",
    crop: "Wheat",
    icon: <Wheat size={40} aria-hidden="true" />,
    cause: "Fungal infection spread by airborne spores.",
    symptoms:
      "Orange-red powdery pustules on leaves and stems. Severe infection causes yellowing and shriveled grains.",
    prevention:
      "Use resistant varieties and monitor fields regularly.",
    remedies:
      "Apply systemic fungicides like Propiconazole or Tebuconazole.",
  },
  {
    id: "black-rot",
    name: "Black Rot",
    crop: "Cabbage & Cauliflower",
    icon: <Salad size={40} aria-hidden="true" />,
    cause: "Bacterial disease caused by Xanthomonas campestris.",
    symptoms:
      "Yellow V-shaped lesions on leaves with blackened veins.",
    prevention:
      "Treat seeds with hot water and practice crop rotation.",
    remedies:
      "Use copper sprays and remove infected debris from fields.",
  },
  {
    id: "powdery-mildew",
    name: "Powdery Mildew",
    crop: "Peas & Cucurbits",
    icon: <Bean size={40} aria-hidden="true" />,
    cause: "Fungal disease thriving in warm dry climates.",
    symptoms:
      "White powdery growth on leaves and stems. Leaves curl and dry prematurely.",
    prevention:
      "Ensure sunlight exposure and proper airflow between plants.",
    remedies:
      "Apply sulfur fungicides, neem oil, or baking soda spray.",
  },
  {
    id: "citrus-canker",
    name: "Citrus Canker",
    crop: "Citrus Fruits",
    icon: <Citrus size={40} aria-hidden="true" />,
    cause: "Bacterial disease spread by wind and rain.",
    symptoms:
      "Raised corky lesions on fruits, leaves, and branches with yellow halos.",
    prevention:
      "Use disease-free nursery stock and disinfect tools.",
    remedies:
      "Apply copper bactericides and prune infected branches.",
  },
  {
    id: "downy-mildew",
    name: "Downy Mildew",
    crop: "Grapes & Vegetables",
    icon: <Grape size={40} aria-hidden="true" />,
    cause: "Fungal-like pathogen favored by cool, humid conditions.",
    symptoms:
      "Yellow patches on upper leaf surfaces with gray mold underneath.",
    prevention:
      "Improve air circulation and avoid excessive watering.",
    remedies:
      "Apply Mancozeb or Metalaxyl-based fungicides.",
  },
  {
    id: "bacterial-leaf-blight",
    name: "Bacterial Leaf Blight",
    crop: "Rice",
    icon: <Leaf size={40} aria-hidden="true" />,
    cause: "Bacterial infection caused by Xanthomonas oryzae.",
    symptoms:
      "Leaves turn yellow from tips downward and dry out.",
    prevention:
      "Use resistant varieties and balanced fertilizer application.",
    remedies:
      "Apply copper bactericides and avoid excessive nitrogen.",
  },
  {
    id: "anthracnose",
    name: "Anthracnose",
    crop: "Chili & Mango",
    icon: <Cherry size={40} aria-hidden="true" />,
    cause: "Fungal disease common during rainy weather.",
    symptoms:
      "Dark sunken lesions on fruits, leaves, and stems.",
    prevention:
      "Ensure proper drainage and avoid overhead irrigation.",
    remedies:
      "Use fungicides such as Chlorothalonil or Copper Oxychloride.",
  },
  {
    id: "root-rot",
    name: "Root Rot",
    crop: "Vegetables & Pulses",
    icon: <Carrot size={40} aria-hidden="true" />,
    cause: "Soil-borne fungi thriving in waterlogged soils.",
    symptoms:
      "Roots become brown and mushy. Plants wilt despite watering.",
    prevention:
      "Improve soil drainage and avoid overwatering.",
    remedies:
      "Treat soil with fungicides and remove infected plants.",
  },
  {
    id: "mosaic-virus",
    name: "Mosaic Virus",
    crop: "Tomato & Tobacco",
    icon: <Apple size={40} aria-hidden="true" />,
    cause: "Virus spread through insects and infected tools.",
    symptoms:
      "Mottled yellow-green leaf patterns with stunted growth.",
    prevention:
      "Control aphids and sanitize gardening tools.",
    remedies:
      "Remove infected plants immediately to stop spread.",
  },
  {
    id: "stem-borer",
    name: "Stem Borer",
    crop: "Rice & Maize",
    icon: <Wheat size={40} aria-hidden="true" />,
    cause: "Insect pest larvae boring into stems.",
    symptoms:
      "Dead hearts in young plants and hollow stems.",
    prevention:
      "Use pheromone traps and resistant crop varieties.",
    remedies:
      "Apply suitable insecticides and destroy crop residues.",
  },
  {
    id: "leaf-curl-disease",
    name: "Leaf Curl Disease",
    crop: "Chili & Tomato",
    icon: <Leaf size={40} aria-hidden="true" />,
    cause: "Virus transmitted by whiteflies.",
    symptoms:
      "Leaves curl upward and plants become stunted.",
    prevention:
      "Control whitefly populations and use healthy seedlings.",
    remedies:
      "Spray neem oil and remove infected plants.",
  },
  {
    id: "wilt-disease",
    name: "Wilt Disease",
    crop: "Banana & Tomato",
    icon: <Banana size={40} aria-hidden="true" />,
    cause: "Soil-borne fungal infection blocking water transport.",
    symptoms:
      "Sudden wilting and yellowing despite adequate moisture.",
    prevention:
      "Practice crop rotation and use disease-free planting material.",
    remedies:
      "Apply bio-fungicides and remove infected plants.",
  },
  {
    id: "scab-disease",
    name: "Scab Disease",
    crop: "Apple & Potato",
    icon: <Apple size={40} aria-hidden="true" />,
    cause: "Fungal infection affecting fruits and tubers.",
    symptoms:
      "Rough corky spots appear on fruit or potato skin.",
    prevention:
      "Use resistant varieties and avoid excess soil moisture.",
    remedies:
      "Apply preventive fungicides during early growth stages.",
  },
  {
    id: "fruit-rot",
    name: "Fruit Rot",
    crop: "Tomato & Papaya",
    icon: <Apple size={40} aria-hidden="true" />,
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
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCrop, setSelectedCrop] = useState("All");

  const crops = ["All", ...new Set(diseaseData.map(d => d.crop))];

  const filteredDiseases = diseaseData.filter(disease => {
    const matchesSearch = disease.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         disease.crop.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesCrop = selectedCrop === "All" || disease.crop === selectedCrop;
    return matchesSearch && matchesCrop;
  });

  return (
    <div className="disease-awareness-container">
      <header className="disease-header">
        <h1>
          <FaLeaf aria-hidden="true" /> Crop Disease Awareness
        </h1>

        <p>
          Learn about common crop diseases, their causes, prevention methods,
          and treatments to improve crop health and productivity.
        </p>

        <div className="disease-filters">
          <div className="search-wrapper">
            <FaSearch className="search-icon" aria-hidden="true" />
            <input
              type="text"
              placeholder="Search diseases or crops..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="disease-search"
              aria-label="Search diseases"
            />
          </div>

          <select
            value={selectedCrop}
            onChange={(e) => setSelectedCrop(e.target.value)}
            className="crop-filter"
            aria-label="Filter by crop"
          >
            {crops.map(crop => (
              <option key={crop} value={crop}>{crop}</option>
            ))}
          </select>
        </div>
      </header>

      <div className="disease-grid">
        {filteredDiseases.length > 0 ? (
          filteredDiseases.map((disease) => (
            <article key={disease.id} className="disease-card">
              <div className="disease-image-placeholder" aria-hidden="true">
                {disease.icon}
              </div>

              <div className="disease-content">
                <span className="crop-tag">{disease.crop}</span>

                <h2 className="disease-title">{disease.name}</h2>

                <section className="disease-section" aria-label="Cause">
                  <h4>
                    <FaBug aria-hidden="true" /> Cause
                  </h4>
                  <p>{disease.cause}</p>
                </section>

                <section className="disease-section" aria-label="Symptoms">
                  <h4>
                    <FaStethoscope aria-hidden="true" /> Symptoms
                  </h4>
                  <p>{disease.symptoms}</p>
                </section>

                <section className="disease-section" aria-label="Prevention">
                  <h4>
                    <FaShieldAlt aria-hidden="true" /> Prevention
                  </h4>
                  <p>{disease.prevention}</p>
                </section>
              </div>

              <footer className="disease-footer">
                <div className="remedy-badge">
                  <FaFlask size={20} aria-hidden="true" />
                  <p>
                    <strong>Recommended Treatment:</strong> {disease.remedies}
                  </p>
                </div>
              </footer>
            </article>
          ))
        ) : (
          <div className="no-results" role="status">
            No diseases found matching your criteria.
          </div>
        )}
      </div>
      <div>
      <h2>Crop Recommendations</h2>
      <Recommendations results={results} />
    </div>
    </div>
  );
};

export default CropDiseaseAwareness;
