import React, { useState } from "react";
import {
  FaShieldAlt,
  FaStethoscope,
  FaFlask,
  FaSearch,
} from "react-icons/fa";
import "./CropDiseaseAwareness.css";

const diseaseData = [
  {
    id: 1,
    name: "Late Blight",
    crop: "Potato & Tomato",
    icon: "🥔",
    symptoms:
      "Water-soaked spots on leaves that turn brown/black. White fungal growth appears on leaf undersides in humid conditions. Fruits and tubers develop firm, brown decay.",
    prevention:
      "Use certified disease-free seeds. Avoid overhead irrigation. Maintain spacing and crop rotation.",
    remedies:
      "Apply copper fungicides. Remove infected plants. Use resistant varieties.",
  },
  {
    id: 2,
    name: "Rice Blast",
    crop: "Rice",
    icon: "🌾",
    symptoms:
      "Diamond-shaped lesions with gray centers and brown borders. Nodes weaken and grains fail.",
    prevention:
      "Avoid excess nitrogen. Maintain water levels. Use resistant seeds.",
    remedies:
      "Use Tricyclazole or Carbendazim. Destroy infected residues.",
  },
  {
    id: 3,
    name: "Wheat Rust",
    crop: "Wheat",
    icon: "🌿",
    symptoms:
      "Reddish-brown powdery pustules on leaves and stems causing shriveled grains.",
    prevention:
      "Use resistant varieties and proper sowing time.",
    remedies:
      "Spray Propiconazole or Tebuconazole.",
  },
  {
    id: 4,
    name: "Black Rot",
    crop: "Cabbage & Cauliflower",
    icon: "🥬",
    symptoms:
      "Yellow V-shaped lesions with black veins.",
    prevention:
      "Crop rotation and seed treatment.",
    remedies:
      "Copper sprays and removal of infected plants.",
  },
  {
    id: 5,
    name: "Powdery Mildew",
    crop: "Multiple Crops",
    icon: "🍃",
    symptoms:
      "White powdery growth on leaves causing yellowing.",
    prevention:
      "Ensure sunlight and airflow.",
    remedies:
      "Use sulfur sprays or neem oil.",
  },
  {
    id: 6,
    name: "Citrus Canker",
    crop: "Citrus",
    icon: "🍋",
    symptoms:
      "Brown corky lesions with yellow halo.",
    prevention:
      "Use disease-free plants and sanitize tools.",
    remedies:
      "Apply copper bactericides.",
  },
];

const CropDiseaseAwareness = () => {
  const [search, setSearch] = useState("");
  const [expandedId, setExpandedId] = useState(null);

  const toggleExpand = (id) => {
    setExpandedId(expandedId === id ? null : id);
  };

  const filteredData = diseaseData.filter((d) =>
    (d.name + d.crop).toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="disease-awareness-container">

      {/* HEADER */}
      <header className="disease-header">
        <h1>🌱 Crop Disease Awareness</h1>
        <p>Identify, prevent, and treat crop diseases effectively.</p>

        {/* SEARCH */}
        <div className="search-bar">
          <FaSearch />
          <input
            type="text"
            placeholder="Search disease or crop..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
      </header>

      {/* GRID */}
      <div className="disease-grid">
        {filteredData.length > 0 ? (
          filteredData.map((disease) => {
            const isExpanded = expandedId === disease.id;

            return (
              <div key={disease.id} className="disease-card">

                {/* ICON */}
                <div className="disease-icon">{disease.icon}</div>

                {/* CONTENT */}
                <div className="disease-content">
                  <span className="crop-tag">{disease.crop}</span>
                  <h2>{disease.name}</h2>

                  <div className="section">
                    <h4><FaStethoscope /> Symptoms</h4>
                    <p>
                      {isExpanded
                        ? disease.symptoms
                        : disease.symptoms.slice(0, 90) + "..."}
                    </p>
                  </div>

                  <div className="section">
                    <h4><FaShieldAlt /> Prevention</h4>
                    <p>
                      {isExpanded
                        ? disease.prevention
                        : disease.prevention.slice(0, 90) + "..."}
                    </p>
                  </div>

                  <button
                    className="toggle-btn"
                    onClick={() => toggleExpand(disease.id)}
                  >
                    {isExpanded ? "Show Less ▲" : "Read More ▼"}
                  </button>
                </div>

                {/* FOOTER */}
                <div className="disease-footer">
                  <div className="remedy">
                    <FaFlask />
                    <p>
                      <strong>Remedies:</strong>{" "}
                      {isExpanded
                        ? disease.remedies
                        : disease.remedies.slice(0, 80) + "..."}
                    </p>
                  </div>

                  <div className="actions">
                    <button className="primary">Learn More</button>
                    <button className="secondary">Save</button>
                  </div>
                </div>

              </div>
            );
          })
        ) : (
          <p className="no-results">No diseases found.</p>
        )}
      </div>
    </div>
  );
};

export default CropDiseaseAwareness;