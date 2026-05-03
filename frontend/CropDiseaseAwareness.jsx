import React, { useState, useEffect } from "react";
import {
  FaShieldAlt,
  FaStethoscope,
  FaFlask,
  FaVolumeUp,
  FaBookmark,
} from "react-icons/fa";
import "./CropDiseaseAwareness.css";

const diseaseData = [
  {
    id: 1,
    name: "Late Blight",
    crop: "Potato",
    image: "https://upload.wikimedia.org/wikipedia/commons/6/60/Phytophthora_infestans_on_potato_leaf.jpg",
    severity: "High",
    season: "Monsoon",
    symptoms:
      "Water-soaked spots on leaves that turn brown/black. White fungal growth appears underneath.",
    prevention:
      "Use disease-free seeds, avoid overhead irrigation, maintain spacing.",
    remedies:
      "Apply copper fungicides, remove infected plants immediately.",
    action: "Remove infected leaves and spray fungicide within 24 hours.",
  },
  {
    id: 2,
    name: "Rice Blast",
    crop: "Rice",
    image: "https://upload.wikimedia.org/wikipedia/commons/3/3b/Rice_blast.jpg",
    severity: "High",
    season: "Kharif",
    symptoms:
      "Diamond-shaped lesions with gray centers. Affects leaves and nodes.",
    prevention:
      "Avoid excess nitrogen, maintain water level, use resistant seeds.",
    remedies:
      "Use Tricyclazole or Carbendazim spray.",
    action: "Spray fungicide immediately and control nitrogen usage.",
  },
  {
    id: 3,
    name: "Powdery Mildew",
    crop: "Cucurbits",
    image: "https://upload.wikimedia.org/wikipedia/commons/5/5c/Powdery_mildew.jpg",
    severity: "Medium",
    season: "Winter",
    symptoms:
      "White powdery growth on leaves and stems, leading to leaf drop.",
    prevention:
      "Ensure sunlight, avoid overcrowding.",
    remedies:
      "Use neem oil or sulfur spray.",
    action: "Spray neem oil weekly and remove infected parts.",
  },
];

const CropDiseaseAwareness = () => {
  const [search, setSearch] = useState("");
  const [filterCrop, setFilterCrop] = useState("All");
  const [openCard, setOpenCard] = useState(null);
  const [saved, setSaved] = useState([]);

  // Voice function
  const speak = (text) => {
    const msg = new SpeechSynthesisUtterance(text);
    window.speechSynthesis.speak(msg);
  };

  // Filter logic
  const filteredDiseases = diseaseData.filter((d) => {
    return (
      d.name.toLowerCase().includes(search.toLowerCase()) &&
      (filterCrop === "All" || d.crop === filterCrop)
    );
  });

  return (
    <div className="disease-awareness-container">
      <header className="disease-header">
        <h1>🌱 Crop Disease Awareness</h1>
        <p>Identify, prevent, and treat crop diseases effectively.</p>
      </header>

      {/* 🔍 Search + Filter */}
      <div className="disease-controls">
        <input
          type="text"
          placeholder="Search disease..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />

        <select onChange={(e) => setFilterCrop(e.target.value)}>
          <option>All</option>
          <option>Rice</option>
          <option>Potato</option>
          <option>Cucurbits</option>
        </select>
      </div>

      {/* 📸 AI Upload Section */}
      <div className="upload-section">
        <h3>📸 Detect Disease Using AI</h3>
        <input type="file" accept="image/*" />
        <button>Analyze Crop</button>
      </div>

      {/* 🌾 Disease Cards */}
      <div className="disease-grid">
        {filteredDiseases.map((disease) => (
          <div
            key={disease.id}
            className={`disease-card ${disease.severity.toLowerCase()}`}
          >
            <img src={disease.image} alt={disease.name} />

            <div className="disease-content">
              <span className="crop-tag">{disease.crop}</span>
              <span className={`severity ${disease.severity.toLowerCase()}`}>
                {disease.severity} Risk
              </span>
              <span className="season-tag">🌤 {disease.season}</span>

              <h2>{disease.name}</h2>

              {/* ⚡ Quick Action */}
              <div className="quick-action">
                ⚡ {disease.action}
              </div>

              {/* 🔘 Controls */}
              <div className="card-actions">
                <button onClick={() => speak(disease.symptoms)}>
                  <FaVolumeUp /> Listen
                </button>

                <button
                  onClick={() =>
                    setSaved((prev) =>
                      prev.includes(disease.id)
                        ? prev.filter((id) => id !== disease.id)
                        : [...prev, disease.id]
                    )
                  }
                >
                  <FaBookmark />{" "}
                  {saved.includes(disease.id) ? "Saved" : "Save"}
                </button>
              </div>

              {/* 📖 Expand */}
              <button
                className="expand-btn"
                onClick={() =>
                  setOpenCard(openCard === disease.id ? null : disease.id)
                }
              >
                {openCard === disease.id ? "Hide Details" : "View Details"}
              </button>

              {/* 📋 Details */}
              {openCard === disease.id && (
                <div className="disease-details">
                  <div>
                    <h4>
                      <FaStethoscope /> Symptoms
                    </h4>
                    <p>{disease.symptoms}</p>
                  </div>

                  <div>
                    <h4>
                      <FaShieldAlt /> Prevention
                    </h4>
                    <p>{disease.prevention}</p>
                  </div>

                  <div>
                    <h4>
                      <FaFlask /> Remedies
                    </h4>
                    <p>{disease.remedies}</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* ⭐ Saved Section */}
      {saved.length > 0 && (
        <div className="saved-section">
          <h3>⭐ Saved Diseases</h3>
          <p>You have saved {saved.length} diseases for quick access.</p>
        </div>
      )}
    </div>
  );
};

export default CropDiseaseAwareness;