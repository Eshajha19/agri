import React, { useState } from "react";
import "./CropGuide.css";

const cropsData = [
  {
    id: 1,
    name: "Rice",
    season: "Kharif",
    soil: "Clayey / Loamy Soil",
  },
  {
    id: 2,
    name: "Wheat",
    season: "Rabi",
    soil: "Well-drained Loamy Soil",
  },
  {
    id: 3,
    name: "Maize",
    season: "Kharif",
    soil: "Alluvial Soil",
  },
  {
    id: 4,
    name: "Sugarcane",
    season: "Year-round",
    soil: "Deep Loamy Soil",
  },
  {
    id: 5,
    name: "Cotton",
    season: "Kharif",
    soil: "Black Soil",
  },
  {
    id: 6,
    name: "Mustard",
    season: "Rabi",
    soil: "Sandy Loam Soil",
  },
];

export default function CropGuide() {
  const [filter, setFilter] = useState("All");

  const filteredCrops = cropsData.filter((crop) => {
    if (filter === "All") return true;
    return crop.season === filter;
  });

  return (
    <div className="crop-page">

      {/* HEADER */}
      <div className="crop-hero">
        <h1>🌾 Crop Guide</h1>
        <p>Explore crops based on season and soil type</p>
      </div>

      {/* FILTER */}
      <div className="crop-filter">
        {["All", "Kharif", "Rabi", "Year-round"].map((item) => (
          <button
            key={item}
            className={filter === item ? "active" : ""}
            onClick={() => setFilter(item)}
          >
            {item}
          </button>
        ))}
      </div>

      {/* GRID */}
      <div className="crop-grid">
        {filteredCrops.map((crop) => (
          <div key={crop.id} className="crop-card">
            <div className="crop-icon">🌱</div>

            <h2>{crop.name}</h2>

            <div className="crop-info">
              <p><strong>Season:</strong> {crop.season}</p>
              <p><strong>Soil:</strong> {crop.soil}</p>
            </div>

            <button>View Details</button>
          </div>
        ))}
      </div>
    </div>
  );
}