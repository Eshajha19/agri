"use client";
import React, { useState } from "react";
import "./CropGuide.css";

const cropsData = [
  {
    id: 1,
    name: "Rice",
    season: "Kharif",
    soil: "Clayey / Loamy",
    water: "High",
    duration: "120-150 days",
    yield: "20-30 quintals/acre",
    tips: "Requires standing water and high humidity",
  },
  {
    id: 2,
    name: "Wheat",
    season: "Rabi",
    soil: "Well-drained Loamy",
    water: "Medium",
    duration: "110-130 days",
    yield: "15-25 quintals/acre",
    tips: "Needs cool climate during growth",
  },
  {
    id: 3,
    name: "Maize",
    season: "Kharif",
    soil: "Alluvial",
    water: "Medium",
    duration: "90-110 days",
    yield: "18-28 quintals/acre",
    tips: "Avoid waterlogging",
  },
  {
    id: 4,
    name: "Sugarcane",
    season: "Year-round",
    soil: "Deep Loamy",
    water: "High",
    duration: "10-12 months",
    yield: "300-400 quintals/acre",
    tips: "Requires consistent irrigation",
  },
  {
    id: 5,
    name: "Cotton",
    season: "Kharif",
    soil: "Black Soil",
    water: "Medium",
    duration: "150-180 days",
    yield: "10-20 quintals/acre",
    tips: "Needs warm climate",
  },
  {
    id: 6,
    name: "Mustard",
    season: "Rabi",
    soil: "Sandy Loam",
    water: "Low",
    duration: "90-110 days",
    yield: "8-15 quintals/acre",
    tips: "Good for low rainfall areas",
  },
];

export default function CropGuide() {
  const [filter, setFilter] = useState("All");
  const [search, setSearch] = useState("");
  const [selectedCrop, setSelectedCrop] = useState(null);

  // FILTER + SEARCH
  const filteredCrops = cropsData.filter((crop) => {
    const matchesFilter =
      filter === "All" || crop.season === filter;

    const matchesSearch = crop.name
      .toLowerCase()
      .includes(search.toLowerCase());

    return matchesFilter && matchesSearch;
  });

  return (
    <div className="crop-page">

      {/* HEADER */}
      <div className="crop-hero">
        <h1>🌾 Crop Guide</h1>
        <p>Find the best crops based on season & soil</p>
      </div>

      {/* SEARCH */}
      <div className="crop-search">
        <input
          type="text"
          placeholder="Search crop..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
      </div>
        {/* HEADER */}
        <div className="crop-hero">
          <h1>🌾 <span className="notranslate">Crop Guide</span></h1>
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
              <p><strong>Water:</strong> {crop.water}</p>
            </div>

            <button onClick={() => setSelectedCrop(crop)}>
              View Details
            </button>
          </div>
        ))}
      </div>

      {/* MODAL */}
      {selectedCrop && (
        <div className="crop-modal">
          <div className="crop-popup">

            <button
              className="close-btn"
              onClick={() => setSelectedCrop(null)}
            >
              ✖
            </button>

            <h2>🌾 {selectedCrop.name}</h2>

            <p><strong>Season:</strong> {selectedCrop.season}</p>
            <p><strong>Soil:</strong> {selectedCrop.soil}</p>
            <p><strong>Water Requirement:</strong> {selectedCrop.water}</p>
            <p><strong>Duration:</strong> {selectedCrop.duration}</p>
            <p><strong>Expected Yield:</strong> {selectedCrop.yield}</p>

            <div className="tips">
              💡 {selectedCrop.tips}
            </div>

          </div>
        </div>
      )}
    </div>
  );
}