"use client";
import React, { useState, useMemo } from "react";
import "./CropGuide.css";

// 📦 DATA
const CROPS = [
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

const FILTERS = ["All", "Kharif", "Rabi", "Year-round"];

export default function CropGuide() {
  const [selectedSeason, setSelectedSeason] = useState("All");
  const [searchQuery, setSearchQuery] = useState("");
  const [activeCrop, setActiveCrop] = useState(null);

  // 🔍 FILTER + SEARCH (memoized for performance)
  const filteredCrops = useMemo(() => {
    return CROPS.filter((crop) => {
      const matchesSeason =
        selectedSeason === "All" || crop.season === selectedSeason;

      const matchesSearch = crop.name
        .toLowerCase()
        .includes(searchQuery.toLowerCase());

      return matchesSeason && matchesSearch;
    });
  }, [selectedSeason, searchQuery]);

  return (
    <div className="crop-page">

      {/* 🌾 HERO */}
      <header className="crop-hero">
        <h1>🌾 Crop Guide</h1>
        <p>Explore crops based on season, soil & water needs</p>
      </header>

      {/* 🔍 SEARCH */}
      <div className="crop-search">
        <input
          type="text"
          placeholder="Search crops..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
        />
      </div>

      {/* 🧭 FILTERS */}
      <div className="crop-filter">
        {FILTERS.map((season) => (
          <button
            key={season}
            className={selectedSeason === season ? "active" : ""}
            onClick={() => setSelectedSeason(season)}
          >
            {season}
          </button>
        ))}
      </div>

      {/* 🌱 GRID */}
      <div className="crop-grid">
        {filteredCrops.length > 0 ? (
          filteredCrops.map((crop) => (
            <div key={crop.id} className="crop-card">
              <div className="crop-icon">🌱</div>

              <h2>{crop.name}</h2>

              <div className="crop-info">
                <p><strong>Season:</strong> {crop.season}</p>
                <p><strong>Soil:</strong> {crop.soil}</p>
                <p><strong>Water:</strong> {crop.water}</p>
              </div>

              <button onClick={() => setActiveCrop(crop)}>
                View Details
              </button>
            </div>
          ))
        ) : (
          <p className="no-results">No crops found 🌾</p>
        )}
      </div>

      {/* 📋 MODAL */}
      {activeCrop && (
        <div className="crop-modal" onClick={() => setActiveCrop(null)}>
          <div
            className="crop-popup"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              className="close-btn"
              onClick={() => setActiveCrop(null)}
            >
              ✖
            </button>

            <h2>🌾 {activeCrop.name}</h2>

            <div className="modal-info">
              <p><strong>Season:</strong> {activeCrop.season}</p>
              <p><strong>Soil:</strong> {activeCrop.soil}</p>
              <p><strong>Water:</strong> {activeCrop.water}</p>
              <p><strong>Duration:</strong> {activeCrop.duration}</p>
              <p><strong>Yield:</strong> {activeCrop.yield}</p>
            </div>

            <div className="tips">
              💡 {activeCrop.tips}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}