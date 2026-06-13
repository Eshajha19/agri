"use client";
import React, { useState, useMemo, useEffect } from "react";
import "./CropGuide.css";
import { getBookmarks, toggleBookmark } from "./utils/bookmarkStorage";
import { Wheat, Lightbulb, X } from "lucide-react";

// 🖼️ DIRECT PUBLIC FOLDER TRACKING
const CROP_IMAGES = {
  Rice: "/crops/Rice.webp",
  Potato: "/crops/potato.webp",
  Turmeric: "/crops/turmeric.webp",
  Peas: "/crops/peas.webp",
  Groundnut: "/crops/groundnut.webp",
  Chickpea: "/crops/chickpea.webp",
  Wheat: "/crops/wheat.webp",
  Cotton: "/crops/cotton.webp",
  Mustard: "/crops/mustard.webp",
  Tomato: "/crops/tomato.webp",
  Barley: "/crops/barley.webp",
  Sunflower: "/crops/sunflower.webp",
  Onion: "/crops/onion.webp",
  Maize: "/crops/maize.webp",
  Sugarcane: "/crops/sugarcane.webp",
  Soybean: "/crops/soybean.webp",
};

// 📦 DATA
const CROPS = [
  { id: 1, name: "Rice", season: "Kharif", soil: "Clayey / Loamy", water: "High", duration: "120-150 days", yield: "20-30 quintals/acre", tips: "Requires standing water and high humidity" },
  { id: 2, name: "Wheat", season: "Rabi", soil: "Well-drained Loamy", water: "Medium", duration: "110-130 days", yield: "15-25 quintals/acre", tips: "Needs cool climate during growth" },
  { id: 3, name: "Maize", season: "Kharif", soil: "Alluvial", water: "Medium", duration: "90-110 days", yield: "18-28 quintals/acre", tips: "Avoid waterlogging" },
  { id: 4, name: "Sugarcane", season: "Year-round", soil: "Deep Loamy", water: "High", duration: "10-12 months", yield: "300-400 quintals/acre", tips: "Requires consistent irrigation" },
  { id: 5, name: "Cotton", season: "Kharif", soil: "Black Soil", water: "Medium", duration: "150-180 days", yield: "10-20 quintals/acre", tips: "Needs warm climate" },
  { id: 6, name: "Mustard", season: "Rabi", soil: "Sandy Loam", water: "Low", duration: "90-110 days", yield: "8-15 quintals/acre", tips: "Good for low rainfall areas" },
  { id: 7, name: "Tomato", season: "Year-round", soil: "Loamy", water: "Medium", duration: "90-120 days", yield: "25-35 tons/hectare", tips: "Requires regular watering and sunlight" },
  { id: 8, name: "Potato", season: "Rabi", soil: "Sandy Loam", water: "Medium", duration: "80-100 days", yield: "20-25 tons/hectare", tips: "Avoid excessive waterlogging" },
  { id: 9, name: "Barley", season: "Rabi", soil: "Loamy", water: "Low", duration: "90-110 days", yield: "18-22 quintals/acre", tips: "Suitable for dry and cool climates" },
  { id: 10, name: "Turmeric", season: "Kharif", soil: "Well-drained Loamy", water: "Medium", duration: "210-300 days", yield: "20-25 tons/hectare", tips: "Requires warm and humid climate conditions" },
  { id: 11, name: "Peas", season: "Rabi", soil: "Clay Loam", water: "Low", duration: "60-90 days", yield: "8-10 quintals/acre", tips: "Grows best in cool weather with moderate irrigation" },
  { id: 12, name: "Groundnut", season: "Kharif", soil: "Sandy Loam", water: "Medium", duration: "120-140 days", yield: "15-20 quintals/acre", tips: "Requires warm climate and well-drained soil" },
  { id: 13, name: "Soybean", season: "Kharif", soil: "Loamy", water: "Medium", duration: "90-120 days", yield: "10-15 quintals/acre", tips: "Needs moderate rainfall and fertile soil" },
  { id: 14, name: "Chickpea", season: "Rabi", soil: "Sandy Loam", water: "Low", duration: "100-120 days", yield: "8-12 quintals/acre", tips: "Grows best in cool and dry climates" },
  { id: 15, name: "Sunflower", season: "Year-round", soil: "Loamy", water: "Medium", duration: "80-100 days", yield: "7-10 quintals/acre", tips: "Requires full sunlight for better yield" },
  { id: 16, name: "Onion", season: "Rabi", soil: "Silty Loam", water: "Medium", duration: "100-150 days", yield: "100-120 quintals/acre", tips: "Needs regular irrigation during bulb formation" }
];

const FILTERS = ["All", "Kharif", "Rabi", "Year-round"];

export default function CropGuide() {
  const [selectedSeason, setSelectedSeason] = useState("All");
  const [searchQuery, setSearchQuery] = useState("");
  const [activeCrop, setActiveCrop] = useState(null);
  const [bookmarkedCropIds, setBookmarkedCropIds] = useState(() =>
    getBookmarks("crops").map((crop) => crop.id),
  );

  useEffect(() => {
    setBookmarkedCropIds(getBookmarks("crops").map((crop) => crop.id));
  }, []);

  const handleToggleCropBookmark = (crop) => {
    const updated = toggleBookmark("crops", crop);
    setBookmarkedCropIds(updated.map((item) => item.id));
  };

  const filteredCrops = useMemo(() => {
    return CROPS.filter((crop) => {
      const matchesSeason = selectedSeason === "All" || crop.season === selectedSeason;
      const matchesSearch = crop.name.toLowerCase().includes(searchQuery.toLowerCase());
      return matchesSeason && matchesSearch;
    });
  }, [selectedSeason, searchQuery]);

  return (
    <div className="crop-page">
      <header className="crop-hero">
        <h1><Wheat size={28} aria-hidden="true" /> Crop Guide</h1>
        <p>Explore crops based on season, soil & water needs</p>
      </header>

      <div className="crop-search">
        <input
          type="text"
          placeholder="Search crops..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
        />
      </div>

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

      <div className="crop-grid">
        {filteredCrops.length > 0 ? (
          filteredCrops.map((crop) => (
            <div key={crop.id} className="crop-card">
              
              {/* IMAGE HEADER CONTAINER */}
              <div className="crop-card-image-wrapper">
                <img 
                  src={CROP_IMAGES[crop.name]} 
                  alt={crop.name} 
                  className="crop-card-img"
                  loading="lazy"
                />
              </div>

              <h2>{crop.name}</h2>

              <div className="crop-info">
                <p><strong>Season:</strong> {crop.season}</p>
                <p><strong>Soil:</strong> {crop.soil}</p>
                <p><strong>Water:</strong> {crop.water}</p>
              </div>

              <div className="crop-card-actions">
                <button onClick={() => setActiveCrop(crop)}>View Details</button>
                <button
                  className={`bookmark-btn ${bookmarkedCropIds.includes(crop.id) ? "active" : ""}`}
                  onClick={(e) => {
                    e.stopPropagation();
                    handleToggleCropBookmark(crop);
                  }}
                >
                  {bookmarkedCropIds.includes(crop.id) ? "Saved" : "Bookmark"}
                </button>
              </div>
            </div>
          ))
        ) : (
          <p className="no-results">No crops found <Wheat size={16} aria-hidden="true" /></p>
        )}
      </div>

      {activeCrop && (
        <div className="crop-modal" onClick={() => setActiveCrop(null)}>
          <div className="crop-popup" onClick={(e) => e.stopPropagation()}>
            <button className="close-btn" onClick={() => setActiveCrop(null)} aria-label="Close crop details"><X size={16} /></button>

            {/* IMAGE HEADER IN MODAL */}
            <div className="modal-crop-image-wrapper">
              <img 
                src={CROP_IMAGES[activeCrop.name]} 
                alt={activeCrop.name} 
                className="modal-crop-img"
                loading="lazy"
              />
            </div>

            <div className="modal-header-row">
              <h2><Wheat size={20} aria-hidden="true" /> {activeCrop.name}</h2>
              <button
                className={`bookmark-btn modal-bookmark ${bookmarkedCropIds.includes(activeCrop.id) ? "active" : ""}`}
                onClick={() => handleToggleCropBookmark(activeCrop)}
              >
                {bookmarkedCropIds.includes(activeCrop.id) ? "Saved" : "Bookmark"}
              </button>
            </div>

            <div className="modal-info">
              <p><strong>Season:</strong> {activeCrop.season}</p>
              <p><strong>Soil:</strong> {activeCrop.soil}</p>
              <p><strong>Water:</strong> {activeCrop.water}</p>
              <p><strong>Duration:</strong> {activeCrop.duration}</p>
              <p><strong>Yield:</strong> {activeCrop.yield}</p>
            </div>

            <div className="tips"><Lightbulb size={16} aria-hidden="true" /> {activeCrop.tips}</div>
          </div>
        </div>
      )}
    </div>
  );
}