import React, { useState } from "react";
import "./CropGuide.css";

const cropsData = [
  {
    id: 1,
    name: "Rice",
    season: "Kharif",
    soil: "Clayey / Loamy Soil",
    temperature: "20°C - 35°C",
    water: "High (Requires flooded fields)",
    climate: "Humid & Rainy",
    duration: "120–150 days",
    yield: "3–6 tons/hectare",
    pests: "Stem borer, Brown planthopper",
  },
  {
    id: 2,
    name: "Wheat",
    season: "Rabi",
    soil: "Well-drained Loamy Soil",
    temperature: "10°C - 25°C",
    water: "Moderate",
    climate: "Cool & Dry",
    duration: "100–120 days",
    yield: "2–5 tons/hectare",
    pests: "Aphids, Rust disease",
  },
  {
    id: 3,
    name: "Maize",
    season: "Kharif",
    soil: "Alluvial Soil",
    temperature: "18°C - 30°C",
    water: "Moderate",
    climate: "Warm",
    duration: "90–110 days",
    yield: "2–4 tons/hectare",
    pests: "Armyworm, Stem borer",
  },
  {
    id: 4,
    name: "Sugarcane",
    season: "Year-round",
    soil: "Deep Loamy Soil",
    temperature: "20°C - 35°C",
    water: "High",
    climate: "Tropical",
    duration: "10–18 months",
    yield: "60–100 tons/hectare",
    pests: "Early shoot borer, Pyrilla",
  },
  {
    id: 5,
    name: "Cotton",
    season: "Kharif",
    soil: "Black Soil",
    temperature: "21°C - 30°C",
    water: "Moderate",
    climate: "Warm & Dry",
    duration: "150–180 days",
    yield: "1–2 tons/hectare",
    pests: "Bollworm, Whitefly",
  },
  {
    id: 6,
    name: "Mustard",
    season: "Rabi",
    soil: "Sandy Loam Soil",
    temperature: "10°C - 25°C",
    water: "Low to Moderate",
    climate: "Cool",
    duration: "80–100 days",
    yield: "1–2 tons/hectare",
    pests: "Aphids, White rust",
  },

  {
    id: 7,
    name: "Barley",
    season: "Rabi",
    soil: "Loamy Soil",
    temperature: "12°C - 25°C",
    water: "Low",
    climate: "Dry & Cool",
    duration: "90–110 days",
    yield: "2–4 tons/hectare",
    pests: "Aphids, Leaf rust",
  },
  {
    id: 8,
    name: "Millets (Bajra)",
    season: "Kharif",
    soil: "Sandy Soil",
    temperature: "25°C - 35°C",
    water: "Low",
    climate: "Dry & Arid",
    duration: "75–90 days",
    yield: "1–2 tons/hectare",
    pests: "Shoot fly",
  },
  {
    id: 9,
    name: "Chickpea (Gram)",
    season: "Rabi",
    soil: "Loamy Soil",
    temperature: "20°C - 25°C",
    water: "Low",
    climate: "Dry",
    duration: "90–110 days",
    yield: "1–2 tons/hectare",
    pests: "Pod borer",
  },
  {
    id: 10,
    name: "Lentil",
    season: "Rabi",
    soil: "Alluvial Soil",
    temperature: "18°C - 30°C",
    water: "Low",
    climate: "Cool",
    duration: "100–120 days",
    yield: "1–1.5 tons/hectare",
    pests: "Aphids",
  },
  {
    id: 11,
    name: "Groundnut",
    season: "Kharif",
    soil: "Sandy Loam",
    temperature: "25°C - 35°C",
    water: "Moderate",
    climate: "Warm",
    duration: "100–120 days",
    yield: "2–3 tons/hectare",
    pests: "Leaf miner",
  },
  {
    id: 12,
    name: "Sunflower",
    season: "Kharif/Rabi",
    soil: "Well-drained Soil",
    temperature: "20°C - 30°C",
    water: "Moderate",
    climate: "Sunny",
    duration: "80–100 days",
    yield: "1.5–2 tons/hectare",
    pests: "Cutworm",
  },
  {
    id: 13,
    name: "Tea",
    season: "Year-round",
    soil: "Acidic Soil",
    temperature: "18°C - 30°C",
    water: "High",
    climate: "Humid",
    duration: "Perennial",
    yield: "Varies",
    pests: "Tea mosquito bug",
  },
  {
    id: 14,
    name: "Coffee",
    season: "Year-round",
    soil: "Rich Loamy Soil",
    temperature: "15°C - 28°C",
    water: "Moderate",
    climate: "Tropical",
    duration: "3–4 years",
    yield: "Varies",
    pests: "Berry borer",
  },
  {
    id: 15,
    name: "Potato",
    season: "Rabi",
    soil: "Sandy Loam",
    temperature: "15°C - 20°C",
    water: "Moderate",
    climate: "Cool",
    duration: "70–90 days",
    yield: "20–30 tons/hectare",
    pests: "Late blight",
  },
  {
    id: 16,
    name: "Tomato",
    season: "Year-round",
    soil: "Loamy Soil",
    temperature: "20°C - 30°C",
    water: "Moderate",
    climate: "Warm",
    duration: "60–80 days",
    yield: "25–35 tons/hectare",
    pests: "Whitefly",
  },
  {
    id: 17,
    name: "Onion",
    season: "Rabi",
    soil: "Sandy Loam",
    temperature: "13°C - 24°C",
    water: "Moderate",
    climate: "Mild",
    duration: "90–120 days",
    yield: "20–25 tons/hectare",
    pests: "Thrips",
  },
  {
    id: 18,
    name: "Mango",
    season: "Summer",
    soil: "Alluvial Soil",
    temperature: "24°C - 30°C",
    water: "Moderate",
    climate: "Tropical",
    duration: "3–5 years",
    yield: "Varies",
    pests: "Fruit fly",
  },
  {
    id: 19,
    name: "Banana",
    season: "Year-round",
    soil: "Rich Loamy Soil",
    temperature: "20°C - 35°C",
    water: "High",
    climate: "Humid",
    duration: "9–12 months",
    yield: "30–50 tons/hectare",
    pests: "Panama disease",
  }
];

export default function CropGuide() {
  const [filter, setFilter] = useState("All");
  const [activeId, setActiveId] = useState(null); // ✅ added

  const filteredCrops = cropsData.filter((crop) => {
    if (filter === "All") return true;
    return crop.season.includes(filter); // ✅ fixed
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

            <button onClick={() => setActiveId(activeId === crop.id ? null : crop.id)}>
              {activeId === crop.id ? "Hide Details" : "View Details"}
            </button>

            {activeId === crop.id && (
              <div className="crop-details">
                <p> Temperature: {crop.temperature}</p>
                <p> Water: {crop.water}</p>
                <p> Climate: {crop.climate}</p>
                <p> Duration: {crop.duration}</p>
                <p> Yield: {crop.yield}</p>
                <p> Pests: {crop.pests}</p>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}