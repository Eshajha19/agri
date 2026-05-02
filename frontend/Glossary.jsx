import React, { useState } from "react";
import { 
  FaSearch, 
  FaBook, 
  FaLeaf, 
  FaWater, 
  FaSeedling, 
  FaVial, 
  FaHistory, 
  FaBug,
  FaGlobeAmericas
} from "react-icons/fa";
import "./Glossary.css";

const glossaryTerms = [
  {
    term: "Drip Irrigation",
    definition: "A precise watering method that delivers water and nutrients directly to the plant's root zone through a network of pipes and emitters, significantly reducing water wastage.",
    category: "Irrigation",
    icon: <FaWater />
  },
  {
    term: "Mulching",
    definition: "The process of covering the top layer of soil with organic (straw, bark) or inorganic materials to retain soil moisture, suppress weed growth, and regulate soil temperature.",
    category: "Soil Management",
    icon: <FaLeaf />
  },
  {
    term: "Composting",
    definition: "The natural process of recycling organic matter, such as leaves and food scraps, into a rich soil amendment called compost, which improves soil health and fertility.",
    category: "Fertilization",
    icon: <FaSeedling />
  },
  {
    term: "Soil Moisture",
    definition: "The amount of water contained in the soil, which is crucial for seed germination, nutrient uptake, and overall plant growth and health.",
    category: "Soil Management",
    icon: <FaVial />
  },
  {
    term: "Crop Rotation",
    definition: "The practice of growing different types of crops in the same area in sequential seasons to improve soil health, optimize nutrients, and combat pest and weed pressure.",
    category: "Planning",
    icon: <FaHistory />
  },
  {
    term: "Organic Farming",
    definition: "An agricultural system that uses fertilizers of organic origin (compost, green manure) and places emphasis on techniques such as crop rotation and companion planting.",
    category: "Sustainability",
    icon: <FaGlobeAmericas />
  },
  {
    term: "Kharif Crops",
    definition: "Crops that are sown during the monsoon season (usually June-July) and harvested in autumn (September-October). Examples include Rice, Maize, and Cotton.",
    category: "Seasons",
    icon: <FaSeedling />
  },
  {
    term: "Rabi Crops",
    definition: "Crops that are sown in the winter season (usually October-November) and harvested in the spring (March-April). Examples include Wheat, Mustard, and Gram.",
    category: "Seasons",
    icon: <FaSeedling />
  },
  {
    term: "Integrated Pest Management (IPM)",
    definition: "An ecosystem-based strategy that focuses on long-term prevention of pests through a combination of biological control, habitat manipulation, and resistant varieties.",
    category: "Pest Control",
    icon: <FaBug />
  },
  {
    term: "Soil pH",
    definition: "A measure of the acidity or alkalinity of the soil. A pH of 7.0 is neutral; below 7.0 is acidic and above 7.0 is alkaline, affecting nutrient availability.",
    category: "Soil Management",
    icon: <FaVial />
  }
];

const Glossary = () => {
  const [searchTerm, setSearchTerm] = useState("");
  const [activeCategory, setActiveCategory] = useState("All");

  const categories = ["All", ...new Set(glossaryTerms.map(t => t.category))];

  const filteredTerms = glossaryTerms.filter(t => {
    const matchesSearch = t.term.toLowerCase().includes(searchTerm.toLowerCase()) || 
                         t.definition.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesCategory = activeCategory === "All" || t.category === activeCategory;
    return matchesSearch && matchesCategory;
  });

  return (
    <div className="glossary-container">
      <header className="glossary-header">
        <div className="header-badge">LEARNING CENTER</div>
        <h1>Agricultural Glossary</h1>
        <p>Master the language of modern farming with our simplified guide to common agricultural terms.</p>
        
        <div className="glossary-search-bar">
          <FaSearch className="search-icon" />
          <input 
            type="text" 
            placeholder="Search for a term (e.g., Mulching, Irrigation)..." 
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>

        <div className="category-filters">
          {categories.map(cat => (
            <button 
              key={cat}
              className={`filter-btn ${activeCategory === cat ? 'active' : ''}`}
              onClick={() => setActiveCategory(cat)}
            >
              {cat}
            </button>
          ))}
        </div>
      </header>

      <div className="glossary-grid">
        {filteredTerms.length > 0 ? (
          filteredTerms.map((item, index) => (
            <div key={index} className="glossary-card">
              <div className="card-icon">{item.icon}</div>
              <div className="card-content">
                <span className="term-category">{item.category}</span>
                <h3>{item.term}</h3>
                <p>{item.definition}</p>
              </div>
            </div>
          ))
        ) : (
          <div className="no-results">
            <FaBook />
            <p>No terms found matching your search.</p>
          </div>
        )}
      </div>

      <footer className="glossary-footer">
        <div className="learning-tip">
          <FaLeaf />
          <p><strong>Tip:</strong> Understanding these terms helps you communicate better with agricultural experts and utilize modern farming techniques more effectively.</p>
        </div>
      </footer>
    </div>
  );
};

export default Glossary;
