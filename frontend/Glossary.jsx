import React, { useMemo, useState } from "react";
import {
  FaSearch,
  FaBook,
  FaLeaf,
  FaWater,
  FaSeedling,
  FaVial,
  FaHistory,
  FaBug,
  FaGlobeAmericas,
  FaTimes
} from "react-icons/fa";
import "./Glossary.css";

const glossaryTerms = [
  {
    term: "Drip Irrigation",
    definition:
      "A precise watering method that delivers water and nutrients directly to the plant's root zone through a network of pipes and emitters, significantly reducing water wastage.",
    category: "Irrigation",
    icon: <FaWater />
  },
  {
    term: "Mulching",
    definition:
      "The process of covering the top layer of soil with organic or inorganic materials to retain soil moisture, suppress weed growth, and regulate soil temperature.",
    category: "Soil Management",
    icon: <FaLeaf />
  },
  {
    term: "Composting",
    definition:
      "The natural process of recycling organic matter into nutrient-rich compost that improves soil fertility and plant health.",
    category: "Fertilization",
    icon: <FaSeedling />
  },
  {
    term: "Soil Moisture",
    definition:
      "The amount of water contained in the soil, crucial for nutrient absorption and healthy crop growth.",
    category: "Soil Management",
    icon: <FaVial />
  },
  {
    term: "Crop Rotation",
    definition:
      "The practice of growing different crops sequentially to maintain soil nutrients and reduce pests.",
    category: "Planning",
    icon: <FaHistory />
  },
  {
    term: "Organic Farming",
    definition:
      "An agricultural system focused on natural fertilizers and sustainable cultivation techniques.",
    category: "Sustainability",
    icon: <FaGlobeAmericas />
  },
  {
    term: "Kharif Crops",
    definition:
      "Crops grown during the monsoon season such as Rice, Maize, and Cotton.",
    category: "Seasons",
    icon: <FaSeedling />
  },
  {
    term: "Rabi Crops",
    definition:
      "Crops sown in winter and harvested in spring such as Wheat and Mustard.",
    category: "Seasons",
    icon: <FaSeedling />
  },
  {
    term: "Integrated Pest Management (IPM)",
    definition:
      "A sustainable pest-control strategy using biological, cultural, and chemical methods together.",
    category: "Pest Control",
    icon: <FaBug />
  },
  {
    term: "Soil pH",
    definition:
      "A measurement of soil acidity or alkalinity affecting nutrient availability.",
    category: "Soil Management",
    icon: <FaVial />
  }
];

const Glossary = () => {
  const [searchTerm, setSearchTerm] = useState("");
  const [activeCategory, setActiveCategory] = useState("All");

  const categories = useMemo(
    () => ["All", ...new Set(glossaryTerms.map((item) => item.category))],
    []
  );

  const filteredTerms = useMemo(() => {
    return glossaryTerms.filter((item) => {
      const query = searchTerm.toLowerCase();

      const matchesSearch =
        item.term.toLowerCase().includes(query) ||
        item.definition.toLowerCase().includes(query) ||
        item.category.toLowerCase().includes(query);

      const matchesCategory =
        activeCategory === "All" || item.category === activeCategory;

      return matchesSearch && matchesCategory;
    });
  }, [searchTerm, activeCategory]);

  const clearSearch = () => {
    setSearchTerm("");
  };

  return (
    <div className="glossary-container">

      {/* HEADER */}
      <header className="glossary-header">
        <div className="header-badge">LEARNING CENTER</div>

        <h1>Agricultural Glossary</h1>

        <p>
          Explore important agricultural terms and improve your understanding
          of modern farming concepts.
        </p>

        {/* SEARCH BAR */}
        <div className="glossary-search-bar">
          <FaSearch className="search-icon" />

          <input
            type="text"
            placeholder="Search terms like Mulching, Irrigation..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />

          {searchTerm && (
            <button className="clear-btn" onClick={clearSearch}>
              <FaTimes />
            </button>
          )}
        </div>

        {/* FILTERS */}
        <div className="category-filters">
          {categories.map((category) => (
            <button
              key={category}
              className={`filter-btn ${
                activeCategory === category ? "active" : ""
              }`}
              onClick={() => setActiveCategory(category)}
            >
              {category}
            </button>
          ))}
        </div>
      </header>

      {/* RESULT COUNT */}
      <div className="results-info">
        <span>
          {filteredTerms.length} term
          {filteredTerms.length !== 1 ? "s" : ""} found
        </span>
      </div>

      {/* GLOSSARY GRID */}
      <div className="glossary-grid">
        {filteredTerms.length > 0 ? (
          filteredTerms.map((item, index) => (
            <div className="glossary-card" key={index}>
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

            <h3>No Results Found</h3>

            <p>
              Try searching with another keyword or choose a different category.
            </p>

            <button
              className="reset-btn"
              onClick={() => {
                setSearchTerm("");
                setActiveCategory("All");
              }}
            >
              Reset Filters
            </button>
          </div>
        )}
      </div>

      {/* FOOTER */}
      <footer className="glossary-footer">
        <div className="learning-tip">
          <FaLeaf />

          <p>
            <strong>Tip:</strong> Learning agricultural terminology helps
            farmers, students, and researchers communicate more effectively and
            adopt smarter farming practices.
          </p>
        </div>
      </footer>
    </div>
  );
};

export default Glossary;