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
  FaTractor,
  FaThermometerHalf,
  FaWarehouse,
  FaChartLine,
  FaSitemap,
  FaTimes
} from "react-icons/fa";
import "./Glossary.css";

const glossaryTerms = [
  {
    term: "Drip Irrigation",
    definition:
      "A precision irrigation method that delivers water and dissolved nutrients directly to the root zone through emitters, minimizing evaporation, runoff, and weed growth while improving water-use efficiency.",
    category: "Irrigation",
    icon: <FaWater />
  },
  {
    term: "Mulching",
    definition:
      "The practice of covering soil with organic or synthetic material to conserve moisture, suppress weeds, reduce erosion, moderate soil temperature, and improve the long-term condition of the topsoil.",
    category: "Soil Management",
    icon: <FaLeaf />
  },
  {
    term: "Composting",
    definition:
      "The controlled biological decomposition of crop residues, manure, and other organic materials into stable humus-like compost that improves nutrient availability, soil structure, and microbial activity.",
    category: "Fertilization",
    icon: <FaSeedling />
  },
  {
    term: "Soil Moisture",
    definition:
      "The water held in the soil profile and available for plant uptake, which strongly influences germination, nutrient transport, root development, and irrigation scheduling.",
    category: "Soil Management",
    icon: <FaVial />
  },
  {
    term: "Crop Rotation",
    definition:
      "The planned sequence of different crops on the same field across seasons or years to balance nutrient demand, interrupt pest and disease cycles, and support healthier soils.",
    category: "Planning",
    icon: <FaHistory />
  },
  {
    term: "Organic Farming",
    definition:
      "An agricultural production system that relies on biological inputs, crop diversity, compost, and ecological processes instead of synthetic fertilizers and pesticides to maintain soil fertility and sustainability.",
    category: "Sustainability",
    icon: <FaGlobeAmericas />
  },
  {
    term: "Kharif Crops",
    definition:
      "Crops sown with the arrival of the monsoon and harvested after the rainy season, such as rice, maize, cotton, soybean, and pigeon pea in many South Asian systems.",
    category: "Seasons",
    icon: <FaSeedling />
  },
  {
    term: "Rabi Crops",
    definition:
      "Crops planted after the monsoon in cooler months and harvested in spring, including wheat, mustard, barley, and several pulses that benefit from mild winter conditions.",
    category: "Seasons",
    icon: <FaSeedling />
  },
  {
    term: "Integrated Pest Management (IPM)",
    definition:
      "A decision-based pest control strategy that combines scouting, thresholds, resistant varieties, biological controls, cultural practices, and targeted pesticide use to manage pests with less environmental impact.",
    category: "Pest Control",
    icon: <FaBug />
  },
  {
    term: "Soil pH",
    definition:
      "A scale that measures soil acidity or alkalinity and influences nutrient solubility, microbial activity, fertilizer performance, and whether a crop can efficiently absorb key minerals.",
    category: "Soil Management",
    icon: <FaVial />
  },
  {
    term: "Precision Agriculture",
    definition:
      "A farming approach that uses GPS, sensors, satellite imagery, and data analytics to apply water, seed, fertilizer, and crop protection inputs more accurately across variable field conditions.",
    category: "Technology",
    icon: <FaSitemap />
  },
  {
    term: "Remote Sensing",
    definition:
      "The collection of crop and field information from satellites, drones, or aircraft without direct contact, often used to monitor plant health, moisture stress, and spatial variability at scale.",
    category: "Technology",
    icon: <FaGlobeAmericas />
  },
  {
    term: "Evapotranspiration",
    definition:
      "The combined loss of water from soil evaporation and plant transpiration, commonly used to estimate crop water demand and guide irrigation timing and volume.",
    category: "Climate & Water",
    icon: <FaThermometerHalf />
  },
  {
    term: "Fertigation",
    definition:
      "The application of soluble fertilizers through an irrigation system so nutrients can be delivered more evenly and at the right growth stage with reduced labor and nutrient loss.",
    category: "Irrigation",
    icon: <FaWater />
  },
  {
    term: "Cover Crop",
    definition:
      "A crop grown primarily to protect and improve the soil rather than for harvest, helping reduce erosion, suppress weeds, increase organic matter, and capture leftover nutrients.",
    category: "Soil Management",
    icon: <FaLeaf />
  },
  {
    term: "Green Manure",
    definition:
      "A fast-growing crop that is incorporated into the soil while still green to add organic matter, improve structure, and release nutrients as it decomposes.",
    category: "Fertilization",
    icon: <FaSeedling />
  },
  {
    term: "Soil Organic Matter",
    definition:
      "The decomposed and partially decomposed plant and animal material in soil that improves water retention, nutrient cycling, structure, and biological activity.",
    category: "Soil Management",
    icon: <FaLeaf />
  },
  {
    term: "Nutrient Management",
    definition:
      "The planning and application of fertilizers, manures, and soil amendments according to crop demand, soil test results, and environmental conditions to avoid deficiency or over-application.",
    category: "Fertilization",
    icon: <FaChartLine />
  },
  {
    term: "Soil Testing",
    definition:
      "The laboratory analysis of soil samples to determine nutrient levels, pH, salinity, and other properties so farmers can make informed fertilizer and amendment decisions.",
    category: "Planning",
    icon: <FaVial />
  },
  {
    term: "Salinity",
    definition:
      "The concentration of soluble salts in soil or irrigation water, which can reduce germination, restrict water uptake, and lower yields when levels become excessive.",
    category: "Climate & Water",
    icon: <FaWater />
  },
  {
    term: "Raised Bed Planting",
    definition:
      "A cultivation method that grows crops on elevated soil beds separated by furrows or paths, improving drainage, root aeration, and often irrigation efficiency.",
    category: "Planning",
    icon: <FaTractor />
  },
  {
    term: "Cold Storage",
    definition:
      "Temperature-controlled storage used after harvest to slow respiration, reduce spoilage, and extend shelf life for fruits, vegetables, seeds, and other perishables.",
    category: "Post-Harvest",
    icon: <FaWarehouse />
  },
  {
    term: "Post-Harvest Loss",
    definition:
      "The loss of quantity, quality, or value that occurs between harvest and final consumption because of handling, storage, transport, pests, or spoilage.",
    category: "Post-Harvest",
    icon: <FaWarehouse />
  },
  {
    term: "Market Linkage",
    definition:
      "The connection between producers and buyers through aggregators, cooperatives, digital platforms, or contracts that improves price discovery and market access.",
    category: "Farm Economics",
    icon: <FaChartLine />
  },
  {
    term: "Yield Forecasting",
    definition:
      "The use of crop observations, weather data, and statistical or machine-learning models to estimate expected harvest output before the end of the season.",
    category: "Technology",
    icon: <FaChartLine />
  },
  {
    term: "Intercropping",
    definition:
      "The practice of growing two or more crops together in the same field at the same time to improve land use, reduce pest pressure, and stabilize farm output.",
    category: "Planning",
    icon: <FaSeedling />
  },
  {
    term: "Crop Canopy",
    definition:
      "The collective layer of leaves and stems above the ground, which influences sunlight interception, transpiration, temperature moderation, and photosynthetic productivity.",
    category: "Crop Science",
    icon: <FaLeaf />
  },
  {
    term: "Germination",
    definition:
      "The stage when a seed begins to sprout and produce a root and shoot under suitable moisture, temperature, and oxygen conditions.",
    category: "Crop Science",
    icon: <FaSeedling />
  },
  {
    term: "Biological Control",
    definition:
      "The use of natural enemies such as predators, parasitoids, or beneficial microbes to suppress pests, diseases, or weeds.",
    category: "Pest Control",
    icon: <FaBug />
  },
  {
    term: "Weather Advisory",
    definition:
      "A localized forecast-based recommendation that helps farmers time irrigation, spraying, harvesting, and field operations around expected rain, wind, heat, or frost.",
    category: "Climate & Water",
    icon: <FaThermometerHalf />
  },
  {
    term: "Carbon Sequestration",
    definition:
      "The capture and storage of atmospheric carbon in vegetation and soil, often promoted through conservation agriculture, cover crops, reduced tillage, and agroforestry.",
    category: "Sustainability",
    icon: <FaGlobeAmericas />
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