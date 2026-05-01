import React, { useState } from "react";
import { FaLandmark, FaSearch, FaExternalLinkAlt, FaInfoCircle, FaCheckCircle, FaUserCheck } from "react-icons/fa";
import "./GovernmentSchemes.css";

const SCHEMES_DATA = [
  {
    id: 1,
    title: "PM-KISAN",
    fullName: "Pradhan Mantri Kisan Samman Nidhi",
    category: "Financial Support",
    icon: "💰",
    benefits: "Fixed income support of ₹6,000 per year in three equal installments to all landholding farmer families.",
    eligibility: "All landholding farmer families across the country (with some exclusions for high-income groups).",
    link: "https://pmkisan.gov.in/",
  },
  {
    id: 2,
    title: "PMFBY",
    fullName: "Pradhan Mantri Fasal Bima Yojana",
    category: "Insurance",
    icon: "🛡️",
    benefits: "Comprehensive insurance cover against crop failure due to non-preventable natural risks.",
    eligibility: "All farmers growing notified crops in notified areas, including tenant farmers.",
    link: "https://pmfby.gov.in/",
  },
  {
    id: 3,
    title: "KCC",
    fullName: "Kisan Credit Card",
    category: "Credit",
    icon: "💳",
    benefits: "Timely credit for agriculture and allied activities with low interest rates (as low as 4% with prompt repayment).",
    eligibility: "Owner cultivators, tenant farmers, oral lessees, sharecroppers, and SHGs.",
    link: "https://www.myscheme.gov.in/schemes/kcc",
  },
  {
    id: 4,
    title: "PM-KMY",
    fullName: "Pradhan Mantri Kisan Maan Dhan Yojana",
    category: "Pension",
    icon: "👴",
    benefits: "Minimum fixed pension of ₹3,000 per month upon reaching 60 years of age.",
    eligibility: "Small and Marginal Farmers (SMFs) aged between 18 to 40 years.",
    link: "https://maandhan.in/",
  },
  {
    id: 5,
    title: "Soil Health Card",
    fullName: "Soil Health Card Scheme",
    category: "Resources",
    icon: "🧪",
    benefits: "Detailed report on soil nutrient status and recommendations for fertilizers to improve yield.",
    eligibility: "All farmers in India can get their soil samples tested every 2 years.",
    link: "https://www.soilhealth.dac.gov.in/",
  },
  {
    id: 6,
    title: "PMKSY",
    fullName: "Pradhan Mantri Krishi Sinchai Yojana",
    category: "Irrigation",
    icon: "💧",
    benefits: "Subsidies for micro-irrigation (Drip/Sprinkler) to ensure 'Per Drop More Crop'.",
    eligibility: "Farmers, Self Help Groups, and Trusts focused on agriculture.",
    link: "https://pmksy.gov.in/",
  },
  {
    id: 7,
    title: "PKVY",
    fullName: "Paramparagat Krishi Vikas Yojana",
    category: "Organic Farming",
    icon: "🌱",
    benefits: "Financial assistance for organic cultivation, certification, and marketing.",
    eligibility: "Groups of 20 or more farmers forming clusters for organic farming.",
    link: "https://dmsouthwest.delhi.gov.in/scheme/paramparagat-krishi-vikas-yojana/",
  },
  {
    id: 8,
    title: "NHM",
    fullName: "National Horticulture Mission",
    category: "Horticulture",
    icon: "🍎",
    benefits: "Support for cold storage, greenhouses, and planting materials for high-value crops.",
    eligibility: "Farmers interested in growing fruits, vegetables, flowers, and spices.",
    link: "https://www.myscheme.gov.in/schemes/midh",
  },
  {
    id: 9,
    title: "Rythu Bandhu",
    fullName: "Farmers' Investment Support Scheme (FISS)",
    category: "State Specific",
    icon: "🌾",
    benefits: "Investment support of ₹5,000 per acre per season for purchase of inputs like Seeds, Fertilizers, Pesticides.",
    eligibility: "Farmers in Telangana owning agricultural land.",
    link: "https://rythubharosa.telangana.gov.in",
  },
  {
    id: 10,
    title: "KALIA",
    fullName: "Krushak Assistance for Livelihood and Income Augmentation",
    category: "State Specific",
    icon: "🚜",
    benefits: "Financial assistance of ₹25,000 per farm family over five seasons for small and marginal farmers.",
    eligibility: "Small and marginal farmers, landless agricultural households in Odisha.",
    link: "https://www.myscheme.gov.in/schemes/kalia",
  }
];

const CATEGORIES = ["All", "Financial Support", "Insurance", "Credit", "Pension", "Resources", "Irrigation", "Organic Farming", "Horticulture", "State Specific"];

export default function Schemes() {
  const [searchTerm, setSearchTerm] = useState("");
  const [activeCategory, setActiveCategory] = useState("All");

  // Scroll to top on mount
  React.useEffect(() => {
    window.scrollTo(0, 0);
  }, []);

  const filteredSchemes = SCHEMES_DATA.filter((scheme) => {
    const matchesSearch =
      scheme.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
      scheme.fullName.toLowerCase().includes(searchTerm.toLowerCase());

    const matchesCategory = activeCategory === "All" || scheme.category === activeCategory;

    return matchesSearch && matchesCategory;
  });

  return (
    <div className="schemes-page">
      <div className="schemes-hero">
        <h1>🇮🇳 <span className="notranslate">Government Schemes</span></h1>
        <p>Empowering farmers with direct benefits, insurance, and financial assistance. Find the right scheme for your growth.</p>
      </div>

      <div className="schemes-controls">
        <div className="schemes-search">
          <input
            type="text"
            placeholder="Search schemes (e.g. PM-KISAN, Insurance)..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>

        <div className="schemes-filter-chips">
          {CATEGORIES.map((cat) => (
            <button
              key={cat}
              className={`filter-chip ${activeCategory === cat ? "active" : ""}`}
              onClick={() => setActiveCategory(cat)}
            >
              {cat}
            </button>
          ))}
        </div>
      </div>

      <div className="schemes-grid">
        {filteredSchemes.length > 0 ? (
          filteredSchemes.map((scheme) => (
            <div key={scheme.id} className="scheme-card">
              <div className="scheme-header">
                <div className="scheme-icon">{scheme.icon}</div>
                <div className="scheme-category">{scheme.category}</div>
              </div>

              <h2 className="scheme-title">{scheme.title}</h2>
              <p style={{ marginTop: '-15px', fontSize: '0.9rem', color: '#64748b', fontWeight: '500' }}>
                {scheme.fullName}
              </p>

              <div className="scheme-info">
                <div className="info-box">
                  <h4><FaInfoCircle /> Benefits</h4>
                  <p className="info-content">{scheme.benefits}</p>
                </div>
                <div className="info-box">
                  <h4><FaUserCheck /> Eligibility</h4>
                  <p className="info-content">{scheme.eligibility}</p>
                </div>
              </div>

              <div className="scheme-footer">
                <a href={scheme.link} target="_blank" rel="noopener noreferrer" className="btn-visit">
                  Apply on Official Website <FaExternalLinkAlt size={14} />
                </a>
              </div>
            </div>
          ))
        ) : (
          <div className="no-schemes">
            <h3>No schemes found matching your criteria.</h3>
            <p>Try searching for different keywords or categories.</p>
          </div>
        )}
      </div>
    </div>
  );
}
