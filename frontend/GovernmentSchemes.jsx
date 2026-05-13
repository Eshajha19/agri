import React, { useEffect, useMemo, useState } from "react";
import {
  FaSearch,
  FaExternalLinkAlt,
  FaInfoCircle,
  FaUserCheck,
  FaCheckCircle,
  FaFilter,
  FaArrowRight,
} from "react-icons/fa";
import {
  MdOutlineWorkspacePremium,
  MdOutlineWaterDrop,
} from "react-icons/md";
import "./GovernmentSchemes.css";

const SCHEMES_DATA = [
  {
    id: 1,
    title: "PM-KISAN",
    fullName: "Pradhan Mantri Kisan Samman Nidhi",
    category: "Financial Support",
    icon: "💰",
    benefits:
      "Fixed income support of ₹6,000 per year in three equal installments to all landholding farmer families.",
    eligibility:
      "All landholding farmer families across the country (with some exclusions for high-income groups).",
    link: "https://pmkisan.gov.in/",
    featured: true,
  },
  {
    id: 2,
    title: "PMFBY",
    fullName: "Pradhan Mantri Fasal Bima Yojana",
    category: "Insurance",
    icon: "🛡️",
    benefits:
      "Comprehensive insurance cover against crop failure due to non-preventable natural risks.",
    eligibility:
      "All farmers growing notified crops in notified areas, including tenant farmers.",
    link: "https://pmfby.gov.in/",
    featured: true,
  },
  {
    id: 3,
    title: "KCC",
    fullName: "Kisan Credit Card",
    category: "Credit",
    icon: "💳",
    benefits:
      "Timely credit for agriculture and allied activities with low interest rates.",
    eligibility:
      "Owner cultivators, tenant farmers, oral lessees, sharecroppers, and SHGs.",
    link: "https://www.myscheme.gov.in/schemes/kcc",
  },
  {
    id: 4,
    title: "PM-KMY",
    fullName: "Pradhan Mantri Kisan Maan Dhan Yojana",
    category: "Pension",
    icon: "👴",
    benefits:
      "Minimum fixed pension of ₹3,000 per month upon reaching 60 years of age.",
    eligibility:
      "Small and Marginal Farmers aged between 18 to 40 years.",
    link: "https://maandhan.in/",
  },
  {
    id: 5,
    title: "Soil Health Card",
    fullName: "Soil Health Card Scheme",
    category: "Resources",
    icon: "🧪",
    benefits:
      "Detailed report on soil nutrient status and fertilizer recommendations.",
    eligibility:
      "All farmers in India can get their soil samples tested every 2 years.",
    link: "https://www.soilhealth.dac.gov.in/",
  },
  {
    id: 6,
    title: "PMKSY",
    fullName: "Pradhan Mantri Krishi Sinchai Yojana",
    category: "Irrigation",
    icon: "💧",
    benefits:
      "Subsidies for micro-irrigation systems to improve water efficiency.",
    eligibility:
      "Farmers, SHGs, and Trusts focused on agriculture.",
    link: "https://pmksy.gov.in/",
    featured: true,
  },
  {
    id: 7,
    title: "PKVY",
    fullName: "Paramparagat Krishi Vikas Yojana",
    category: "Organic Farming",
    icon: "🌱",
    benefits:
      "Financial assistance for organic cultivation and certification.",
    eligibility:
      "Groups of 20 or more farmers forming clusters.",
    link:
      "https://dmsouthwest.delhi.gov.in/scheme/paramparagat-krishi-vikas-yojana/",
  },
  {
    id: 8,
    title: "NHM",
    fullName: "National Horticulture Mission",
    category: "Horticulture",
    icon: "🍎",
    benefits:
      "Support for cold storage, greenhouses, and high-value crops.",
    eligibility:
      "Farmers interested in fruits, vegetables, flowers, and spices.",
    link: "https://www.myscheme.gov.in/schemes/midh",
  },
  {
    id: 9,
    title: "Rythu Bandhu",
    fullName: "Farmers' Investment Support Scheme",
    category: "State Specific",
    icon: "🌾",
    benefits:
      "Investment support for seeds, fertilizers, and pesticides.",
    eligibility:
      "Farmers in Telangana owning agricultural land.",
    link: "https://rythubharosa.telangana.gov.in",
  },
  {
    id: 10,
    title: "KALIA",
    fullName:
      "Krushak Assistance for Livelihood and Income Augmentation",
    category: "State Specific",
    icon: "🚜",
    benefits:
      "Financial assistance for small and marginal farmers.",
    eligibility:
      "Small and marginal farmers and landless agricultural households.",
    link: "https://www.myscheme.gov.in/schemes/kalia",
  },
];

const CATEGORIES = [
  "All",
  "Financial Support",
  "Insurance",
  "Credit",
  "Pension",
  "Resources",
  "Irrigation",
  "Organic Farming",
  "Horticulture",
  "State Specific",
];

export default function Schemes() {
  const [searchTerm, setSearchTerm] = useState("");
  const [activeCategory, setActiveCategory] = useState("All");

  useEffect(() => {
    window.scrollTo(0, 0);
  }, []);

  const filteredSchemes = useMemo(() => {
    return SCHEMES_DATA.filter((scheme) => {
      const matchesSearch =
        scheme.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
        scheme.fullName
          .toLowerCase()
          .includes(searchTerm.toLowerCase()) ||
        scheme.category
          .toLowerCase()
          .includes(searchTerm.toLowerCase());

      const matchesCategory =
        activeCategory === "All" ||
        scheme.category === activeCategory;

      return matchesSearch && matchesCategory;
    });
  }, [searchTerm, activeCategory]);

  const featuredSchemes = SCHEMES_DATA.filter(
    (scheme) => scheme.featured
  );

  return (
    <div className="schemes-page">
      {/* HERO */}
      <section className="schemes-hero">
        <div className="hero-badge">
          <MdOutlineWorkspacePremium />
          Trusted Government Schemes
        </div>

        <h1>
          Empowering Farmers Through <span>Smart Support</span>
        </h1>

        <p>
          Discover agricultural schemes for insurance, irrigation,
          pension, loans, subsidies, and income support designed
          for Indian farmers.
        </p>

        <div className="hero-stats">
          <div className="stat-card">
            <h3>{SCHEMES_DATA.length}+</h3>
            <p>Government Schemes</p>
          </div>

          <div className="stat-card">
            <h3>100%</h3>
            <p>Official Resources</p>
          </div>

          <div className="stat-card">
            <h3>Easy</h3>
            <p>Application Access</p>
          </div>
        </div>
      </section>

      {/* FEATURED */}
      <section className="featured-section">
        <div className="section-heading">
          <h2>Featured Schemes</h2>
          <p>Popular schemes farmers apply for most frequently.</p>
        </div>

        <div className="featured-grid">
          {featuredSchemes.map((scheme) => (
            <div className="featured-card" key={scheme.id}>
              <div className="featured-top">
                <span className="featured-icon">
                  {scheme.icon}
                </span>

                <span className="featured-tag">
                  <FaCheckCircle />
                  Recommended
                </span>
              </div>

              <h3>{scheme.title}</h3>
              <p>{scheme.fullName}</p>

              <a
                href={scheme.link}
                target="_blank"
                rel="noopener noreferrer"
                className="featured-btn"
              >
                Explore Scheme
                <FaArrowRight />
              </a>
            </div>
          ))}
        </div>
      </section>

      {/* CONTROLS */}
      <section className="schemes-controls">
        <div className="schemes-search">
          <FaSearch className="search-icon" />

          <input
            type="text"
            placeholder="Search schemes, benefits, categories..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>

        <div className="filter-title">
          <FaFilter />
          Filter Categories
        </div>

        <div className="schemes-filter-chips">
          {CATEGORIES.map((cat) => (
            <button
              key={cat}
              className={`filter-chip ${
                activeCategory === cat ? "active" : ""
              }`}
              onClick={() => setActiveCategory(cat)}
            >
              {cat}
            </button>
          ))}
        </div>
      </section>

      {/* GRID */}
      <section className="schemes-grid">
        {filteredSchemes.length > 0 ? (
          filteredSchemes.map((scheme) => (
            <div className="scheme-card" key={scheme.id}>
              <div className="scheme-header">
                <div className="scheme-icon">
                  {scheme.icon}
                </div>

                <span className="scheme-category">
                  {scheme.category}
                </span>
              </div>

              <div className="scheme-content">
                <h2>{scheme.title}</h2>

                <p className="scheme-fullname">
                  {scheme.fullName}
                </p>

                <div className="scheme-info">
                  <div className="info-box">
                    <h4>
                      <FaInfoCircle />
                      Benefits
                    </h4>

                    <p>{scheme.benefits}</p>
                  </div>

                  <div className="info-box">
                    <h4>
                      <FaUserCheck />
                      Eligibility
                    </h4>

                    <p>{scheme.eligibility}</p>
                  </div>
                </div>
              </div>

              <div className="scheme-footer">
                <a
                  href={scheme.link}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="btn-visit"
                >
                  Apply on Official Website
                  <FaExternalLinkAlt size={13} />
                </a>
              </div>
            </div>
          ))
        ) : (
          <div className="no-schemes">
            <MdOutlineWaterDrop size={50} />

            <h3>No schemes found</h3>

            <p>
              Try changing category filters or search keywords.
            </p>
          </div>
        )}
      </section>
    </div>
  );
}