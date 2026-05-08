import React, { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import "./ResourcesPage.css";

const resourcesData = [
  {
    id: 1,
    type: "Farming Tips",
    title: "Smart Seasonal Farming",
    description:
      "Learn how to choose crops based on season, soil health, and weather conditions.",
    tags: ["Seasonal", "Soil", "Irrigation"],
    icon: "🌱",
  },
  {
    id: 2,
    type: "Articles",
    title: "Modern Agriculture Trends",
    description:
      "Explore smart farming technologies, AI in agriculture, and government schemes.",
    tags: ["Tech", "AI", "Govt"],
    icon: "🚜",
  },
  {
    id: 3,
    type: "Guides",
    title: "Complete Farming Guide",
    description:
      "Step-by-step guide for soil testing, fertilizer usage, and crop rotation.",
    tags: ["Beginner", "Advanced", "Yield"],
    icon: "📘",
  },
  {
    id: 4,
    type: "Farming Tips",
    title: "Pest Control Methods",
    description:
      "Natural and chemical methods to protect crops from pests effectively.",
    tags: ["Pest", "Organic", "Protection"],
    icon: "🪲",
  },
  {
    id: 5,
    type: "Guides",
    title: "Crop Disease Awareness",
    description:
      "Comprehensive guide on identifying symptoms, prevention, and remedies for crop diseases.",
    tags: ["Disease", "Symptoms", "Remedy"],
    link: "/disease-awareness",
    icon: "🩺",
  },
];

export default function ResourcesPage() {
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState("All");
  const [visibleCount, setVisibleCount] = useState(6);

  const categories = ["All", "Farming Tips", "Articles", "Guides"];

  const filteredResources = useMemo(() => {
    return resourcesData.filter((item) => {
      const searchText = search.toLowerCase();

      const matchSearch =
        item.title.toLowerCase().includes(searchText) ||
        item.description.toLowerCase().includes(searchText) ||
        item.tags.some((tag) =>
          tag.toLowerCase().includes(searchText)
        );

      const matchFilter =
        filter === "All" || item.type === filter;

      return matchSearch && matchFilter;
    });
  }, [search, filter]);

  const visibleResources = filteredResources.slice(0, visibleCount);

  return (
    <div className="resources-page">
      {/* HERO SECTION */}
      <section className="resources-hero">
        <div className="hero-content">
          <span className="hero-badge">🌾 Smart Agriculture Learning</span>

          <h1>
            Explore the <span>Knowledge Hub</span>
          </h1>

          <p>
            Discover expert farming guides, crop protection methods,
            seasonal tips, and modern agriculture technologies.
          </p>

          <div className="search-wrapper">
            <input
              type="text"
              placeholder="Search farming guides, tags, or topics..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="search-box"
            />
          </div>
        </div>
      </section>

      {/* FILTERS */}
      <section className="filter-section">
        <div className="filter-header">
          <h2>Browse Categories</h2>
          <p>{filteredResources.length} resources available</p>
        </div>

        <div className="filter-bar">
          {categories.map((type) => {
            const count =
              type === "All"
                ? resourcesData.length
                : resourcesData.filter(
                    (r) => r.type === type
                  ).length;

            return (
              <button
                key={type}
                className={filter === type ? "active" : ""}
                onClick={() => setFilter(type)}
              >
                {type}
                <span>{count}</span>
              </button>
            );
          })}
        </div>
      </section>

      {/* RESOURCES GRID */}
      <section className="resources-grid">
        {visibleResources.length > 0 ? (
          visibleResources.map((item) => (
            <div key={item.id} className="resource-card">
              <div className="card-top">
                <div className="card-icon">{item.icon}</div>
                <div className="card-type">{item.type}</div>
              </div>

              <h3>{item.title}</h3>
              <p>{item.description}</p>

              <div className="tags">
                {item.tags.map((tag, i) => (
                  <span key={i}>{tag}</span>
                ))}
              </div>

              {item.link ? (
                <Link to={item.link}>
                  <button className="explore-btn">
                    Explore Resource →
                  </button>
                </Link>
              ) : (
                <button className="explore-btn">
                  Explore Resource →
                </button>
              )}
            </div>
          ))
        ) : (
          <div className="no-results">
            <h3>No resources found 😕</h3>
            <p>Try different keywords or reset filters.</p>

            <button
              onClick={() => {
                setSearch("");
                setFilter("All");
              }}
            >
              Reset Filters
            </button>
          </div>
        )}
      </section>

      {/* LOAD MORE */}
      {visibleCount < filteredResources.length && (
        <div className="load-more">
          <button
            onClick={() =>
              setVisibleCount((prev) => prev + 3)
            }
          >
            Load More Resources
          </button>
        </div>
      )}

      {/* ABOUT SECTION */}
      <section className="about-section">
        <div className="about-header">
          <h2>Why Use Knowledge Hub? 🌱</h2>
          <p>
            Empowering farmers and learners with practical,
            modern, and reliable agriculture knowledge.
          </p>
        </div>

        <div className="about-features">
          <div className="feature-card">
            <div className="feature-icon">🌿</div>
            <h4>Practical Farming Tips</h4>
            <p>
              Easy-to-follow real-world farming advice for
              better crop productivity.
            </p>
          </div>

          <div className="feature-card">
            <div className="feature-icon">📚</div>
            <h4>Detailed Guides</h4>
            <p>
              Learn step-by-step methods from beginner to
              advanced agriculture practices.
            </p>
          </div>

          <div className="feature-card">
            <div className="feature-icon">🚀</div>
            <h4>Modern Agriculture</h4>
            <p>
              Stay updated with smart farming, AI tools,
              sustainability, and innovations.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}
