import React, { useState } from "react";
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
  },
  {
    id: 2,
    type: "Articles",
    title: "Modern Agriculture Trends",
    description:
      "Explore smart farming technologies, AI in agriculture, and government schemes.",
    tags: ["Tech", "AI", "Govt"],
  },
  {
    id: 3,
    type: "Guides",
    title: "Complete Farming Guide",
    description:
      "Step-by-step guide for soil testing, fertilizer usage, and crop rotation.",
    tags: ["Beginner", "Advanced", "Yield"],
  },
  {
    id: 4,
    type: "Farming Tips",
    title: "Pest Control Methods",
    description:
      "Natural and chemical methods to protect crops from pests effectively.",
    tags: ["Pest", "Organic", "Protection"],
  },
  {
    id: 5,
    type: "Guides",
    title: "Crop Disease Awareness",
    description:
      "Comprehensive guide on identifying symptoms, prevention, and remedies for crop diseases.",
    tags: ["Disease", "Symptoms", "Remedy"],
    link: "/disease-awareness"
  },
];

export default function ResourcesPage() {
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState("All");
  const [visibleCount, setVisibleCount] = useState(3);

  const filteredResources = resourcesData.filter((item) => {
    const searchText = search.toLowerCase();

    const matchSearch =
      item.title.toLowerCase().includes(searchText) ||
      item.description.toLowerCase().includes(searchText) ||
      item.tags.some((tag) => tag.toLowerCase().includes(searchText));

    const matchFilter = filter === "All" || item.type === filter;

    return matchSearch && matchFilter;
  });

  const visibleResources = filteredResources.slice(0, visibleCount);

  const categories = ["All", "Farming Tips", "Articles", "Guides"];

  return (
    <div className="resources-page">
      
      {/* HERO */}
      <div className="resources-hero">
        <h1>Knowledge Hub 🌱</h1>
        <p>Explore farming tips, guides, and agriculture insights</p>

        <input
          type="text"
          placeholder="Search by title, tags, or content..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="search-box"
        />
      </div>

      {/* FILTERS WITH COUNT */}
      <div className="filter-bar">
        {categories.map((type) => {
          const count =
            type === "All"
              ? resourcesData.length
              : resourcesData.filter((r) => r.type === type).length;

          return (
            <button
              key={type}
              className={filter === type ? "active" : ""}
              onClick={() => setFilter(type)}
            >
              {type} ({count})
            </button>
          );
        })}
      </div>

      {/* GRID */}
      <div className="resources-grid">
        {visibleResources.length > 0 ? (
          visibleResources.map((item) => (
            <div key={item.id} className="resource-card">
              <div className="card-type">{item.type}</div>

              <h3>{item.title}</h3>
              <p>{item.description}</p>

              <div className="tags">
                {item.tags.map((tag, i) => (
                   <span key={i}>{tag}</span>
                ))}
              </div>

              {item.link ? (
                <Link to={item.link}>
                  <button className="explore-btn">Explore →</button>
                </Link>
              ) : (
                <button className="explore-btn">Explore →</button>
              )}
            </div>
          ))
        ) : (
          <div className="no-results">
            <p>No resources found 😕</p>
            <button onClick={() => { setSearch(""); setFilter("All"); }}>
              Reset Filters
            </button>
          </div>
        )}
      </div>

      {/* LOAD MORE */}
      {visibleCount < filteredResources.length && (
        <div className="load-more">
          <button onClick={() => setVisibleCount((prev) => prev + 3)}>
            Load More
          </button>
        </div>
      )}

      {/* ABOUT SECTION */}
      <div className="about-section">
        <h2>About Knowledge Hub 🌾</h2>
        <p>
          This platform is designed to help farmers, students, and agriculture 
          enthusiasts access reliable knowledge in one place. From seasonal 
          farming tips to modern agricultural technologies, we aim to simplify 
          learning and improve productivity.
        </p>

        <div className="about-features">
          <div>
            <h4>🌱 Practical Tips</h4>
            <p>Real-world farming advice you can apply instantly.</p>
          </div>

          <div>
            <h4>📘 Guides</h4>
            <p>Step-by-step resources for beginners and experts.</p>
          </div>

          <div>
            <h4>🚀 Modern Tech</h4>
            <p>Stay updated with AI, smart farming, and innovations.</p>
          </div>
        </div>
      </div>
    </div>
  );
}