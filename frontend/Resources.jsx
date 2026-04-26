import React, { useState } from "react";
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
];

export default function ResourcesPage() {
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState("All");

  const filteredResources = resourcesData.filter((item) => {
    const matchSearch =
      item.title.toLowerCase().includes(search.toLowerCase());

    const matchFilter = filter === "All" || item.type === filter;

    return matchSearch && matchFilter;
  });

  return (
    <div className="resources-page">

      {/* HERO */}
      <div className="resources-hero">
        <h1>Knowledge Hub 🌱</h1>
        <p>Explore farming tips, guides, and agriculture insights</p>

        {/* SEARCH */}
        <input
          type="text"
          placeholder="Search resources..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="search-box"
        />
      </div>

      {/* FILTER BUTTONS */}
      <div className="filter-bar">
        {["All", "Farming Tips", "Articles", "Guides"].map((type) => (
          <button
            key={type}
            className={filter === type ? "active" : ""}
            onClick={() => setFilter(type)}
          >
            {type}
          </button>
        ))}
      </div>

      {/* GRID */}
      <div className="resources-grid">
        {filteredResources.length > 0 ? (
          filteredResources.map((item) => (
            <div key={item.id} className="resource-card">
              <div className="card-type">{item.type}</div>

              <h3>{item.title}</h3>
              <p>{item.description}</p>

              <div className="tags">
                {item.tags.map((tag, i) => (
                  <span key={i}>{tag}</span>
                ))}
              </div>

              <button>Explore →</button>
            </div>
          ))
        ) : (
          <p className="no-results">No resources found 😕</p>
        )}
      </div>
    </div>
  );
}