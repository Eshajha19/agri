import React, { useState, useMemo, useEffect } from "react";
import { Link } from "react-router-dom";
import {
  FaSearch, FaArrowRight, FaClock, FaUser,
  FaLeaf
} from "react-icons/fa";
import "./Blog.css";

export default function Blog() {
  const [activeCategory, setActiveCategory] = useState("All");
  const [searchTerm, setSearchTerm] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");

  useEffect(() => {
    window.scrollTo(0, 0);
  }, []);

  // ✅ Debounce search (better performance UX)
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedSearch(searchTerm.trim().toLowerCase());
    }, 300);

    return () => clearTimeout(timer);
  }, [searchTerm]);

  // ✅ Optimized filtering
  const filteredPosts = useMemo(() => {
    return BLOG_POSTS.filter((post) => {
      const matchesCategory =
        activeCategory === "All" || post.category === activeCategory;

      if (!debouncedSearch) return matchesCategory;

      const matchesSearch =
        post.title.toLowerCase().includes(debouncedSearch) ||
        post.description.toLowerCase().includes(debouncedSearch) ||
        post.author.toLowerCase().includes(debouncedSearch);

      return matchesCategory && matchesSearch;
    });
  }, [activeCategory, debouncedSearch]);

  return (
    <div className="blog-page">

      {/* Hero */}
      <div className="blog-hero">
        <div className="blog-hero-content">
          <div className="blog-hero-badge">
            <FaLeaf /> Knowledge Hub
          </div>

          <h1>Farming Insights & Guides</h1>

          <p>
            Expert articles on crop management, weather planning, and modern
            agricultural practices to help you farm smarter.
          </p>
        </div>
      </div>

      {/* Controls */}
      <div className="blog-controls">

        {/* Search */}
        <div className="blog-search-wrap">
          <FaSearch className="blog-search-icon" />
          <input
            type="text"
            placeholder="Search articles, topics, or authors..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="blog-search-input"
            aria-label="Search blog articles"
          />
        </div>

        {/* Filters */}
        <div className="blog-filter-chips">
          {CATEGORIES.map((cat) => (
            <button
              key={cat}
              className={`blog-chip ${activeCategory === cat ? "active" : ""}`}
              onClick={() => setActiveCategory(cat)}
              aria-pressed={activeCategory === cat}
            >
              {cat !== "All" && (
                <span className="chip-icon">
                  {CATEGORY_ICONS[cat]}
                </span>
              )}
              {cat}
            </button>
          ))}
        </div>
      </div>

      {/* Results */}
      <div className="blog-results-meta">
        <span>
          {filteredPosts.length === 0
            ? "No results"
            : `${filteredPosts.length} article${filteredPosts.length !== 1 ? "s" : ""}`}
        </span>
      </div>

      {/* Grid */}
      {filteredPosts.length > 0 ? (
        <div className="blog-grid">
          {filteredPosts.map((post) => (
            <article key={post.id} className="blog-card">

              {/* Thumbnail */}
              <div className="blog-card-thumbnail">
                <img
                  src={post.thumbnail}
                  alt={`Thumbnail for ${post.title}`}
                  loading="lazy"
                />

                <div className="blog-card-category">
                  <span className="cat-icon">
                    {CATEGORY_ICONS[post.category]}
                  </span>
                  {post.category}
                </div>
              </div>

              {/* Body */}
              <div className="blog-card-body">
                <h2 className="blog-card-title">
                  {post.title}
                </h2>

                <p className="blog-card-desc">
                  {post.description.length > 120
                    ? post.description.slice(0, 120) + "..."
                    : post.description}
                </p>

                <div className="blog-card-meta">
                  <span className="meta-item">
                    <FaUser /> {post.author}
                  </span>

                  <span className="meta-item">
                    <FaClock /> {post.readTime}
                  </span>
                </div>

                <div className="blog-card-footer">
                  <span className="blog-card-date">
                    {post.date}
                  </span>

                  <Link
                    to={`/blog/${post.id}`}
                    className="btn-read-more"
                    aria-label={`Read more about ${post.title}`}
                  >
                    Read More <FaArrowRight />
                  </Link>
                </div>
              </div>

            </article>
          ))}
        </div>
      ) : (
        <div className="blog-empty">
          <FaLeaf className="empty-icon" />
          <h3>No articles found</h3>
          <p>
            Try a different keyword or category.
          </p>
        </div>
      )}
    </div>
  );
}