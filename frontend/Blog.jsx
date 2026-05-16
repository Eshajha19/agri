import React, { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { FaSearch, FaArrowRight, FaClock, FaUser, FaLeaf, FaCloudSun, FaLandmark, FaBug, FaTint, FaSeedling, FaBookmark, FaRegBookmark } from "react-icons/fa";
import "./Blog.css";
import { getBookmarks, toggleBookmark } from "./utils/bookmarkStorage";

const BLOG_POSTS = [
  {
    id: 1,
    title: "Modern Drip Irrigation Techniques for Small Farms",
    description: "Discover how drip irrigation can reduce water consumption by up to 60% while boosting crop yields. Learn the setup process, maintenance tips, and which crops benefit most.",
    category: "Irrigation",
    author: "Dr. Anita Sharma",
    date: "April 28, 2026",
    readTime: "6 min read",
    thumbnail: "https://images.unsplash.com/photo-1464226184884-fa280b87c399?w=600&q=80",
  },
  {
    id: 2,
    title: "PM-KISAN Scheme: How to Apply and Maximise Your Benefits",
    description: "A complete guide to the Pradhan Mantri Kisan Samman Nidhi scheme. Learn the eligibility criteria, application process, and common mistakes to avoid when claiming support.",
    category: "Government Schemes",
    author: "Rajesh Verma",
    date: "April 22, 2026",
    readTime: "8 min read",
    thumbnail: "https://images.unsplash.com/photo-1554224155-8d04cb21cd6c?w=600&q=80",
  },
  {
    id: 3,
    title: "Identifying and Managing Rice Blast Disease",
    description: "Rice blast is one of the most destructive fungal diseases affecting paddy crops. This guide covers early identification signs, preventive cultural practices, and effective fungicide timings.",
    category: "Crop Management",
    author: "Prof. Suresh Patel",
    date: "April 18, 2026",
    readTime: "7 min read",
    thumbnail: "https://images.unsplash.com/photo-1595841696677-6489ff3f8cd1?w=600&q=80",
  },
  {
    id: 4,
    title: "Understanding the Southwest Monsoon and Your Kharif Season",
    description: "Accurate monsoon prediction is critical for kharif crop planning. Learn how to use IMD forecasts, interpret weather data, and plan sowing windows to minimise risk.",
    category: "Weather",
    author: "Meena Krishnan",
    date: "April 14, 2026",
    readTime: "5 min read",
    thumbnail: "https://images.unsplash.com/photo-1561470508-fd4df1ed90b2?w=600&q=80",
  },
  {
    id: 5,
    title: "Soil Health Card Scheme: Getting the Most from Your Soil Test",
    description: "Your soil health card contains crucial data about nutrients, pH, and micro-nutrients. This article explains how to read each parameter and apply inputs efficiently.",
    category: "Government Schemes",
    author: "Dr. Kavita Rao",
    date: "April 10, 2026",
    readTime: "6 min read",
    thumbnail: "https://images.unsplash.com/photo-1500937386664-56d1dfef3854?w=600&q=80",
  },
  {
    id: 6,
    title: "Integrated Pest Management for Cotton Crops",
    description: "Chemical-only pest control is becoming less effective. Explore IPM strategies for cotton including scouting protocols, biological controls, and pheromone traps to cut input costs.",
    category: "Pest Management",
    author: "Arvind Kulkarni",
    date: "April 5, 2026",
    readTime: "9 min read",
    thumbnail: "https://images.unsplash.com/photo-1500382017468-9049fed747ef?w=600&q=80",
  },
  {
    id: 7,
    title: "Organic Farming Transition: A Step-by-Step Guide",
    description: "Transitioning to organic farming can unlock premium markets and improve soil health. This guide covers the three-year transition process, NPOP certification, and marketing your produce.",
    category: "Crop Management",
    author: "Sunita Devi",
    date: "March 30, 2026",
    readTime: "10 min read",
    thumbnail: "https://images.unsplash.com/photo-1625246333195-78d9c38ad449?w=600&q=80",
  },
  {
    id: 8,
    title: "Rainfall Deficiency and Drought Management Strategies",
    description: "Climate variability is increasing drought frequency across India. Learn water harvesting techniques, drought-tolerant variety selection, and crop insurance options to safeguard your livelihood.",
    category: "Weather",
    author: "Ramesh Kumar",
    date: "March 20, 2026",
    readTime: "8 min read",
    thumbnail: "https://images.unsplash.com/photo-1542601906990-b4d3fb778b09?w=600&q=80",
  },
];

const CATEGORIES = ["All", "Crop Management", "Weather", "Government Schemes", "Irrigation", "Pest Management"];

const CATEGORY_ICONS = {
  "Crop Management": <FaSeedling />,
  Weather: <FaCloudSun />,
  "Government Schemes": <FaLandmark />,
  Irrigation: <FaTint />,
  "Pest Management": <FaBug />,
};

export default function Blog() {
  const [activeCategory, setActiveCategory] = useState("All");
  const [searchTerm, setSearchTerm] = useState("");
  const [bookmarkedArticleIds, setBookmarkedArticleIds] = useState(() =>
    getBookmarks("articles").map((article) => article.id)
  );

  useEffect(() => {
    setBookmarkedArticleIds(getBookmarks("articles").map((article) => article.id));
  }, []);

  const handleToggleArticleBookmark = (post) => {
    const updated = toggleBookmark("articles", post);
    setBookmarkedArticleIds(updated.map((item) => item.id));
  };

  React.useEffect(() => {
    window.scrollTo(0, 0);
  }, []);

  const filteredPosts = BLOG_POSTS.filter((post) => {
    const matchesCategory = activeCategory === "All" || post.category === activeCategory;
    const matchesSearch =
      post.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
      post.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
      post.author.toLowerCase().includes(searchTerm.toLowerCase());
    return matchesCategory && matchesSearch;
  });

  return (
    <div className="blog-page">
      <div className="blog-hero">
        <div className="blog-hero-content">
          <div className="blog-hero-badge">
            <FaLeaf />
            Knowledge Hub
          </div>
          <h1>Farming Insights &amp; Guides</h1>
          <p>
            Expert articles on crop management, weather planning, government schemes, and
            modern agricultural practices to help you farm smarter.
          </p>
        </div>
      </div>

      <div className="blog-controls">
        <div className="blog-search-wrap">
          <FaSearch className="blog-search-icon" />
           <input
             id="blog-search-input"
             type="text"
             placeholder="Search articles, topics, or authors..."
             value={searchTerm}
             onChange={(e) => setSearchTerm(e.target.value)}
             className="blog-search-input"
           />
         </div>

        <div className="blog-filter-chips">
          {CATEGORIES.map((cat) => (
            <button
              key={cat}
              id={`blog-filter-${cat.replace(/\s+/g, "-").toLowerCase()}`}
              className={`blog-chip ${activeCategory === cat ? "active" : ""}`}
              onClick={() => setActiveCategory(cat)}
            >
              {cat !== "All" && <span className="chip-icon">{CATEGORY_ICONS[cat]}</span>}
              {cat}
            </button>
          ))}
        </div>
      </div>

      <div className="blog-results-meta">
        <span>
          {filteredPosts.length} article{filteredPosts.length !== 1 ? "s" : ""} found
        </span>
      </div>

      {filteredPosts.length > 0 ? (
        <div className="blog-grid">
          {filteredPosts.map((post) => (
            <article key={post.id} className="blog-card">
              <div className="blog-card-thumbnail">
                <img src={post.thumbnail} alt={post.title} loading="lazy" />
                <div className="blog-card-category">
                  <span className="cat-icon">{CATEGORY_ICONS[post.category]}</span>
                  {post.category}
                </div>
              </div>
              <div className="blog-card-body">
                <h2 className="blog-card-title">{post.title}</h2>
                <p className="blog-card-desc">{post.description}</p>
                <div className="blog-card-meta">
                  <span className="meta-item">
                    <FaUser /> {post.author}
                  </span>
                  <span className="meta-item">
                    <FaClock /> {post.readTime}
                  </span>
                </div>
                <div className="blog-card-footer">
                  <span className="blog-card-date">{post.date}</span>
                  <div className="blog-card-actions">
                    <button
                      className={`bookmark-btn ${bookmarkedArticleIds.includes(post.id) ? "active" : ""}`}
                      onClick={() => handleToggleArticleBookmark(post)}
                      aria-label={bookmarkedArticleIds.includes(post.id) ? "Remove bookmark" : "Bookmark article"}
                    >
                      {bookmarkedArticleIds.includes(post.id) ? <FaBookmark /> : <FaRegBookmark />} {bookmarkedArticleIds.includes(post.id) ? "Saved" : "Save"}
                    </button>
                    <Link
                      to={`/blog/${post.id}`}
                      id={`blog-read-more-${post.id}`}
                      className="btn-read-more"
                    >
                      Read More <FaArrowRight />
                    </Link>
                  </div>
                </div>
              </div>
            </article>
          ))}
        </div>
      ) : (
        <div className="blog-empty">
          <FaLeaf className="empty-icon" />
          <h3>No articles found</h3>
          <p>Try adjusting your search term or selecting a different category.</p>
        </div>
      )}
    </div>
  );
}