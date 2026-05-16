import React, { useState, useEffect } from 'react';
import { FaSearch, FaClock, FaUser, FaLeaf, FaSpinner, FaExclamationCircle } from 'react-icons/fa';
import { useTheme } from './ThemeContext';
import { fetchFarmingNews, getNewsCategories, formatNewsDate } from './services/newsApi';
import './FarmingNews.css';

export default function FarmingNews() {
  const { theme } = useTheme();
  const [articles, setArticles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize] = useState(10);
  const [selectedCategory, setSelectedCategory] = useState('All');
  const [searchTerm, setSearchTerm] = useState('');
  const [totalCount, setTotalCount] = useState(0);
  const [hasMore, setHasMore] = useState(false);

  const categories = getNewsCategories();

  // Fetch news articles
  useEffect(() => {
    const loadNews = async () => {
      try {
        setLoading(true);
        setError(null);

        const category = selectedCategory === 'All' ? null : selectedCategory;
        const response = await fetchFarmingNews(
          currentPage,
          pageSize,
          category,
          searchTerm || null
        );

        setArticles(response.articles);
        setTotalCount(response.total_count);
        setHasMore(response.has_more);
        // Scroll to top when new results load
        window.scrollTo({ top: 0, behavior: 'smooth' });
      } catch (err) {
        console.error('Failed to load news:', err);
        setError('Unable to load farming news. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    loadNews();
  }, [currentPage, selectedCategory, searchTerm, pageSize]);

  // Handle category change
  const handleCategoryChange = (category) => {
    setSelectedCategory(category);
    setCurrentPage(1);
  };

  // Handle search
  const handleSearch = (e) => {
    const value = e.target.value;
    setSearchTerm(value);
    setCurrentPage(1);
  };

  // Clear search
  const handleClearSearch = () => {
    setSearchTerm('');
    setCurrentPage(1);
  };

  // Handle pagination
  const handleNextPage = () => {
    if (hasMore) {
      setCurrentPage((prev) => prev + 1);
    }
  };

  const handlePrevPage = () => {
    if (currentPage > 1) {
      setCurrentPage((prev) => prev - 1);
    }
  };

  // Render skeleton loaders
  const renderSkeletons = () => {
    return Array.from({ length: pageSize }).map((_, i) => (
      <div key={`skeleton-${i}`} className="news-card-skeleton">
        <div className="skeleton-thumbnail"></div>
        <div className="skeleton-content">
          <div className="skeleton-line" style={{ width: '80%' }}></div>
          <div className="skeleton-line" style={{ width: '100%' }}></div>
          <div className="skeleton-line" style={{ width: '70%' }}></div>
        </div>
      </div>
    ));
  };

  // Render news cards
  const renderNewsCards = () => {
    if (articles.length === 0 && !loading) {
      return (
        <div className="farming-news-empty">
          <div className="empty-icon">
            <FaLeaf />
          </div>
          <h3>No News Found</h3>
          <p>
            {searchTerm
              ? 'Try adjusting your search terms'
              : 'No articles available in this category'}
          </p>
        </div>
      );
    }

    return articles.map((article) => (
      <div
        key={article.id}
        className="news-card"
        onClick={() => article.url && window.open(article.url, '_blank')}
        role="article"
        tabIndex="0"
        onKeyDown={(e) => {
          if ((e.key === 'Enter' || e.key === ' ') && article.url) {
            e.preventDefault();
            window.open(article.url, '_blank');
          }
        }}
      >
        {/* Thumbnail */}
        <div className="news-card-thumbnail">
          <img
            src={article.thumbnail}
            alt={article.title}
            loading="lazy"
            onError={(e) => {
              e.target.src =
                'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 200"%3E%3Crect fill="%232e7d32" width="400" height="200"/%3E%3Ctext x="50%25" y="50%25" dominant-baseline="middle" text-anchor="middle" font-family="Arial" font-size="24" fill="white"%3EFarming News%3C/text%3E%3C/svg%3E';
            }}
          />
          <div className="news-card-category-badge">{article.category}</div>
        </div>

        {/* Content */}
        <div className="news-card-content">
          <h3 className="news-card-title">{article.title}</h3>
          <p className="news-card-description">{article.description}</p>

          {/* Meta Information */}
          <div className="news-card-meta">
            <div className="news-meta-author">
              <FaUser size={14} aria-hidden="true" />
              <span>{article.author}</span>
            </div>
            <div className="news-meta-time">
              <FaClock size={14} aria-hidden="true" />
              <span title={article.date}>{formatNewsDate(article.date)}</span>
            </div>
          </div>

          {/* Read Time Badge */}
          {article.read_time && (
            <div style={{ fontSize: '0.85rem', color: 'var(--news-text-secondary)', marginTop: '8px' }}>
              {article.read_time}
            </div>
          )}
        </div>
      </div>
    ));
  };

  return (
    <div className="farming-news-container">
      {/* Hero Section */}
      <section className="farming-news-hero">
        <div className="farming-news-hero-content">
          <div className="farming-news-badge">
            <FaLeaf aria-hidden="true" />
            Latest Updates
          </div>
          <h1>Farming News & Updates</h1>
          <p>
            Stay informed with real-time agriculture news, weather alerts, and policy updates.
            Get the latest insights to optimize your farming decisions.
          </p>
        </div>
      </section>

      {/* Error State */}
      {error && (
        <div className="farming-news-error" role="alert">
          <FaExclamationCircle style={{ marginRight: '10px', verticalAlign: 'middle' }} />
          {error}
        </div>
      )}

      {/* Controls Section */}
      <div className="farming-news-controls">
        {/* Search Bar */}
        <div className="farming-news-search">
          <FaSearch className="search-icon" aria-hidden="true" />
          <input
            type="text"
            className="search-input"
            placeholder="Search news..."
            value={searchTerm}
            onChange={handleSearch}
            aria-label="Search farming news"
          />
          {searchTerm && (
            <button
              onClick={handleClearSearch}
              style={{
                background: 'none',
                border: 'none',
                cursor: 'pointer',
                color: 'var(--news-text-secondary)',
                fontSize: '1rem',
              }}
              aria-label="Clear search"
              title="Clear search"
            >
              ✕
            </button>
          )}
        </div>

        {/* Category Filter */}
        <div className="category-filter">
          {categories.map((category) => (
            <button
              key={category}
              className={`category-btn ${selectedCategory === category ? 'active' : ''}`}
              onClick={() => handleCategoryChange(category)}
              aria-label={`Filter by ${category}`}
              aria-pressed={selectedCategory === category}
            >
              {category}
            </button>
          ))}
        </div>
      </div>

      {/* News Grid */}
      <div className="farming-news-grid">
        {loading ? renderSkeletons() : renderNewsCards()}
      </div>

      {/* Pagination */}
      {!loading && articles.length > 0 && (
        <div className="farming-news-pagination">
          <button
            className="pagination-btn"
            onClick={handlePrevPage}
            disabled={currentPage === 1}
            aria-label="Previous page"
          >
            ← Previous
          </button>

          <span className="pagination-info">
            Page {currentPage} of {Math.ceil(totalCount / pageSize)} ({totalCount} articles)
          </span>

          <button
            className="pagination-btn"
            onClick={handleNextPage}
            disabled={!hasMore}
            aria-label="Next page"
          >
            Next →
          </button>
        </div>
      )}

      {/* Loading indicator */}
      {loading && (
        <div style={{ textAlign: 'center', padding: '20px', color: 'var(--news-text-secondary)' }}>
          <FaSpinner className="spinner" size={24} style={{ animation: 'spin 1s linear infinite' }} />
          <style>{`
            @keyframes spin {
              from { transform: rotate(0deg); }
              to { transform: rotate(360deg); }
            }
          `}</style>
        </div>
      )}
    </div>
  );
}
