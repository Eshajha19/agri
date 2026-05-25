import React, { useState, useEffect, useMemo, useCallback } from 'react';
import {
  FaSearch,
  FaClock,
  FaUser,
  FaLeaf,
  FaSpinner,
  FaExclamationCircle,
  FaRedo,
} from 'react-icons/fa';

import { useTheme } from './ThemeContext';
import {
  fetchFarmingNews,
  getNewsCategories,
  formatNewsDate,
} from './services/newsApi';

import './FarmingNews.css';

export default function FarmingNews({ userData }) {
  const { theme } = useTheme();

  const [articles, setArticles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize] = useState(10);

  const [searchInput, setSearchInput] = useState('');
  const [searchTerm, setSearchTerm] = useState('');

  const [totalCount, setTotalCount] = useState(0);
  const [hasMore, setHasMore] = useState(false);

  /**
   * Auto-select crop related category
   */
  const cropSpecificCategory = useMemo(() => {
    if (!userData?.cropType) return null;

    const crop = userData.cropType.toLowerCase();

    const supportedCrops = [
      'rice',
      'paddy',
      'wheat',
      'cotton',
      'maize',
      'sugarcane',
      'vegetables',
      'fruits',
      'soybean',
      'potato',
      'onion',
      'tomato',
    ];

    return supportedCrops.some((item) =>
      crop.includes(item)
    )
      ? 'Crop Management'
      : null;
  }, [userData?.cropType]);

  const initialCategory = cropSpecificCategory || 'All';

  const [selectedCategory, setSelectedCategory] =
    useState(initialCategory);

  /**
   * Reset category when crop changes
   */
  useEffect(() => {
    setSelectedCategory(cropSpecificCategory || 'All');
    setCurrentPage(1);
  }, [cropSpecificCategory]);

  const categories = getNewsCategories();

  /**
   * Load News
   */
  const loadNews = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const category =
        selectedCategory === 'All'
          ? null
          : selectedCategory;

      const response = await fetchFarmingNews(
        currentPage,
        pageSize,
        category,
        searchTerm || null
      );

      const fetchedArticles =
        response?.articles || [];

      setArticles(
        Array.isArray(fetchedArticles)
          ? fetchedArticles
          : []
      );

      setTotalCount(
        response?.total_count ||
          response?.total ||
          fetchedArticles.length ||
          0
      );

      setHasMore(
        response?.has_more ||
          currentPage * pageSize <
            (response?.total_count ||
              response?.total ||
              0)
      );

      window.scrollTo({
        top: 0,
        behavior: 'smooth',
      });
    } catch (err) {
      console.error(
        'Failed to load farming news:',
        err
      );

      setArticles([]);

      setError(
        err?.message ||
          'Unable to load farming news. Please try again later.'
      );
    } finally {
      setLoading(false);
    }
  }, [
    currentPage,
    pageSize,
    selectedCategory,
    searchTerm,
  ]);

  /**
   * Fetch news on change
   */
  useEffect(() => {
    loadNews();
  }, [loadNews]);

  /**
   * Debounced Search
   */
  useEffect(() => {
    const delay = setTimeout(() => {
      setSearchTerm(searchInput.trim());
      setCurrentPage(1);
    }, 500);

    return () => clearTimeout(delay);
  }, [searchInput]);

  /**
   * Category Change
   */
  const handleCategoryChange = (category) => {
    if (loading) return;

    setSelectedCategory(category);
    setCurrentPage(1);
  };

  /**
   * Clear Search
   */
  const handleClearSearch = () => {
    setSearchInput('');
    setSearchTerm('');
    setCurrentPage(1);
  };

  /**
   * Pagination
   */
  const handleNextPage = () => {
    if (hasMore && !loading) {
      setCurrentPage((prev) => prev + 1);
    }
  };

  const handlePrevPage = () => {
    if (currentPage > 1 && !loading) {
      setCurrentPage((prev) => prev - 1);
    }
  };

  /**
   * Skeleton Loader
   */
  const renderSkeletons = () => {
    return Array.from({
      length: pageSize,
    }).map((_, i) => (
      <div
        key={`skeleton-${i}`}
        className="news-card-skeleton"
      >
        <div className="skeleton-thumbnail" />

        <div className="skeleton-content">
          <div
            className="skeleton-line"
            style={{ width: '80%' }}
          />

          <div
            className="skeleton-line"
            style={{ width: '100%' }}
          />

          <div
            className="skeleton-line"
            style={{ width: '70%' }}
          />
        </div>
      </div>
    ));
  };

  /**
   * News Cards
   */
  const renderNewsCards = () => {
    if (!loading && articles.length === 0) {
      return (
        <div className="farming-news-empty">
          <div className="empty-icon">
            <FaLeaf />
          </div>

          <h3>No News Found</h3>

          <p>
            {searchTerm
              ? 'Try adjusting your search keywords.'
              : 'No articles available in this category.'}
          </p>

          <button
            className="retry-btn"
            onClick={loadNews}
          >
            Reload News
          </button>
        </div>
      );
    }

    return articles.map((article, index) => (
      <div
        key={article.id || index}
        className="news-card"
        onClick={() => {
          if (article.url) {
            window.open(
              article.url,
              '_blank',
              'noopener,noreferrer'
            );
          }
        }}
        role="article"
        tabIndex="0"
        onKeyDown={(e) => {
          if (
            (e.key === 'Enter' || e.key === ' ') &&
            article.url
          ) {
            e.preventDefault();

            window.open(
              article.url,
              '_blank',
              'noopener,noreferrer'
            );
          }
        }}
      >
        {/* Thumbnail */}
        <div className="news-card-thumbnail">
          <img
            src={
              article.thumbnail ||
              article.image ||
              'https://via.placeholder.com/400x200?text=Farming+News'
            }
            alt={article.title || 'Farming News'}
            loading="lazy"
            onError={(e) => {
              e.target.src =
                'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 200"%3E%3Crect fill="%232e7d32" width="400" height="200"/%3E%3Ctext x="50%25" y="50%25" dominant-baseline="middle" text-anchor="middle" font-family="Arial" font-size="24" fill="white"%3EFarming News%3C/text%3E%3C/svg%3E';
            }}
          />

          <div className="news-card-category-badge">
            {article.category || 'Agriculture'}
          </div>
        </div>

        {/* Content */}
        <div className="news-card-content">
          <h3 className="news-card-title">
            {article.title ||
              'Untitled Farming Article'}
          </h3>

          <p className="news-card-description">
            {article.description ||
              'No description available for this article.'}
          </p>

          {/* Meta */}
          <div className="news-card-meta">
            <div className="news-meta-author">
              <FaUser
                size={14}
                aria-hidden="true"
              />

              <span>
                {article.author || 'Agri News'}
              </span>
            </div>

            <div className="news-meta-time">
              <FaClock
                size={14}
                aria-hidden="true"
              />

              <span title={article.date}>
                {article.date
                  ? formatNewsDate(article.date)
                  : 'Recent'}
              </span>
            </div>
          </div>

          {/* Read Time */}
          {article.read_time && (
            <div className="read-time">
              {article.read_time}
            </div>
          )}
        </div>
      </div>
    ));
  };

  return (
    <div
      className={`farming-news-container ${theme}`}
    >
      {/* Hero */}
      <section className="farming-news-hero">
        <div className="farming-news-hero-content">
          <div className="farming-news-badge">
            <FaLeaf aria-hidden="true" />
            Latest Updates
          </div>

          <h1>Farming News & Updates</h1>

          <p>
            Stay informed with real-time
            agriculture news, weather alerts,
            crop management insights, and
            government policy updates.
          </p>
        </div>
      </section>

      {/* Error */}
      {error && (
        <div
          className="farming-news-error"
          role="alert"
        >
          <FaExclamationCircle
            style={{
              marginRight: '10px',
              verticalAlign: 'middle',
            }}
          />

          {error}

          <button
            className="retry-btn"
            onClick={loadNews}
          >
            <FaRedo />
            Retry
          </button>
        </div>
      )}

      {/* Controls */}
      <div className="farming-news-controls">
        {/* Search */}
        <div className="farming-news-search">
          <FaSearch
            className="search-icon"
            aria-hidden="true"
          />

          <input
            type="text"
            className="search-input"
            placeholder="Search farming news..."
            value={searchInput}
            onChange={(e) =>
              setSearchInput(e.target.value)
            }
            aria-label="Search farming news"
            disabled={loading && articles.length === 0}
          />

          {searchInput && (
            <button
              onClick={handleClearSearch}
              className="clear-search-btn"
              aria-label="Clear search"
            >
              ✕
            </button>
          )}
        </div>

        {/* Categories */}
        <div className="category-filter">
          {categories.map((category) => (
            <button
              key={category}
              className={`category-btn ${
                selectedCategory === category
                  ? 'active'
                  : ''
              }`}
              onClick={() =>
                handleCategoryChange(category)
              }
              disabled={
                loading && articles.length === 0
              }
              aria-pressed={
                selectedCategory === category
              }
            >
              {category}
            </button>
          ))}
        </div>
      </div>

      {/* News Grid */}
      <div className="farming-news-grid">
        {loading
          ? renderSkeletons()
          : renderNewsCards()}
      </div>

      {/* Pagination */}
      {!loading && articles.length > 0 && (
        <div className="farming-news-pagination">
          <button
            className="pagination-btn"
            onClick={handlePrevPage}
            disabled={currentPage === 1}
          >
            ← Previous
          </button>

          <span className="pagination-info">
            Page {currentPage} of{' '}
            {Math.max(
              1,
              Math.ceil(totalCount / pageSize)
            )}{' '}
            ({totalCount} articles)
          </span>

          <button
            className="pagination-btn"
            onClick={handleNextPage}
            disabled={!hasMore}
          >
            Next →
          </button>
        </div>
      )}

      {/* Loading Spinner */}
      {loading && (
        <div className="loading-indicator">
          <FaSpinner
            className="spinner"
            size={24}
          />

          <span>Loading farming news...</span>
        </div>
      )}
    </div>
  );
}