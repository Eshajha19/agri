import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import { FaSearch, FaClock, FaUser, FaLeaf, FaSpinner, FaExclamationCircle, FaSync, FaArrowDown } from 'react-icons/fa';
import { useTheme } from './ThemeContext';
import { fetchFarmingNews, getNewsCategories, formatNewsDate } from './services/newsApi';
import './FarmingNews.css';

const AUTO_REFRESH_INTERVAL = 60000;
const PAGE_SIZE = 10;

export default function FarmingNews({ userData }) {
  const { theme } = useTheme();
  const [articles, setArticles] = useState([]);
  const [featuredArticles, setFeaturedArticles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [selectedCategory, setSelectedCategory] = useState('All');
  const [searchTerm, setSearchTerm] = useState('');
  const [totalCount, setTotalCount] = useState(0);
  const [hasMore, setHasMore] = useState(false);
  const [sortBy, setSortBy] = useState('latest');
  const [lastUpdated, setLastUpdated] = useState(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshCountdown, setRefreshCountdown] = useState(AUTO_REFRESH_INTERVAL / 1000);
  const sentinelRef = useRef(null);
  const refreshTimerRef = useRef(null);
  const countdownTimerRef = useRef(null);

  const categories = getNewsCategories();

  const cropSpecificCategory = useMemo(() => {
    if (!userData?.cropType) return null;
    const crop = userData.cropType.toLowerCase();
    const cropMap = ['rice', 'paddy', 'wheat', 'cotton', 'maize', 'sugarcane', 'vegetables', 'fruits', 'soybean', 'potato', 'onion', 'tomato'];
    if (cropMap.some(c => crop.includes(c))) return 'Crop Management';
    return null;
  }, [userData?.cropType]);

  useEffect(() => {
    setSelectedCategory(cropSpecificCategory || 'All');
    setCurrentPage(1);
  }, [cropSpecificCategory]);

  const loadNews = useCallback(async (page, append = false) => {
    try {
      if (append) {
        setLoadingMore(true);
      } else {
        setLoading(true);
      }
      setError(null);

      const category = selectedCategory === 'All' ? null : selectedCategory;
      const response = await fetchFarmingNews(page, PAGE_SIZE, category, searchTerm || null, sortBy);

      if (append) {
        setArticles(prev => [...prev, ...response.articles]);
      } else {
        setArticles(response.articles);
      }
      setTotalCount(response.total_count);
      setHasMore(response.has_more);
      setLastUpdated(new Date());

      if (!append) {
        window.scrollTo({ top: 0, behavior: 'smooth' });
      }
    } catch (err) {
      console.error('Failed to load news:', err);
      setError('Unable to load farming news. Please try again later.');
    } finally {
      setLoading(false);
      setLoadingMore(false);
    }
  }, [selectedCategory, searchTerm, sortBy]);

  useEffect(() => {
    loadNews(1, false);
  }, [loadNews]);

  useEffect(() => {
    const loadFeatured = async () => {
      try {
        const category = selectedCategory === 'All' ? null : selectedCategory;
        const params = category ? `?category=${encodeURIComponent(category)}` : '';
        const response = await fetch(`https://${window.location.hostname}:${window.location.port || '5173'}/api/farming-news/featured${params}`);
        if (response.ok) {
          const data = await response.json();
          setFeaturedArticles(data.articles || []);
        }
      } catch {
        setFeaturedArticles([]);
      }
    };
    loadFeatured();
  }, [selectedCategory]);

  useEffect(() => {
    if (!autoRefresh) {
      clearInterval(refreshTimerRef.current);
      clearInterval(countdownTimerRef.current);
      return;
    }

    setRefreshCountdown(AUTO_REFRESH_INTERVAL / 1000);

    countdownTimerRef.current = setInterval(() => {
      setRefreshCountdown(prev => (prev <= 1 ? AUTO_REFRESH_INTERVAL / 1000 : prev - 1));
    }, 1000);

    refreshTimerRef.current = setInterval(() => {
      loadNews(1, false);
      setRefreshCountdown(AUTO_REFRESH_INTERVAL / 1000);
    }, AUTO_REFRESH_INTERVAL);

    return () => {
      clearInterval(refreshTimerRef.current);
      clearInterval(countdownTimerRef.current);
    };
  }, [autoRefresh, loadNews]);

  useEffect(() => {
    if (!sentinelRef.current || !hasMore || loading || loadingMore) return;

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting && hasMore && !loading && !loadingMore) {
          setCurrentPage(prev => {
            const nextPage = prev + 1;
            loadNews(nextPage, true);
            return nextPage;
          });
        }
      },
      { threshold: 0.1 }
    );

    observer.observe(sentinelRef.current);
    return () => observer.disconnect();
  }, [hasMore, loading, loadingMore, loadNews]);

  const handleCategoryChange = (category) => {
    setSelectedCategory(category);
    setCurrentPage(1);
    setArticles([]);
  };

  const handleSearch = (e) => {
    setSearchTerm(e.target.value);
    setCurrentPage(1);
  };

  const handleClearSearch = () => {
    setSearchTerm('');
    setCurrentPage(1);
  };

  const handleRefresh = () => {
    loadNews(1, false);
    setRefreshCountdown(AUTO_REFRESH_INTERVAL / 1000);
  };

  const renderSkeletons = (count) => {
    return Array.from({ length: count }).map((_, i) => (
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

  const renderFeaturedCard = (article) => (
    <div
      key={`featured-${article.id}`}
      className="news-card news-card-featured"
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
      <div className="news-card-thumbnail">
        <img
          src={article.thumbnail}
          alt={article.title}
          loading="lazy"
          onError={(e) => {
            e.target.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 200"%3E%3Crect fill="%232e7d32" width="400" height="200"/%3E%3Ctext x="50%25" y="50%25" dominant-baseline="middle" text-anchor="middle" font-family="Arial" font-size="24" fill="white"%3EFarming News%3C/text%3E%3C/svg%3E';
          }}
        />
        <div className="news-card-featured-badge">Featured</div>
        <div className="news-card-category-badge">{article.category}</div>
      </div>
      <div className="news-card-content">
        <h3 className="news-card-title">{article.title}</h3>
        <p className="news-card-description">{article.description}</p>
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
      </div>
    </div>
  );

  const renderNewsCard = (article) => (
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
      <div className="news-card-thumbnail">
        <img
          src={article.thumbnail}
          alt={article.title}
          loading="lazy"
          onError={(e) => {
            e.target.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 200"%3E%3Crect fill="%232e7d32" width="400" height="200"/%3E%3Ctext x="50%25" y="50%25" dominant-baseline="middle" text-anchor="middle" font-family="Arial" font-size="24" fill="white"%3EFarming News%3C/text%3E%3C/svg%3E';
          }}
        />
        <div className="news-card-category-badge">{article.category}</div>
      </div>
      <div className="news-card-content">
        <h3 className="news-card-title">{article.title}</h3>
        <p className="news-card-description">{article.description}</p>
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
        {article.read_time && (
          <div className="news-card-readtime">{article.read_time}</div>
        )}
      </div>
    </div>
  );

  const renderEmptyState = () => (
    <div className="farming-news-empty">
      <div className="empty-icon"><FaLeaf /></div>
      <h3>No News Found</h3>
      <p>{searchTerm ? 'Try adjusting your search terms' : 'No articles available in this category'}</p>
    </div>
  );

  return (
    <div className="farming-news-container">
      <section className="farming-news-hero">
        <div className="farming-news-hero-content">
          <div className="farming-news-badge">
            <FaLeaf aria-hidden="true" />
            Real-Time Updates
          </div>
          <h1>Farming News & Updates</h1>
          <p>
            Stay informed with real-time agriculture news, weather alerts, and policy updates.
            News auto-refreshes every 60 seconds.
          </p>
        </div>
      </section>

      {error && (
        <div className="farming-news-error" role="alert">
          <FaExclamationCircle style={{ marginRight: '10px', verticalAlign: 'middle' }} />
          {error}
        </div>
      )}

      <div className="farming-news-controls">
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
            <button onClick={handleClearSearch} className="search-clear-btn" aria-label="Clear search" title="Clear search">✕</button>
          )}
        </div>

        <div className="farming-news-toolbar">
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

          <div className="farming-news-controls-right">
            <div className="sort-controls">
              <label className="sort-label">Sort:</label>
              <select
                className="sort-select"
                value={sortBy}
                onChange={(e) => { setSortBy(e.target.value); setCurrentPage(1); }}
                aria-label="Sort news articles"
              >
                <option value="latest">Latest</option>
                <option value="relevant">Most Relevant</option>
              </select>
            </div>

            <div className="auto-refresh-controls">
              <button
                className={`refresh-toggle ${autoRefresh ? 'active' : ''}`}
                onClick={() => setAutoRefresh(!autoRefresh)}
                aria-label={autoRefresh ? 'Disable auto-refresh' : 'Enable auto-refresh'}
                title={autoRefresh ? `Auto-refreshes in ${refreshCountdown}s` : 'Enable auto-refresh'}
              >
                <FaSync className={autoRefresh ? 'spin' : ''} size={14} />
                {autoRefresh && <span className="refresh-countdown">{refreshCountdown}s</span>}
              </button>
              <button className="refresh-btn" onClick={handleRefresh} aria-label="Refresh news now" title="Refresh now">
                <FaSync size={14} /> Refresh
              </button>
            </div>
          </div>
        </div>
      </div>

      {lastUpdated && (
        <div className="farming-news-last-updated">
          Last updated: {formatNewsDate(lastUpdated.toISOString())}
          {autoRefresh && <span className="auto-refresh-badge">Auto-refresh ON</span>}
        </div>
      )}

      {featuredArticles.length > 0 && !searchTerm && selectedCategory === 'All' && (
        <div className="farming-news-featured-section">
          <h2 className="featured-section-title">Featured Stories</h2>
          <div className="farming-news-featured-grid">
            {featuredArticles.map(renderFeaturedCard)}
          </div>
        </div>
      )}

      <div className="farming-news-grid">
        {loading ? renderSkeletons(PAGE_SIZE) : articles.length === 0 ? renderEmptyState() : articles.map(renderNewsCard)}
      </div>

      {loadingMore && (
        <div className="farming-news-loading-more">
          <FaSpinner className="spin" size={20} />
          <span>Loading more articles...</span>
        </div>
      )}

      <div ref={sentinelRef} className="farming-news-sentinel" />

      {!loading && articles.length > 0 && (
        <div className="farming-news-footer">
          <span className="farming-news-count">{totalCount} articles total</span>
          {!hasMore && <span className="farming-news-end">You've reached the end</span>}
        </div>
      )}
    </div>
  );
}
