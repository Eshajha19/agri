import React, { useState, useEffect, useMemo } from "react";
import { FaGithub, FaLinkedin, FaTwitter, FaCrown, FaCode, FaStar, FaSearch, FaSort } from "react-icons/fa";
import "./Contributors.css";
import heroIllustration from "./hero-illustration.png";

export default function Contributors() {
  const [contributors, setContributors] = useState([]);
  const [filter, setFilter] = useState("All");
  const [search, setSearch] = useState("");
  const [sortBy, setSortBy] = useState("contributions");
  const [loading, setLoading] = useState(true);
  const [errorMessage, setErrorMessage] = useState("");

  // Fetch contributors from GitHub API
  useEffect(() => {
    const fallbackContributors = [
      {
        id: 1,
        name: "eshajha19",
        role: "Owner & Founder",
        image: "https://avatars.githubusercontent.com/u/1?v=4",
        github: "https://github.com/eshajha19",
        contributions: 999,
        isOwner: true,
      },
    ];

    const fetchContributors = async () => {
      setLoading(true);
      try {
         const response = await fetch(
           "https://api.github.com/repos/Eshajha19/agri/contributors?per_page=100"
         );
         if (response.ok) {
           const data = await response.json();
           const mappedContributors = data.map((contributor) => ({
             id: contributor.id,
             name: contributor.login,
             role: contributor.login.toLowerCase() === 'eshajha19' ? 'Owner & Founder' : 'Contributor',
             image: contributor.avatar_url,
             github: contributor.html_url,
             contributions: contributor.contributions,
             isOwner: contributor.login.toLowerCase() === 'eshajha19',
           }));
           setContributors(mappedContributors);
         } else if (response.status === 403) {
           // Likely rate-limited; show friendly message and fallback
           setErrorMessage("GitHub API rate limit reached. Showing fallback contributors.");
           setContributors(fallbackContributors);
         } else {
           setErrorMessage("Unable to fetch contributors. Showing fallback data.");
           setContributors(fallbackContributors);
         }
      } catch (error) {
        console.error("Error fetching GitHub contributors:", error);
        setErrorMessage("Error fetching contributors. Showing fallback data.");
        setContributors(fallbackContributors);
      } finally {
        setLoading(false);
      }
    };

    fetchContributors();
  }, []);

  const roles = ["All", ...new Set(contributors.map((c) => c.role))];

  const filteredContributors = useMemo(() => {
    let list = contributors.slice();

    // role filter
    if (filter !== "All") {
      list = list.filter((c) => c.role === filter);
    }

    // search filter
    if (search && search.trim().length > 0) {
      const q = search.toLowerCase();
      list = list.filter((c) => c.name.toLowerCase().includes(q));
    }

    // sort
    if (sortBy === "contributions") {
      list.sort((a, b) => (b.contributions || 0) - (a.contributions || 0));
    } else if (sortBy === "name") {
      list.sort((a, b) => a.name.localeCompare(b.name));
    }

    return list;
  }, [contributors, filter, search, sortBy]);

  return (
    <div className="contributors-page">
      {/* HERO + CTA SECTION (enhanced for new contributors) */}
      <div className="contributors-hero enhanced-hero">
        <div className="hero-inner">
          <div className="hero-left">
            <h1>Join the Fasal Saathi Community</h1>
            <p className="hero-animate">
              <span className="notranslate" translate="no">Fasal Saathi</span> is an open, farmer-first platform — built
              by developers, agronomists, designers, and farmers. We welcome contributors of every skill level. Start small,
              learn fast, and see your work help real farms.
            </p>

            <div className="hero-badges">
              <span className="badge badge-pulse">Beginner-friendly</span>
              <span className="badge">Mentored PRs</span>
              <span className="badge">Field-tested</span>
            </div>

            <div className="cta-buttons hero-cta">
              <a href="https://github.com/Eshajha19/agri/issues" target="_blank" rel="noopener noreferrer" className="btn btn-primary btn-cta">
                Explore Issues
              </a>
              <a href="https://github.com/Eshajha19/agri/blob/main/CONTRIBUTING.md" target="_blank" rel="noopener noreferrer" className="btn btn-outline btn-cta">
                Get Started Guide
              </a>
            </div>

            <ul className="how-to">
              <li>🟢 Pick a "good first issue"</li>
              <li>🛠️ Open a PR — we'll review and mentor</li>
              <li>🌾 Share farmer feedback — make impact</li>
            </ul>
          </div>

          <div className="hero-right">
            {(() => {
              const imgSrc = heroIllustration;
              // log computed src for debugging in DevTools
              // eslint-disable-next-line no-console
              console.log('Contributors hero image src:', imgSrc);
              return (
                <div className="illustration-frame" data-img-src={imgSrc}>
                  <img
                    src={imgSrc}
                    alt="Fasal Saathi illustration"
                    className="hero-illustration"
                    loading="lazy"
                    onError={(e) => {
                      // hide broken image and mark frame so CSS can show fallback
                      e.currentTarget.style.display = 'none';
                      e.currentTarget.parentNode?.classList?.add('illustration-missing');
                      // eslint-disable-next-line no-console
                      console.warn('Hero illustration failed to load:', e.currentTarget.src);
                    }}
                    onLoad={(e) => {
                      const img = e.currentTarget;
                      img.parentNode?.classList?.add('illustration-loaded');
                      // eslint-disable-next-line no-console
                      console.log('Hero illustration loaded —', img.naturalWidth, 'x', img.naturalHeight, img.src);
                    }}
                  />
                </div>
              );
            })()}
          </div>
        </div>
      </div>

      {/* FILTER BUTTONS */}
      {/* TOOLBAR: search + sort + role filter */}
      <div className="contributors-toolbar">
        {errorMessage && (
          <div className="error-banner" role="alert" aria-live="polite">
            {errorMessage}
          </div>
        )}

        <div className="toolbar-row">
          <div className="search-box">
            <label htmlFor="contributor-search" className="visually-hidden">Search contributors</label>
            <FaSearch aria-hidden="true" />
            <input
              id="contributor-search"
              type="search"
              placeholder="Search by GitHub username"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              aria-label="Search contributors by username"
            />
          </div>

          {/* sort toggle removed per design */}
        </div>

        <div className="contributors-filter">
          <h3>Filter by Role:</h3>
          <div className="filter-buttons" role="tablist" aria-label="Contributor roles">
            {roles.map((role) => (
              <button
                key={role}
                className={`filter-btn ${filter === role ? "active" : ""}`}
                onClick={() => setFilter(role)}
                role="tab"
                aria-selected={filter === role}
              >
                {role}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* CONTRIBUTORS GRID */}
      <div className="contributors-grid">
        {loading ? (
          Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="skeleton-card" aria-hidden="true" />
          ))
        ) : filteredContributors.length > 0 ? (
          filteredContributors.map((contributor) => (
            <div
              key={contributor.id}
              className={`contributor-card ${contributor.isOwner ? 'founder-card' : ''}`}
            >
              {contributor.isOwner && (
                <div className="founder-badge">
                  <FaCrown /> Owner & Founder
                </div>
              )}
              <div className="card-image-container">
                <img
                  src={contributor.image}
                  alt={contributor.name}
                  className="contributor-image"
                />
              </div>

              <div className="card-content">
                 <h3><span className="notranslate">{contributor.name}</span></h3>
                <p className="role">{contributor.role}</p>

                {contributor.contributions && (
                  <p className="contributions">
                    <FaCode /> {contributor.contributions} contributions
                  </p>
                )}

                <div className="social-links">
                  {contributor.github && (
                    <a
                      href={contributor.github}
                      target="_blank"
                      rel="noopener noreferrer"
                      title="GitHub Profile"
                      className="social-icon"
                    >
                      <FaGithub />
                    </a>
                  )}
                  {contributor.linkedin && (
                    <a
                      href={contributor.linkedin}
                      target="_blank"
                      rel="noopener noreferrer"
                      title="LinkedIn Profile"
                      className="social-icon"
                    >
                      <FaLinkedin />
                    </a>
                  )}
                  {contributor.twitter && (
                    <a
                      href={contributor.twitter}
                      target="_blank"
                      rel="noopener noreferrer"
                      title="Twitter Profile"
                      className="social-icon"
                    >
                      <FaTwitter />
                    </a>
                  )}
                </div>
              </div>
            </div>
          ))
        ) : (
          <div className="no-contributors">
            No contributors found for this filter.
          </div>
        )}
      </div>

      {/* FOOTER CTA */}
      <div className="contributors-footer">
        <h2>Made with 💚 by farmers and developers</h2>
        <p>
          <span className="notranslate" translate="no">Fasal Saathi</span> is an open-source project dedicated to empowering
          farmers with AI-driven insights
        </p>
        <a
          href="https://github.com/Eshajha19/agri"
          target="_blank"
          rel="noopener noreferrer"
          className="btn btn-outline"
        >
          <span className="notranslate">View Repository on GitHub</span>
        </a>
      </div>
    </div>
  );
}