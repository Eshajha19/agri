import React, { useState, useEffect, useCallback, useMemo } from "react";
import { FaGithub, FaLinkedin, FaTwitter, FaCrown, FaCode, FaStar, FaSearch, FaSort, FaSortUp, FaSortDown, FaUserPlus, FaShieldAlt, FaGlobe, FaFilter, FaTimes, FaBook } from "react-icons/fa";
import "./Contributors.css";

/**
 * Configuration for the GitHub organization and repository.
 * Easily update these values to point to a different repo or org.
 */
const GITHUB_CONFIG = {
  owner: "Eshajha19",
  repo: "agri",
  perPage: 100,
  apiBase: "https://api.github.com",
};

/**
 * Helper function to safely fetch data from the GitHub API
 * with proper error handling.
 */
const fetchGitHubData = async (url) => {
  const response = await fetch(url, {
    headers: {
      Accept: "application/vnd.github.v3+json",
      "User-Agent": "FasalSaathi-Contributors",
    },
  });
  if (!response.ok) {
    throw new Error(`GitHub API error: ${response.status} ${response.statusText}`);
  }
  return response.json();
};

export default function Contributors() {
  const [contributors, setContributors] = useState([]);
  const [orgMembers, setOrgMembers] = useState([]);
  const [filter, setFilter] = useState("All");
  const [searchQuery, setSearchQuery] = useState("");
  const [sortBy, setSortBy] = useState("contributions");
  const [sortOrder, setSortOrder] = useState("desc");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [retryCount, setRetryCount] = useState(0);

  // Fetch contributors from GitHub API with retry logic
  const fetchContributors = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [contribsData, membersData] = await Promise.all([
        fetchGitHubData(
          `${GITHUB_CONFIG.apiBase}/repos/${GITHUB_CONFIG.owner}/${GITHUB_CONFIG.repo}/contributors?per_page=${GITHUB_CONFIG.perPage}&anon=true`
        ),
        fetchGitHubData(
          `${GITHUB_CONFIG.apiBase}/orgs/${GITHUB_CONFIG.owner}/members?per_page=100`
        ),
      ]);

      const orgLogins = new Set(membersData.map((m) => m.login.toLowerCase()));

      const mappedContributors = contribsData.map((contributor) => {
        const isOrgMember = orgLogins.has((contributor.login || "").toLowerCase());
        let role = "Contributor";
        if (contributor.login.toLowerCase() === GITHUB_CONFIG.owner.toLowerCase()) {
          role = "Owner & Founder";
        } else if (isOrgMember && contributor.contributions >= 10) {
          role = "Core Team";
        } else if (isOrgMember) {
          role = "Team Member";
        }

        return {
          id: contributor.id,
          name: contributor.login || "Unknown",
          role,
          image: contributor.avatar_url || "",
          github: contributor.html_url || "",
          contributions: contributor.contributions || 0,
          isOwner: contributor.login.toLowerCase() === GITHUB_CONFIG.owner.toLowerCase(),
          isOrgMember,
        };
      });

      setContributors(mappedContributors);
    } catch (err) {
      console.error("Error fetching GitHub contributors:", err);
      setError(err.message);
      if (retryCount < 2) {
        setTimeout(() => setRetryCount((c) => c + 1), 2000);
      }
    } finally {
      setLoading(false);
    }
  }, [retryCount]);

  useEffect(() => {
    fetchContributors();
  }, [fetchContributors]);

  const roleDefinitions = useMemo(
    () => [
      { key: "All", label: "All Contributors", icon: FaUserPlus },
      { key: "Owner & Founder", label: "Owners & Founders", icon: FaCrown },
      { key: "Core Team", label: "Core Team", icon: FaShieldAlt },
      { key: "Team Member", label: "Team Members", icon: FaGlobe },
      { key: "Contributor", label: "Contributors", icon: FaCode },
    ],
    []
  );

  const availableRoles = useMemo(() => {
    const roles = contributors.map((c) => c.role);
    return [...new Set(roles)];
  }, [contributors]);

  const filteredContributors = useMemo(() => {
    let result = contributors;
    if (filter !== "All") {
      result = result.filter((c) => c.role === filter);
    }
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase().trim();
      result = result.filter(
        (c) =>
          c.name.toLowerCase().includes(query) ||
          c.role.toLowerCase().includes(query)
      );
    }
    result = [...result].sort((a, b) => {
      let comparison = 0;
      switch (sortBy) {
        case "name":
          comparison = a.name.localeCompare(b.name);
          break;
        case "contributions":
          comparison = a.contributions - b.contributions;
          break;
        default:
          comparison = a.contributions - b.contributions;
      }
      return sortOrder === "asc" ? comparison : -comparison;
    });
    return result;
  }, [contributors, filter, searchQuery, sortBy, sortOrder]);

  const stats = useMemo(() => {
    const total = contributors.length;
    const owners = contributors.filter((c) => c.isOwner).length;
    const coreTeam = contributors.filter((c) => c.role === "Core Team" || c.role === "Team Member").length;
    const totalContributions = contributors.reduce((sum, c) => sum + c.contributions, 0);
    return { total, owners, coreTeam, totalContributions };
  }, [contributors]);

  const handleSort = (field) => {
    if (sortBy === field) {
      setSortOrder((prev) => (prev === "asc" ? "desc" : "asc"));
    } else {
      setSortBy(field);
      setSortOrder("desc");
    }
  };

  const SortIndicator = ({ field }) => {
    if (sortBy !== field) return null;
    return sortOrder === "asc" ? <FaSortUp size={12} /> : <FaSortDown size={12} />;
  };

  if (loading && contributors.length === 0) {
    return (
      <div className="contributors-page">
        <div className="loading-container">
          <div className="loading-spinner" />
          <p>Loading contributors...</p>
          <p className="loading-subtext">Fetching data from GitHub</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="contributors-page">
        <div className="error-container">
          <div className="error-icon">⚠️</div>
          <h2>Failed to Load Contributors</h2>
          <p>{error}</p>
          <button
            className="btn btn-primary"
            onClick={() => setRetryCount((c) => c + 1)}
            disabled={loading}
          >
            {loading ? "Retrying..." : "Try Again"}
          </button>
          {retryCount >= 2 && (
            <p className="error-hint">
              This may be a temporary GitHub API issue. Please try again later.
            </p>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="contributors-page">
      <div className="contributors-hero">
        <div className="hero-content-wrapper">
          <div className="hero-text">
            <h1 className="hero-heading">🌟 Meet Our Contributors</h1>
            <p className="hero-subtitle">
              <span className="notranslate" translate="no">Fasal Saathi</span> is built by a passionate community of
              developers, designers, and farmers dedicated to revolutionizing agriculture through open-source collaboration.
            </p>
          </div>
          <div className="hero-visual">
            <div className="hero-icon-ring">
              <FaCode className="hero-main-icon" />
            </div>
          </div>
        </div>
      </div>

      <section className="contributors-stats" aria-label="Contributor Statistics">
        <div className="stat-card stat-card--total">
          <div className="stat-icon">👥</div>
          <div className="stat-info">
            <h3>{stats.total}</h3>
            <p>Total Contributors</p>
          </div>
        </div>
        <div className="stat-card stat-card--core">
          <div className="stat-icon">🚀</div>
          <div className="stat-info">
            <h3>{stats.coreTeam}</h3>
            <p>Core Team Members</p>
          </div>
        </div>
        <div className="stat-card stat-card--commits">
          <div className="stat-icon">💻</div>
          <div className="stat-info">
            <h3>{stats.totalContributions.toLocaleString()}</h3>
            <p>Total Contributions</p>
          </div>
        </div>
        <div className="stat-card stat-card--open">
          <div className="stat-icon">💚</div>
          <div className="stat-info">
            <h3>Open</h3>
            <p>Source Project</p>
          </div>
        </div>
      </section>

      <section className="contribute-cta" aria-label="Contribute Call to Action">
        <div className="cta-content">
          <h2>Want to Contribute?</h2>
          <p>
            Join our growing community of developers making a difference in agriculture.
            Whether it's code, documentation, bug reports, or feature ideas — every contribution counts.
          </p>
          <div className="cta-buttons">
            <a
              href={`https://github.com/${GITHUB_CONFIG.owner}/${GITHUB_CONFIG.repo}`}
              target="_blank"
              rel="noopener noreferrer"
              className="btn btn-primary"
              aria-label="Star the repository on GitHub"
            >
              <FaGithub /> Star on GitHub
            </a>
            <a
              href={`https://github.com/${GITHUB_CONFIG.owner}/${GITHUB_CONFIG.repo}/blob/main/CONTRIBUTING.md`}
              target="_blank"
              rel="noopener noreferrer"
              className="btn btn-secondary"
              aria-label="View the contributing guide"
            >
              <FaBook /> Contributing Guide
            </a>
            <a
              href={`https://github.com/${GITHUB_CONFIG.owner}/${GITHUB_CONFIG.repo}/issues/new?labels=good+first+issue`}
              target="_blank"
              rel="noopener noreferrer"
              className="btn btn-outline"
              aria-label="View good first issues"
            >
              <FaStar /> Good First Issues
            </a>
          </div>
        </div>
      </section>

      <section className="contributors-section" aria-label="Contributors List">
        <div className="contributors-toolbar">
          <div className="search-bar-wrapper">
            <FaSearch className="search-icon" />
            <input
              type="text"
              className="search-bar"
              placeholder="Search by name or role..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              aria-label="Search contributors by name or role"
            />
            {searchQuery && (
              <button
                className="search-clear-btn"
                onClick={() => setSearchQuery("")}
                aria-label="Clear search"
                title="Clear search"
              >
                <FaTimes size={14} />
              </button>
            )}
          </div>

          <div className="toolbar-actions">
            <div className="filter-wrapper">
              <FaFilter className="filter-icon" />
              <div className="filter-buttons">
                {roleDefinitions
                  .filter((r) => availableRoles.includes(r.key) || r.key === "All")
                  .map((role) => (
                    <button
                      key={role.key}
                      className={`filter-btn ${filter === role.key ? "active" : ""}`}
                      onClick={() => setFilter(role.key)}
                      aria-pressed={filter === role.key}
                    >
                      {role.icon && <role.icon size={14} />} {role.label}
                    </button>
                  ))}
              </div>
            </div>

            <div className="sort-wrapper">
              <label className="sort-label" htmlFor="sort-select">
                <FaSort size={14} /> Sort by:
              </label>
              <select
                id="sort-select"
                className="sort-select"
                value={sortBy}
                onChange={(e) => handleSort(e.target.value)}
                aria-label="Sort contributors"
              >
                <option value="contributions">Contributions</option>
                <option value="name">Name</option>
              </select>
              <button
                className="sort-order-btn"
                onClick={() => setSortOrder((prev) => (prev === "asc" ? "desc" : "asc"))}
                aria-label={`Sort ${sortOrder === "asc" ? "descending" : "ascending"}`}
                title={`Sort ${sortOrder === "asc" ? "descending" : "ascending"}`}
              >
                {sortOrder === "asc" ? <FaSortUp size={16} /> : <FaSortDown size={16} />}
              </button>
            </div>
          </div>
        </div>

        <div className="results-info">
          <span>
            Showing <strong>{filteredContributors.length}</strong> of{" "}
            <strong>{contributors.length}</strong> contributors
          </span>
        </div>

        {filteredContributors.length > 0 ? (
          <div className="contributors-table-wrapper">
            <table className="contributors-table" role="table" aria-label="Contributors list">
              <thead>
                <tr>
                  <th scope="col" className="col-avatar">Avatar</th>
                  <th scope="col" className="col-name">
                    <button className="sortable-header" onClick={() => handleSort("name")}>
                      Name <SortIndicator field="name" />
                    </button>
                  </th>
                  <th scope="col" className="col-role">Role</th>
                  <th scope="col" className="col-contributions">
                    <button className="sortable-header" onClick={() => handleSort("contributions")}>
                      Contributions <SortIndicator field="contributions" />
                    </button>
                  </th>
                  <th scope="col" className="col-links">Links</th>
                </tr>
              </thead>
              <tbody>
                {filteredContributors.map((contributor) => (
                  <tr
                    key={contributor.id}
                    className={`contributor-row ${contributor.isOwner ? "founder-row" : ""}`}
                  >
                    <td className="col-avatar">
                      <div className="avatar-wrapper">
                        {contributor.image ? (
                          <img
                            src={contributor.image}
                            alt={contributor.name}
                            className="contributor-avatar"
                            loading="lazy"
                          />
                        ) : (
                          <div className="avatar-placeholder">
                            {contributor.name.charAt(0).toUpperCase()}
                          </div>
                        )}
                        {contributor.isOwner && (
                          <span className="badge badge--owner" title="Owner & Founder">
                            <FaCrown />
                          </span>
                        )}
                        {contributor.isOrgMember && !contributor.isOwner && (
                          <span className="badge badge--member" title="Organization Member">
                            <FaShieldAlt />
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="col-name">
                      <div className="name-block">
                        <span className="contributor-name">
                          <span className="notranslate">{contributor.name}</span>
                        </span>
                        <span className={`role-badge role--${contributor.role.replace(/\s+/g, "-").toLowerCase()}`}>
                          {contributor.role}
                        </span>
                      </div>
                    </td>
                    <td className="col-role">
                      <span className="role-text">{contributor.role}</span>
                    </td>
                    <td className="col-contributions">
                      <div className="contributions-cell">
                        <FaCode className="contributions-icon" />
                        <span className="contributions-count">
                          {contributor.contributions.toLocaleString()}
                        </span>
                        <div className="contributions-bar">
                          <div
                            className="contributions-bar-fill"
                            style={{
                              width: `${Math.min(
                                (contributor.contributions / (contributors[0]?.contributions || 1)) * 100,
                                100
                              )}%`,
                            }}
                          />
                        </div>
                      </div>
                    </td>
                    <td className="col-links">
                      <div className="social-links">
                        {contributor.github && (
                          <a
                            href={contributor.github}
                            target="_blank"
                            rel="noopener noreferrer"
                            title={`GitHub: ${contributor.name}`}
                            className="social-icon social-icon--github"
                            aria-label={`GitHub profile of ${contributor.name}`}
                          >
                            <FaGithub />
                          </a>
                        )}
                        {contributor.linkedin && (
                          <a
                            href={contributor.linkedin}
                            target="_blank"
                            rel="noopener noreferrer"
                            title={`LinkedIn: ${contributor.name}`}
                            className="social-icon social-icon--linkedin"
                            aria-label={`LinkedIn profile of ${contributor.name}`}
                          >
                            <FaLinkedin />
                          </a>
                        )}
                        {contributor.twitter && (
                          <a
                            href={contributor.twitter}
                            target="_blank"
                            rel="noopener noreferrer"
                            title={`Twitter: ${contributor.name}`}
                            className="social-icon social-icon--twitter"
                            aria-label={`Twitter profile of ${contributor.name}`}
                          >
                            <FaTwitter />
                          </a>
                        )}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="no-contributors">
            <div className="no-contributors-icon">🔍</div>
            <h3>No contributors found</h3>
            <p>
              {searchQuery
                ? `No results match "${searchQuery}". Try a different search term.`
                : `No contributors match the selected filter "${filter}".`}
            </p>
            <button className="btn btn-secondary" onClick={() => { setSearchQuery(""); setFilter("All"); }}>
              Clear Filters
            </button>
          </div>
        )}
      </section>

      <section className="top-contributors-section" aria-label="Top Contributors">
        <h2 className="section-heading">🏆 Top Contributors</h2>
        <div className="top-contributors-grid">
          {contributors
            .filter((c) => c.contributions >= 5)
            .sort((a, b) => b.contributions - a.contributions)
            .slice(0, 6)
            .map((contributor, idx) => (
              <div
                key={contributor.id}
                className={`top-contributor-card ${contributor.isOwner ? "founder-card" : ""}`}
              >
                <div className="rank-badge">#{idx + 1}</div>
                <img
                  src={contributor.image}
                  alt={contributor.name}
                  className="top-avatar"
                  loading="lazy"
                />
                <div className="top-card-info">
                  <h4 className="top-card-name">
                    <span className="notranslate">{contributor.name}</span>
                  </h4>
                  <p className="top-card-role">{contributor.role}</p>
                  <p className="top-card-contributions">
                    <FaCode /> {contributor.contributions} contributions
                  </p>
                </div>
              </div>
            ))}
        </div>
      </section>

      <footer className="contributors-footer" role="contentinfo">
        <div className="footer-content">
          <h2>Made with 💚 by farmers and developers</h2>
          <p>
            <span className="notranslate" translate="no">Fasal Saathi</span> is an open-source project dedicated to
            empowering farmers with AI-driven insights. We believe in the power of community and open collaboration.
          </p>
          <div className="footer-buttons">
            <a
              href={`https://github.com/${GITHUB_CONFIG.owner}/${GITHUB_CONFIG.repo}`}
              target="_blank"
              rel="noopener noreferrer"
              className="btn btn-outline"
              aria-label="View the GitHub repository"
            >
              <FaGithub /> View Repository
            </a>
            <a
              href={`https://github.com/${GITHUB_CONFIG.owner}/${GITHUB_CONFIG.repo}/blob/main/CONTRIBUTING.md`}
              target="_blank"
              rel="noopener noreferrer"
              className="btn btn-primary"
              aria-label="Read the contributing guide"
            >
              <FaBook /> Contributing Guide
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}
