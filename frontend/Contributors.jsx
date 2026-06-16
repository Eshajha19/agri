import React, { useState, useEffect, useMemo } from "react";
import { FaGithub, FaLinkedin, FaTwitter, FaCrown, FaCode, FaSearch } from "react-icons/fa";
import "./Contributors.css";

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

export default function Contributors() {
  const [contributors, setContributors] = useState([]);
  const [filter, setFilter] = useState("All");
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchContributors = async () => {
      setLoading(true);

      // Prefer a static local JSON first so the page works even when GitHub rate limits
      // or when external network calls are blocked.
      // This file is generated/maintained in the repo (see: frontend/contributors.static.json)
      try {
        const staticRes = await fetch("/contributors.static.json", {
          headers: { "Accept": "application/json" },
          cache: "no-store",
        });

        if (staticRes.ok) {
          const staticData = await staticRes.json();
          if (Array.isArray(staticData) && staticData.length > 0) {
            setContributors(staticData);
            return;
          }
        }
      } catch {
        // ignore and fall back to GitHub API
      }

      try {
        const response = await fetch(
          "https://api.github.com/repos/Eshajha19/agri/contributors?per_page=100",
          {
            headers: {
              "Accept": "application/vnd.github.v3+json",
            },
          }
        );

        if (!response.ok) {
          setContributors(fallbackContributors);
          return;
        }

        const data = await response.json();
        const mappedContributors = (Array.isArray(data) ? data : []).map((contributor) => ({
          id: contributor.id,
          name: contributor.login,
          role: contributor.login?.toLowerCase() === "eshajha19" ? "Owner & Founder" : "Contributor",
          image: contributor.avatar_url,
          github: contributor.html_url,
          contributions: contributor.contributions,
          isOwner: contributor.login?.toLowerCase() === "eshajha19",
        }));

        setContributors(mappedContributors.length > 0 ? mappedContributors : fallbackContributors);
      } catch (error) {
        // eslint-disable-next-line no-console
        console.error("Error fetching GitHub contributors:", error);
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

    if (filter !== "All") {
      list = list.filter((c) => c.role === filter);
    }

    if (search && search.trim().length > 0) {
      const q = search.toLowerCase();
      list = list.filter((c) => c.name.toLowerCase().includes(q));
    }

    if (!search || search.trim().length === 0) {
      list.sort((a, b) => (b.contributions || 0) - (a.contributions || 0));
    }

    return list;
  }, [contributors, filter, search]);

  return (
    <div className="contributors-page">
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
            <div className="illustration-frame">
              <img
                src="/hero-illustration.png"
                alt="Fasal Saathi illustration"
                className="hero-illustration"
                loading="lazy"
                onError={(e) => {
                  e.currentTarget.style.display = 'none';
                  e.currentTarget.parentNode?.classList?.add('illustration-missing');
                }}
              />
            </div>
          </div>
        </div>
      </div>

      <div className="contributors-toolbar">
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