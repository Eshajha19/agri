import React, { useState, useEffect, useMemo } from "react";
import {
  FaGithub,
  FaLinkedin,
  FaTwitter,
  FaCrown,
  FaCode,
  FaSearch,
} from "react-icons/fa";
import { useTranslation } from "react-i18next";
import "./Contributors.css";

export default function Contributors() {
  const [contributors, setContributors] = useState([]);
  const [filter, setFilter] = useState("All");
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(true);

  const { t } = useTranslation();

  useEffect(() => {
    const loadContributors = async () => {
      setLoading(true);

      try {
        const response = await fetch("/contributors.static.json", {
          headers: {
            Accept: "application/json",
          },
        });

        if (!response.ok) {
          throw new Error("Failed to load contributors");
        }

        const data = await response.json();

        if (Array.isArray(data)) {
          const formattedData = data.map((user, index) => ({
            id: user.id || index + 1,
            name: user.name || user.login,
            role:
              user.role ||
              (user.isOwner
                ? "Owner & Founder"
                : "Core Contributor"),
            image: user.image || user.avatar_url,
            github:
              user.github ||
              user.html_url ||
              `https://github.com/${user.login}`,
            linkedin: user.linkedin || "",
            twitter: user.twitter || "",
            contributions: user.contributions || 0,
            isOwner:
              user.isOwner ||
              user.login?.toLowerCase() === "eshajha19",
          }));

          setContributors(formattedData);
        } else {
          setContributors([]);
        }
      } catch (error) {
        console.error("Error loading contributors:", error);
        setContributors([]);
      } finally {
        setLoading(false);
      }
    };

    loadContributors();
  }, []);

  const roles = useMemo(
    () => ["All", ...new Set(contributors.map((c) => c.role))],
    [contributors]
  );

  const filteredContributors = useMemo(() => {
    let list = [...contributors];

    if (filter !== "All") {
      list = list.filter((c) => c.role === filter);
    }

    if (search.trim()) {
      const query = search.toLowerCase();

      list = list.filter(
        (c) =>
          c.name.toLowerCase().includes(query) ||
          c.role.toLowerCase().includes(query)
      );
    }

    return list.sort(
      (a, b) => (b.contributions || 0) - (a.contributions || 0)
    );
  }, [contributors, filter, search]);

  return (
    <div className="contributors-page">
      {/* HERO */}
      <section className="contributors-hero enhanced-hero">
        <div className="hero-inner">
          <div className="hero-left">
            <h1>
              {t(
                "Join the Fasal Saathi Community",
                "Join the Fasal Saathi Community"
              )}
            </h1>

            <p className="hero-animate">
              <span className="notranslate" translate="no">
                Fasal Saathi
              </span>{" "}
              is an open, farmer-first platform built by developers,
              agronomists, designers, and farmers. We welcome contributors of
              every skill level. Start small, learn fast, and see your work
              help real farms.
            </p>

            <div className="hero-badges">
              <span className="badge badge-pulse">
                Beginner-friendly
              </span>
              <span className="badge">Mentored PRs</span>
              <span className="badge">Field-tested</span>
            </div>

            <div className="cta-buttons hero-cta">
              <a
                href="https://github.com/Eshajha19/agri/issues"
                target="_blank"
                rel="noopener noreferrer"
                className="btn btn-primary btn-cta"
              >
                Explore Issues
              </a>

              <a
                href="https://github.com/Eshajha19/agri/blob/main/CONTRIBUTING.md"
                target="_blank"
                rel="noopener noreferrer"
                className="btn btn-outline btn-cta"
              >
                Get Started Guide
              </a>
            </div>

            <ul className="how-to">
              <li>🟢 Pick a "good first issue"</li>
              <li>🛠️ Open a PR and receive mentorship</li>
              <li>🌾 Help farmers through open source</li>
            </ul>
          </div>

          <div className="hero-right">
            <div className="illustration-frame">
              <img
                src="/hero-illustration.png"
                alt="Fasal Saathi Community"
                className="hero-illustration"
                loading="lazy"
                onError={(e) => {
                  e.currentTarget.style.display = "none";
                  e.currentTarget.parentNode?.classList?.add(
                    "illustration-missing"
                  );
                }}
              />
            </div>
          </div>
        </div>
      </section>

      {/* TOOLBAR */}
      <section className="contributors-toolbar">
        <div className="toolbar-row">
          <div className="search-box">
            <label
              htmlFor="contributor-search"
              className="visually-hidden"
            >
              Search contributors
            </label>

            <FaSearch aria-hidden="true" />

            <input
              id="contributor-search"
              type="search"
              placeholder="Search contributors..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              aria-label="Search contributors"
            />
          </div>
        </div>

        <div className="contributors-filter">
          <h3>Filter by Role</h3>

          <div
            className="filter-buttons"
            role="tablist"
            aria-label="Contributor roles"
          >
            {roles.map((role) => (
              <button
                key={role}
                className={`filter-btn ${
                  filter === role ? "active" : ""
                }`}
                onClick={() => setFilter(role)}
                role="tab"
                aria-selected={filter === role}
              >
                {role}
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* CONTRIBUTORS */}
      <section className="contributors-grid">
        {loading ? (
          Array.from({ length: 8 }).map((_, index) => (
            <div
              key={index}
              className="skeleton-card"
              aria-hidden="true"
            />
          ))
        ) : filteredContributors.length > 0 ? (
          filteredContributors.map((contributor) => (
            <article
              key={contributor.id}
              className={`contributor-card ${
                contributor.isOwner ? "founder-card" : ""
              }`}
            >
              {contributor.isOwner && (
                <div className="founder-badge">
                  <FaCrown />
                  Owner & Founder
                </div>
              )}

              <div className="card-image-container">
                <img
                  src={contributor.image}
                  alt={contributor.name}
                  className="contributor-image"
                  loading="lazy"
                />
              </div>

              <div className="card-content">
                <h3>
                  <span className="notranslate">
                    {contributor.name}
                  </span>
                </h3>

                <p className="role">{contributor.role}</p>

                <p className="contributions">
                  <FaCode />
                  {contributor.contributions} contributions
                </p>

                <div className="social-links">
                  {contributor.github && (
                    <a
                      href={contributor.github}
                      target="_blank"
                      rel="noopener noreferrer"
                      title="GitHub"
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
                      title="LinkedIn"
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
                      title="Twitter"
                      className="social-icon"
                    >
                      <FaTwitter />
                    </a>
                  )}
                </div>
              </div>
            </article>
          ))
        ) : (
          <div className="no-contributors">
            No contributors found.
          </div>
        )}
      </section>

      {/* FOOTER */}
      <section className="contributors-footer">
        <h2>Made with 💚 by farmers and developers</h2>

        <p>
          <span className="notranslate" translate="no">
            Fasal Saathi
          </span>{" "}
          is an open-source project dedicated to empowering farmers with
          AI-driven insights.
        </p>

        <a
          href="https://github.com/Eshajha19/agri"
          target="_blank"
          rel="noopener noreferrer"
          className="btn btn-outline"
        >
          <span className="notranslate">
            View Repository on GitHub
          </span>
        </a>
      </section>
    </div>
  );
}
