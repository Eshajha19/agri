import React, { useState, useEffect } from "react";
import { FaGithub, FaLinkedin, FaTwitter } from "react-icons/fa";
import "./Contributors.css";

const CONTRIBUTORS_DATA = [
  {
    id: 1,
    name: "John Developer",
    role: "Lead Developer",
    image: "https://api.github.com/users/octocat/avatar_url",
    github: "https://github.com/octocat",
    linkedin: "https://linkedin.com",
    twitter: "https://twitter.com",
  },
  {
    id: 2,
    name: "Sarah Designer",
    role: "UI/UX Designer",
    image: "https://api.dicebear.com/7.x/avataaars/svg?seed=Sarah",
    github: "https://github.com",
    linkedin: "https://linkedin.com",
    twitter: "https://twitter.com",
  },
  {
    id: 3,
    name: "Mike Backend",
    role: "Backend Developer",
    image: "https://api.dicebear.com/7.x/avataaars/svg?seed=Mike",
    github: "https://github.com",
    linkedin: "https://linkedin.com",
  },
  {
    id: 4,
    name: "Emma ML Engineer",
    role: "ML/AI Engineer",
    image: "https://api.dicebear.com/7.x/avataaars/svg?seed=Emma",
    github: "https://github.com",
    linkedin: "https://linkedin.com",
    twitter: "https://twitter.com",
  },
  {
    id: 5,
    name: "Alex DevOps",
    role: "DevOps & Infrastructure",
    image: "https://api.dicebear.com/7.x/avataaars/svg?seed=Alex",
    github: "https://github.com",
    linkedin: "https://linkedin.com",
  },
  {
    id: 6,
    name: "Lisa QA",
    role: "Quality Assurance",
    image: "https://api.dicebear.com/7.x/avataaars/svg?seed=Lisa",
    github: "https://github.com",
    linkedin: "https://linkedin.com",
    twitter: "https://twitter.com",
  },
];

export default function Contributors() {
  const [contributors, setContributors] = useState(CONTRIBUTORS_DATA);
  const [filter, setFilter] = useState("All");
  const [loading, setLoading] = useState(false);

  // Optional: Fetch contributors from GitHub API
  const fetchGitHubContributors = async () => {
    setLoading(true);
    try {
      const response = await fetch(
        "https://api.github.com/repos/YOUR_ORG/YOUR_REPO/contributors?per_page=100"
      );
      if (response.ok) {
        const data = await response.json();
        const mappedContributors = data.map((contributor) => ({
          id: contributor.id,
          name: contributor.login,
          role: "Contributor",
          image: contributor.avatar_url,
          github: contributor.html_url,
          contributions: contributor.contributions,
        }));
        setContributors(mappedContributors);
      }
    } catch (error) {
      console.error("Error fetching GitHub contributors:", error);
      setContributors(CONTRIBUTORS_DATA);
    } finally {
      setLoading(false);
    }
  };

  const roles = [
    "All",
    ...new Set(CONTRIBUTORS_DATA.map((c) => c.role)),
  ];

  const filteredContributors = contributors.filter((contributor) => {
    if (filter === "All") return true;
    return contributor.role === filter;
  });

  return (
    <div className="contributors-page">
      {/* HERO SECTION */}
      <div className="contributors-hero">
        <h1>🌟 Meet Our Contributors</h1>
        <p>
          Fasal Saathi is built by a passionate community of developers, 
          designers, and farmers dedicated to revolutionizing agriculture
        </p>
      </div>

      {/* STATS SECTION */}
      <div className="contributors-stats">
        <div className="stat-card">
          <h3>{contributors.length}+</h3>
          <p>Contributors</p>
        </div>
        <div className="stat-card">
          <h3>🚀</h3>
          <p>Active Development</p>
        </div>
        <div className="stat-card">
          <h3>💚</h3>
          <p>Open Source</p>
        </div>
      </div>

      {/* CTA SECTION */}
      <div className="contribute-cta">
        <h2>Want to Contribute?</h2>
        <p>Join our growing community of developers making a difference in agriculture</p>
        <div className="cta-buttons">
          <a
            href="https://github.com/YOUR_ORG/YOUR_REPO"
            target="_blank"
            rel="noopener noreferrer"
            className="btn btn-primary"
          >
            <FaGithub /> Star on GitHub
          </a>
          <a
            href="https://github.com/YOUR_ORG/YOUR_REPO/blob/main/CONTRIBUTING.md"
            target="_blank"
            rel="noopener noreferrer"
            className="btn btn-secondary"
          >
            View Contributing Guide
          </a>
        </div>
      </div>

      {/* FILTER BUTTONS */}
      <div className="contributors-filter">
        <h3>Filter by Role:</h3>
        <div className="filter-buttons">
          {roles.map((role) => (
            <button
              key={role}
              className={`filter-btn ${filter === role ? "active" : ""}`}
              onClick={() => setFilter(role)}
            >
              {role}
            </button>
          ))}
        </div>
      </div>

      {/* CONTRIBUTORS GRID */}
      <div className="contributors-grid">
        {loading ? (
          <div className="loading">Loading contributors...</div>
        ) : filteredContributors.length > 0 ? (
          filteredContributors.map((contributor) => (
            <div key={contributor.id} className="contributor-card">
              <div className="card-image-container">
                <img
                  src={contributor.image}
                  alt={contributor.name}
                  className="contributor-image"
                />
              </div>

              <div className="card-content">
                <h3>{contributor.name}</h3>
                <p className="role">{contributor.role}</p>

                {contributor.contributions && (
                  <p className="contributions">
                    {contributor.contributions} contributions
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
          Fasal Saathi is an open-source project dedicated to empowering 
          farmers with AI-driven insights
        </p>
        <a
          href="https://github.com/YOUR_ORG/YOUR_REPO"
          target="_blank"
          rel="noopener noreferrer"
          className="btn btn-outline"
        >
          View Repository on GitHub
        </a>
      </div>
    </div>
  );
}
