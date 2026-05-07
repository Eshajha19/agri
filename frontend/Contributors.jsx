import React, { useState, useEffect } from "react";
import { FaGithub, FaLinkedin, FaTwitter, FaCrown, FaCode, FaStar } from "react-icons/fa";
import "./Contributors.css";

export default function Contributors() {
  const [contributors, setContributors] = useState([]);
  const [filter, setFilter] = useState("All");
  const [loading, setLoading] = useState(true);

  // Fetch contributors from GitHub API
  useEffect(() => {
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
         }
      } catch (error) {
        console.error("Error fetching GitHub contributors:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchContributors();
  }, []);

  const roles = ["All", ...new Set(contributors.map((c) => c.role))];

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
          <span className="notranslate" translate="no">Fasal Saathi</span> is built by a passionate community of developers,
          designers, and farmers dedicated to revolutionizing agriculture
        </p>
      </div>

       {/* STATS SECTION */}
       <div className="contributors-stats">
<<<<<<< HEAD
         <div className="stat-card">
           <h3>{contributors.length}</h3>
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
=======

  <div className="stat-card">
    <div className="stat-icon">👥</div>
    <h3>{contributors.length}</h3>
    <p>Contributors</p>
    <span className="stat-subtext">Active developers</span>
  </div>

  <div className="stat-card">
    <div className="stat-icon">🚀</div>
    <h3>Active</h3>
    <p>Development</p>
    <span className="stat-subtext">Continuous updates</span>
  </div>

  <div className="stat-card">
    <div className="stat-icon">💚</div>
    <h3>Open</h3>
    <p>Source Project</p>
    <span className="stat-subtext">Community driven</span>
  </div>

</div>
>>>>>>> upstream/main

      {/* CTA SECTION */}
      <div className="contribute-cta">
        <h2>Want to Contribute?</h2>
        <p>Join our growing community of developers making a difference in agriculture</p>
        <div className="cta-buttons">
          <a
            href="https://github.com/Eshajha19/agri"
            target="_blank"
            rel="noopener noreferrer"
            className="btn btn-primary"
          >
            <FaGithub /> Star on GitHub
          </a>
          <a
            href="https://github.com/Eshajha19/agri/blob/main/CONTRIBUTING.md"
            target="_blank"
            rel="noopener noreferrer"
            className="btn btn-secondary"
          >
            <span className="notranslate">View Contributing Guide</span>
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

                 {contributor.isOwner && (
                   <p className="founder-description">
                     <FaStar /> Visionary leader driving <span className="notranslate" translate="no">Fasal Saathi</span>'s mission to revolutionize agriculture
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
