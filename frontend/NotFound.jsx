import React from "react";
import { Link } from "react-router-dom";
import { FaHome, FaComments, FaLeaf, FaTachometerAlt } from "react-icons/fa";
import "./NotFound.css";

export default function NotFound() {
  return (
    <div className="not-found-container home" role="region" aria-labelledby="not-found-title">
      <div className="not-found-content">
        <h1 className="highlight" id="not-found-title">404</h1>
        <h2>Page Not Found</h2>
        <p>Sorry, the page you're looking for doesn't exist.</p>
        
        <div className="not-found-links">
          <Link to="/" className="not-found-link primary" aria-label="Go back to the home page">
            <FaHome aria-hidden="true" /> Back to Home
          </Link>
          
          <nav className="not-found-suggestions" aria-label="Suggested pages">
            <h3>Popular Pages:</h3>
            <div className="suggestion-grid">
              <Link to="/advisor" className="not-found-link" aria-label="Go to AI Advisor">
                <FaComments aria-hidden="true" /> AI Advisor
              </Link>
              <Link to="/crop-guide" className="not-found-link" aria-label="Go to Crop Guide">
                <FaLeaf aria-hidden="true" /> Crop Guide
              </Link>
              <Link to="/dashboard" className="not-found-link" aria-label="Go to Dashboard">
                <FaTachometerAlt aria-hidden="true" /> Dashboard
              </Link>
            </div>
          </nav>
        </div>
      </div>
    </div>
  );
}
