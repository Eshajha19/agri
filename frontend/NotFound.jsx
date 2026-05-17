import React from "react";
import { Link } from "react-router-dom";
import { FaHome, FaComments, FaLeaf, FaTachometerAlt } from "react-icons/fa";
import "./NotFound.css";

export default function NotFound() {
  return (
    <div className="not-found-container home">
      <div className="not-found-content">
        <h1 className="highlight">404</h1>
        <h2>Page Not Found</h2>
        <p>Sorry, the page you're looking for doesn't exist.</p>
        
        <div className="not-found-links">
          <Link to="/" className="not-found-link primary">
            <FaHome /> Back to Home
          </Link>
          
          <div className="not-found-suggestions">
            <h3>Popular Pages:</h3>
            <div className="suggestion-grid">
              <Link to="/advisor" className="not-found-link">
                <FaComments /> AI Advisor
              </Link>
              <Link to="/crop-guide" className="not-found-link">
                <FaLeaf /> Crop Guide
              </Link>
              <Link to="/dashboard" className="not-found-link">
                <FaTachometerAlt /> Dashboard
              </Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
