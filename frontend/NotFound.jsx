import React from "react";
import { Link, useNavigate } from "react-router-dom";
import { FaHome, FaArrowLeft, FaLeaf, FaSeedling } from "react-icons/fa";
import "./NotFound.css";

export default function NotFound() {
  const navigate = useNavigate();

  return (
    <div className="notfound-page" role="main" aria-labelledby="notfound-title">
      <div className="notfound-container">
        <div className="notfound-illustration" aria-hidden="true">
          <div className="notfound-code">
            <span className="digit">4</span>
            <span className="leaf-zero">
              <FaLeaf />
            </span>
            <span className="digit">4</span>
          </div>
          <FaSeedling className="notfound-sprout" aria-hidden="true" />
        </div>

        <h1 id="notfound-title" className="notfound-title">
          This field looks empty
        </h1>
        <p className="notfound-subtitle">
          The page you're looking for doesn't exist or may have been moved.
          Let's get you back to growing.
        </p>

        <div className="notfound-actions">
          <button
            type="button"
            onClick={() => navigate(-1)}
            className="notfound-btn notfound-btn-secondary"
            aria-label="Go back to the previous page"
          >
            <FaArrowLeft aria-hidden="true" /> Go Back
          </button>
          <Link
            to="/"
            className="notfound-btn notfound-btn-primary"
            aria-label="Return to the home page"
          >
            <FaHome aria-hidden="true" /> Back to Home
          </Link>
        </div>

        <div className="notfound-suggestions">
          <p className="notfound-suggestions-label">Or explore:</p>
          <ul className="notfound-links">
            <li><Link to="/crop-guide">Crop Guide</Link></li>
            <li><Link to="/dashboard">Dashboard</Link></li>
            <li><Link to="/resources">Resources</Link></li>
            <li><Link to="/contact">Contact Us</Link></li>
          </ul>
        </div>
      </div>
    </div>
  );
}
