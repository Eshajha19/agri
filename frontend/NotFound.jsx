import React from "react";
import { Link } from "react-router-dom";
import "./NotFound.css";

export default function NotFound() {
  return (
    <div className="not-found-container home">
      <div className="not-found-content">
        <h1 className="highlight">404</h1>
        <h2>Page Not Found</h2>
        <p>Sorry, the page you're looking for doesn't exist.</p>
        <Link to="/" className="back-home-btn">
            Back to Home
        </Link>
      </div>
    </div>
  );
}