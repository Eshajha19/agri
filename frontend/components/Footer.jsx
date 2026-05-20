import React from "react";
import { Link } from "react-router-dom";
import { FaSeedling, FaPhoneAlt, FaGlobe } from "react-icons/fa";
import "./Footer.css";

const Footer = () => {
  return (
    <footer className="global-footer" role="contentinfo">
        <div className="footer-content">
          <div className="footer-grid">
            <div className="footer-section">
              <div className="footer-brand">
                <FaSeedling className="footer-logo" aria-hidden="true" />
                <span className="notranslate">Fasal Saathi</span>
              </div>
              <p className="footer-description">
                AI-powered agricultural advisor helping farmers with crop planning,
                weather insights, irrigation, and yield optimization.
              </p>
              <div className="footer-contact">
                <FaPhoneAlt aria-hidden="true" />
                <span>+91 98765 43210</span>
              </div>
            </div>
            <div className="footer-section">
              <h4 id="quick-links-heading">Quick Links</h4>
              <nav aria-labelledby="quick-links-heading">
                <Link to="/" aria-label="Go to Home Page"><span className="notranslate">Home</span></Link>
                <Link to="/advisor" aria-label="Consult the AI Advisor"><span className="notranslate">Advisor</span></Link>
                <Link to="/how-it-works" aria-label="How Fasal Saathi helps you"><span className="notranslate">How It Works</span></Link>
                <Link to="/schemes" aria-label="View Government Schemes for farmers"><span className="notranslate">Govt Schemes</span></Link>
                <Link to="/dashboard" aria-label="Go to your farming dashboard"><span className="notranslate">Dashboard</span></Link>
                <Link to="/calendar" aria-label="View your farming activity calendar"><span className="notranslate">Activity Calendar</span></Link>
                <Link to="/market-prices" aria-label="Check latest market prices for crops"><span className="notranslate">Market Prices</span></Link>
                <Link to="/community" aria-label="Join the community discussion"><span className="notranslate">Community</span></Link>
                <Link to="/share-feedback" aria-label="Share your thoughts with us"><span className="notranslate">Share Feedback</span></Link>
              </nav>
            </div>
            <div className="footer-section">
              <h4 id="resources-heading">Resources</h4>
              <nav aria-labelledby="resources-heading">
                <Link to="/crop-guide" aria-label="View the Crop Guide"><span className="notranslate">Crop Guide</span></Link>
                <Link to="/weather" aria-label="Check weather updates"><span className="notranslate">Weather Updates</span></Link>
                <Link to="/soil-analysis" aria-label="Get soil analysis insights"><span className="notranslate">Soil Analysis</span></Link>
                <Link to="/faq" aria-label="Frequently Asked Questions"><span className="notranslate">FAQs</span></Link>
              </nav>
            </div>
            <div className="footer-section">
              <h4 id="company-heading">Company</h4>
              <nav aria-labelledby="company-heading">
                <Link to="/about" aria-label="Learn about Fasal Saathi"><span className="notranslate">About Us</span></Link>
                <Link to="/contact" aria-label="Contact our support team"><span className="notranslate">Contact</span></Link>
                <Link to="/privacy-policy" aria-label="Read our Privacy Policy"><span className="notranslate">Privacy Policy</span></Link>
                <Link to="/terms" aria-label="Read our Terms of Service"><span className="notranslate">Terms of Service</span></Link>
              </nav>
            </div>
          </div>
          <div className="footer-bottom">
            <div className="footer-socials">
              <FaGlobe aria-hidden="true" />
              <span>Available Across India</span>
            </div>
            <p className="footer-copyright">
              © 2026 <span className="notranslate" translate="no">Fasal Saathi</span>. All rights reserved. MIT Licensed.
            </p>
          </div>
        </div>
      </footer>
    );
};

export default Footer;
