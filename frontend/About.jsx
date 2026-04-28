import React from "react";
import { FaLeaf, FaUsers, FaBullseye, FaHeart, FaShieldAlt, FaLightbulb, FaCloud } from "react-icons/fa";
import "./About.css";

export default function About() {
  return (
    <div className="about-page">
      <section className="about-hero">
        <div className="hero-content">
          <FaLeaf className="hero-icon" />
          <h1>About <span className="notranslate">Fasal Saathi</span></h1>
          <p className="hero-subtitle">
            Empowering Indian farmers with AI-driven agricultural intelligence
          </p>
        </div>
      </section>

      <section className="about-mission">
        <div className="container">
          <h2>Our Mission</h2>
          <p>
            <span className="notranslate">Fasal Saathi</span> is dedicated to transforming Indian agriculture by making
            cutting-edge technology accessible to every farmer. We believe that AI and data-driven insights
            can help farmers increase yields, reduce costs, and build sustainable farming practices.
          </p>
        </div>
      </section>

      <section className="about-values">
        <div className="container">
          <h2>What We Do</h2>
          <div className="values-grid">
            <div className="value-card">
              <FaBullseye className="value-icon" />
              <h3>Smart Crop Recommendations</h3>
              <p>AI-powered suggestions based on soil health, weather patterns, and regional data</p>
            </div>
            <div className="value-card">
              <FaShieldAlt className="value-icon" />
              <h3>Soil Health Analysis</h3>
              <p>Comprehensive soil nutrient analysis with actionable improvement recommendations</p>
            </div>
            <div className="value-card">
              <FaCloud className="value-icon" />
              <h3>Weather Intelligence</h3>
              <p>Real-time forecasts, alerts, and irrigation planning based on accurate weather data</p>
            </div>
            <div className="value-card">
              <FaUsers className="value-icon" />
              <h3>Community Collaboration</h3>
              <p>Connect with fellow farmers, share experiences, and learn together</p>
            </div>
          </div>
        </div>
      </section>

      <section className="about-why">
        <div className="container">
          <h2>Why <span className="notranslate">Fasal Saathi</span>?</h2>
          <ul className="why-list">
            <li><FaLightbulb /> Built specifically for Indian farmers and local conditions</li>
            <li><FaHeart /> Easy-to-use interface available in multiple Indian languages</li>
            <li><FaShieldAlt /> Secure and private - your farm data stays yours</li>
            <li><FaUsers /> Community-driven with expert-backed recommendations</li>
          </ul>
        </div>
      </section>

      <section className="about-cta">
        <div className="container">
          <h2>Join the Agricultural Revolution</h2>
          <p>Start your journey towards smarter, more sustainable farming today.</p>
        </div>
      </section>
    </div>
  );
}
