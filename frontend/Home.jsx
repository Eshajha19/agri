import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  FaSeedling,
  FaSun,
  FaBrain,
  FaHandHoldingWater,
  FaChartLine,
  FaPhoneAlt,
  FaQuoteLeft,
  FaFlask,
  FaLeaf,
  FaLock,
  FaGlobe,
} from "react-icons/fa";
import WeatherAlertBar from "./weather/WeatherAlertBar";
import WeatherQuickWidget from "./weather/WeatherQuickWidget";
import "./Home.css";

export default function Home() {
  const features = [
    {
      icon: <FaBrain />,
      title: "AI-Powered Predictions",
      desc: "Smart crop yield predictions using advanced machine learning algorithms",
      category: "Analytics",
    },
    {
      icon: <FaSun />,
      title: "Weather Insights",
      desc: "Real-time weather forecasts and custom alerts tailored for your farm",
      category: "Monitoring",
    },
    {
      icon: <FaHandHoldingWater />,
      title: "Smart Irrigation",
      desc: "Optimize water usage with AI-driven irrigation recommendations",
      category: "Optimization",
    },
    {
      icon: <FaChartLine />,
      title: "Yield Optimization",
      desc: "Maximize your harvest with data-driven farming strategies",
      category: "Analytics",
    },
    {
      icon: <FaFlask />,
      title: "Soil Analysis",
      desc: "Comprehensive soil health monitoring and nutrient level analysis",
      category: "Monitoring",
    },
    {
      icon: <FaLeaf />,
      title: "Crop Recommendations",
      desc: "Get crop suggestions based on soil profile and regional climate",
      category: "Recommendations",
    },
    {
      icon: <FaChartLine />,
      title: "Fertilizer Guidance",
      desc: "Personalized fertilizer and pesticide recommendations",
      category: "Recommendations",
    },
    {
      icon: <FaLock />,
      title: "Secure & Private",
      desc: "Enterprise-grade security with Firebase authentication",
      category: "Protection",
    },
  ];

  const stats = [
    { target: 50, suffix: "K+", label: "Farmers Helped" },
    { target: 120, suffix: "+", label: "Crop Types" },
    { target: 98, suffix: "%", label: "Accuracy" },
    { target: 24, suffix: "/7", label: "Support" },
  ];

  const [statValues, setStatValues] = useState(stats.map(() => 1));

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    const navigationEntry = performance.getEntriesByType("navigation")?.[0];
    if (navigationEntry?.type === "reload") {
      window.scrollTo(0, 0);
    }
  }, []);

  useEffect(() => {
    const duration = 1400;
    const startTime = performance.now();
    let rafId = 0;

    const animateStats = (currentTime) => {
      const progress = Math.min((currentTime - startTime) / duration, 1);

      setStatValues(
        stats.map((stat) => Math.max(1, Math.floor(stat.target * progress)))
      );

      if (progress < 1) {
        rafId = requestAnimationFrame(animateStats);
      }
    };

    rafId = requestAnimationFrame(animateStats);
    return () => cancelAnimationFrame(rafId);
  }, []);

  const testimonials = [
    {
      name: "Ramesh Kumar",
      location: "Maharashtra",
      text: "Fasal Saathi helped me increase my rice yield by 30% this season!",
    },
    {
      name: "Lakshmi Devi",
      location: "Tamil Nadu",
      text: "The weather predictions are accurate. I plan my irrigation accordingly.",
    },
    {
      name: "Suresh Patel",
      location: "Gujarat",
      text: "Best AI farming assistant. Simple to use even for elderly farmers.",
    },
  ];

  return (
    <div className="home">
      <WeatherAlertBar />
      <div className="home-weather-relative-wrap">
        <WeatherQuickWidget />
      </div>
      <section className="hero-section">
        <div className="hero-bg">
          <div className="hero-pattern"></div>
        </div>
        <div className="hero-content">
          <div className="hero-copy">
            <div className="hero-badge">
              <FaSeedling /> AI-Powered Farming Assistant
            </div>
            <h1 className="hero-title">
              Smart Farming with <span className="highlight">AI</span>
            </h1>
            <p className="hero-subtitle">
              Get AI-driven crop recommendations, weather insights, and yield predictions
              to maximize your agricultural productivity.
            </p>
            <div className="hero-buttons">
              <Link to="/advisor" className="btn-primary">
                Get Started
              </Link>
              <Link to="/how-it-works" className="btn-secondary">
                Learn More
              </Link>
            </div>
          </div>
          <div className="hero-stats">
            {stats.map((stat, index) => (
              <div key={index} className="stat-item">
                <span className="stat-number">
                  {statValues[index]}
                  {stat.suffix}
                </span>
                <span className="stat-label">{stat.label}</span>
              </div>
            ))}
          </div>
        </div>
        <div className="hero-visual">
          <div className="floating-card card-1">
            <FaSun className="card-icon" />
            <span>28°C - Sunny</span>
          </div>
          <div className="floating-card card-2">
            <FaSeedling className="card-icon" />
            <span>Yield: +30%</span>
          </div>
          <div className="floating-card card-3">
            <FaHandHoldingWater className="card-icon" />
            <span>Optimal Irrigation</span>
          </div>
        </div>
      </section>

      <section className="features-section">
        <div className="section-header">
          <h2>Powerful Features for Modern Farming</h2>
          <p>Everything you need to succeed in agriculture</p>
        </div>
        <div className="features-grid">
          {features.map((feature, index) => (
            <div key={index} className="feature-card">
              <div className="feature-category">{feature.category}</div>
              <div className="feature-icon">{feature.icon}</div>
              <h3>{feature.title}</h3>
              <p>{feature.desc}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="how-section">
        <div className="section-header">
          <h2>How It Works</h2>
          <p>Three simple steps to smarter farming</p>
        </div>
        <div className="steps-container">
          <div className="step">
            <div className="step-number">1</div>
            <h3>Enter Farm Details</h3>
            <p>Input your crop type, area, and farming conditions</p>
          </div>
          <div className="step-connector"></div>
          <div className="step">
            <div className="step-number">2</div>
            <h3>AI Analysis</h3>
            <p>Our ML models analyze your data instantly</p>
          </div>
          <div className="step-connector"></div>
          <div className="step">
            <div className="step-number">3</div>
            <h3>Get Recommendations</h3>
            <p>Receive personalized farming advice</p>
          </div>
        </div>
        <Link to="/advisor" className="btn-primary">
          Try It Now
        </Link>
      </section>

      <section className="testimonials-section">
        <div className="section-header">
          <h2>What Farmers Say</h2>
          <p>Real experiences from real farmers</p>
        </div>
        <div className="testimonials-grid">
          {testimonials.map((testimonial, index) => (
            <div key={index} className="testimonial-card">
              <FaQuoteLeft className="quote-icon" />
              <p className="testimonial-text">{testimonial.text}</p>
              <div className="testimonial-author">
                <div className="author-avatar">
                  {testimonial.name[0]}
                </div>
                <div className="author-info">
                  <span className="author-name">{testimonial.name}</span>
                  <span className="author-location">{testimonial.location}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="cta-section">
        <h2>Ready to Transform Your Farm?</h2>
        <p>Join thousands of farmers already benefiting from AI-powered agriculture</p>
        <Link to="/advisor" className="btn-primary">
          Start Free Consultation
        </Link>
      </section>

      
          <footer className="home-footer">
  <div className="footer-content">
    <div className="footer-grid">
      <div className="footer-section">
        <div className="footer-brand">
          <FaSeedling className="footer-logo" />
          <span>Fasal Saathi</span>
        </div>
        <p className="footer-description">
          AI-powered agricultural advisor helping farmers with crop planning,
          weather insights, irrigation, and yield optimization.
        </p>
        <div className="footer-contact">
          <FaPhoneAlt />
          <span>+91 98765 43210</span>
        </div>
      </div>

      <div className="footer-section">
        <h4>Quick Links</h4>
        <Link to="/">Home</Link>
        <Link to="/advisor">Advisor</Link>
        <Link to="/how-it-works">How It Works</Link>
        <Link to="/dashboard">Dashboard</Link>
      </div>

      <div className="footer-section">
        <h4>Resources</h4>
        <Link to="/crop-guide">Crop Guide</Link>
        <Link to="/weather">Weather Updates</Link>
        <Link to="/soil-analysis">Soil Analysis</Link>
        <Link to="/faq">FAQs</Link>
      </div>

      <div className="footer-section">
        <h4>Company</h4>
        <Link to="/about">About Us</Link>
        <Link to="/contact">Contact</Link>
        <Link to="/privacy-policy">Privacy Policy</Link>
        <Link to="/terms">Terms of Service</Link>
      </div>
    </div>

    <div className="footer-bottom">
      <div className="footer-socials">
        <FaGlobe />
        <span>Available Across India</span>
      </div>
      <p className="footer-copyright">
        © 2026 Fasal Saathi. All rights reserved. MIT Licensed.
      </p>
    </div>
  </div>
</footer>
    </div>
  );
}
