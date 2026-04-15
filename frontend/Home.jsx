import React from "react";
import { Link } from "react-router-dom";
import {
  FaBrain,
  FaChartLine,
  FaHandHoldingWater,
  FaPhoneAlt,
  FaQuoteLeft,
  FaSeedling,
  FaSun,
} from "react-icons/fa";
import WeatherCard from "./WeatherCard";
import "./Home.css";

export default function Home() {
  const features = [
    {
      icon: <FaBrain />,
      title: "AI-Powered Predictions",
      desc: "Smart crop yield predictions using advanced machine learning.",
    },
    {
      icon: <FaSun />,
      title: "Hyperlocal Weather Alerts",
      desc: "Live rain, frost, and heatwave warnings tailored to your exact farm location.",
    },
    {
      icon: <FaHandHoldingWater />,
      title: "Irrigation Advice",
      desc: "Smart irrigation recommendations to optimize water usage.",
    },
    {
      icon: <FaChartLine />,
      title: "Yield Optimization",
      desc: "Maximize your harvest with data-driven farming strategies.",
    },
  ];

  const stats = [
    { number: "50K+", label: "Farmers Helped" },
    { number: "120+", label: "Crop Types" },
    { number: "98%", label: "Model Accuracy" },
    { number: "24/7", label: "Weather Monitoring" },
  ];

  const testimonials = [
    {
      name: "Ramesh Kumar",
      location: "Maharashtra",
      text: "Fasal Saathi helped me increase my rice yield by 30% this season.",
    },
    {
      name: "Lakshmi Devi",
      location: "Tamil Nadu",
      text: "The weather warnings are timely, so I plan irrigation and spraying with more confidence.",
    },
    {
      name: "Suresh Patel",
      location: "Gujarat",
      text: "The crop-specific weather advice is simple enough for my whole family to follow.",
    },
  ];

  return (
    <div className="home">
      <section className="hero-section">
        <div className="hero-bg">
          <div className="hero-pattern"></div>
        </div>

        <div className="hero-content">
          <div className="hero-badge">
            <FaSeedling /> AI-Powered Farming Assistant
          </div>

          <h1 className="hero-title">
            Farm Smarter With <span className="highlight">Live Weather Intelligence</span>
          </h1>

          <p className="hero-subtitle">
            Get AI-driven crop recommendations, hyperlocal weather alerts, and yield insights
            that help you act before conditions hurt your field.
          </p>

          <div className="hero-buttons">
            <Link to="/advisor" className="btn-primary">
              Open Advisor
            </Link>
            <Link to="/how-it-works" className="btn-secondary">
              Learn More
            </Link>
          </div>

          <div className="hero-stats">
            {stats.map((stat) => (
              <div key={stat.label} className="stat-item">
                <span className="stat-number">{stat.number}</span>
                <span className="stat-label">{stat.label}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="hero-visual">
          <div className="floating-card card-1">
            <FaSun className="card-icon" />
            <span>Live weather alerts</span>
          </div>
          <div className="floating-card card-2">
            <FaSeedling className="card-icon" />
            <span>Crop warnings by field</span>
          </div>
          <div className="floating-card card-3">
            <FaHandHoldingWater className="card-icon" />
            <span>Faster irrigation decisions</span>
          </div>
        </div>
      </section>

      <section className="weather-section">
        <div className="section-header">
          <h2>Dynamic Location-Based Weather on the Landing Page</h2>
          <p>
            Search your village or allow location access to see live conditions, extreme alerts,
            and crop-specific warnings immediately.
          </p>
        </div>

        <div className="weather-section__content">
          <WeatherCard
            embedded
            title="Weather Alerts for Your Exact Farm Location"
            subtitle="Built for quick action on rain, frost, and heatwave conditions that can change crop decisions."
          />
        </div>
      </section>

      <section className="features-section">
        <div className="section-header">
          <h2>Powerful Features for Modern Farming</h2>
          <p>Everything you need to make better field decisions.</p>
        </div>

        <div className="features-grid">
          {features.map((feature) => (
            <div key={feature.title} className="feature-card">
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
          <p>Three simple steps to smarter farming.</p>
        </div>

        <div className="steps-container">
          <div className="step">
            <div className="step-number">1</div>
            <h3>Set your field location</h3>
            <p>Use geolocation or search your village, town, or district.</p>
          </div>
          <div className="step-connector"></div>
          <div className="step">
            <div className="step-number">2</div>
            <h3>Track live risks</h3>
            <p>We translate weather API data into rain, frost, and heatwave alerts.</p>
          </div>
          <div className="step-connector"></div>
          <div className="step">
            <div className="step-number">3</div>
            <h3>Act by crop</h3>
            <p>See field-ready warnings and timing advice for the crop you selected.</p>
          </div>
        </div>

        <Link to="/advisor" className="btn-primary">
          Try It Now
        </Link>
      </section>

      <section className="testimonials-section">
        <div className="section-header">
          <h2>What Farmers Say</h2>
          <p>Real experiences from real farmers.</p>
        </div>

        <div className="testimonials-grid">
          {testimonials.map((testimonial) => (
            <div key={testimonial.name} className="testimonial-card">
              <FaQuoteLeft className="quote-icon" />
              <p className="testimonial-text">{testimonial.text}</p>
              <div className="testimonial-author">
                <div className="author-avatar">{testimonial.name[0]}</div>
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
        <p>Join thousands of farmers already benefiting from AI-powered agriculture.</p>
        <Link to="/advisor" className="btn-primary">
          Start Free Consultation
        </Link>
      </section>

      <footer className="home-footer">
        <div className="footer-content">
          <div className="footer-brand">
            <FaSeedling className="footer-logo" />
            <span>Fasal Saathi</span>
          </div>
          <p>AI-powered agricultural advisor for Indian farmers.</p>
          <div className="footer-contact">
            <FaPhoneAlt /> Need help? Call us: +91 98765 43210
          </div>
          <p className="footer-copyright">
            Copyright 2026 Fasal Saathi. All rights reserved. MIT Licensed.
          </p>
        </div>
      </footer>
    </div>
  );
}
