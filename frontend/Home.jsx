import React from "react";
import { Link } from "react-router-dom";
import {
  FaBrain,
  FaChartLine,
  FaHandHoldingWater,
  FaLeaf,
  FaLock,
  FaSun,
} from "react-icons/fa";
import "./Home.css";

const features = [
  {
    icon: <FaBrain />,
    title: "AI-Powered Predictions",
    description: "Use data-driven crop yield forecasts to plan each season with more confidence.",
  },
  {
    icon: <FaSun />,
    title: "Weather Insights",
    description: "Track weather conditions and adapt your farm operations to changing patterns.",
  },
  {
    icon: <FaHandHoldingWater />,
    title: "Smart Irrigation",
    description: "Reduce water waste with practical irrigation guidance and scheduling support.",
  },
  {
    icon: <FaChartLine />,
    title: "Yield Optimization",
    description: "Compare options and focus on the crops that can improve your results.",
  },
  {
    icon: <FaLeaf />,
    title: "Crop Recommendations",
    description: "Get crop suggestions that fit your soil and local farming conditions.",
  },
  {
    icon: <FaLock />,
    title: "Secure & Private",
    description: "Keep your data protected while you explore the platform.",
  },
];

const stats = [
  { value: "50K+", label: "Farmers Helped" },
  { value: "120+", label: "Crop Types" },
  { value: "98%", label: "Prediction Accuracy" },
  { value: "24/7", label: "Support" },
];

export default function Home() {
  return (
    <main className="home-page">
      <section className="home-hero">
        <div className="home-hero-copy">
          <p className="home-eyebrow">Smart farming made practical</p>
          <h1>Plan your farm with data, weather, and clear guidance.</h1>
          <p className="home-subtitle">
            Fasal Saathi brings yield prediction, crop guidance, weather insight, and farm tools into one place.
          </p>
          <div className="home-actions">
            <Link to="/login" className="home-primary-action">
              Get Started
            </Link>
            <Link to="/advisor" className="home-secondary-action">
              Open Advisor
            </Link>
          </div>
        </div>

        <div className="home-hero-panel">
          <div className="home-stat-grid">
            {stats.map((stat) => (
              <article key={stat.label} className="home-stat-card">
                <strong>{stat.value}</strong>
                <span>{stat.label}</span>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className="home-features">
        <div className="home-section-heading">
          <p>Features</p>
          <h2>Tools that help you make better farming decisions.</h2>
        </div>

        <div className="home-feature-grid">
          {features.map((feature) => (
            <article key={feature.title} className="home-feature-card">
              <div className="home-feature-icon">{feature.icon}</div>
              <h3>{feature.title}</h3>
              <p>{feature.description}</p>
            </article>
          ))}
        </div>
      </section>
    </main>
  );
}
