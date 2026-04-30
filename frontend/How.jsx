import React, { useState } from "react";
import { Link } from "react-router-dom";
import "./How.css";

export default function How() {
  const [activeStep, setActiveStep] = useState(0);

  const steps = [
    {
      icon: "📡",
      title: "Collect Farm Data",
      desc: "Gather soil condition, crop type, weather patterns, and location-based insights.",
      details: "Our sensors and integrations automatically capture comprehensive farm data including soil moisture, temperature, humidity, and GPS location.",
      benefits: ["Real-time monitoring", "Automated data collection", "Historical tracking"],
      color: "blue",
    },
    {
      icon: "🤖",
      title: "Smart AI Analysis",
      desc: "AI studies the data and generates accurate recommendations for farmers.",
      details: "Advanced machine learning models analyze patterns from thousands of farms to provide personalized insights specific to your region.",
      benefits: ["Predictive analytics", "Pattern recognition", "Custom models"],
      color: "green",
    },
    {
      icon: "🌱",
      title: "Crop Suggestions",
      desc: "Receive the best crop, fertilizer, and irrigation guidance for maximum yield.",
      details: "Get evidence-based recommendations for crop selection, planting schedules, and resource optimization tailored to your farm.",
      benefits: ["Yield optimization", "Cost reduction", "Sustainability"],
      color: "yellow",
    },
    {
      icon: "☁️",
      title: "Weather Monitoring",
      desc: "Stay updated with rainfall, temperature, and storm alerts in real time.",
      details: "Receive real-time weather alerts, forecasts, and climate insights to help you plan farming activities effectively.",
      benefits: ["Instant alerts", "15-day forecasts", "Storm warnings"],
      color: "purple",
    },
    {
      icon: "📱",
      title: "Easy Dashboard Access",
      desc: "View all insights on a clean, mobile-friendly dashboard anytime, anywhere.",
      details: "Access all your farm data, recommendations, and insights from any device with an intuitive, user-friendly interface.",
      benefits: ["Mobile-first design", "Real-time updates", "Offline access"],
      color: "orange",
    },
    {
      icon: "🚜",
      title: "Better Farming Results",
      desc: "Improve productivity, reduce waste, and increase profits with smarter decisions.",
      details: "See measurable improvements in crop yield, resource efficiency, and farm profitability within the first season.",
      benefits: ["Higher yields", "Lower costs", "Increased profits"],
      color: "red",
    },
  ];

  const outcomes = [
    { metric: "30-40%", label: "Higher Yield", icon: "📈" },
    { metric: "25%", label: "Cost Reduction", icon: "💰" },
    { metric: "99.9%", label: "Uptime", icon: "✅" },
    { metric: "24/7", label: "Support", icon: "🤝" },
  ];

  return (
    <section className="howitworks">
      <div className="howitworks-header">
        <span className="section-tag">How It Works</span>
        <h1>Transforming Farm Data into Smart Decisions</h1>
        <p>
          Our platform simplifies farming by turning complex data into clear,
          actionable insights for better crop planning and productivity.
        </p>
      </div>

      <div className="steps-container">
        <div className="steps">
          {steps.map((step, index) => (
            <div
              key={index}
              className={`step-card fade-up ${activeStep === index ? "active" : ""}`}
              data-step={index + 1}
              onClick={() => setActiveStep(index)}
              role="button"
              tabIndex="0"
              onKeyPress={(e) => {
                if (e.key === "Enter" || e.key === " ") setActiveStep(index);
              }}
            >
              <div className="step-number">0{index + 1}</div>

              <div className="step-icon">{step.icon}</div>

              <div className="step-content">
                <h3>{step.title}</h3>
                <p>{step.desc}</p>
                
                {activeStep === index && (
                  <div className="step-expanded">
                    <div className="step-details">{step.details}</div>
                    <div className="step-benefits">
                      {step.benefits.map((benefit, i) => (
                        <span key={i} className="benefit-tag">✓ {benefit}</span>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              <div className="step-indicator"></div>
            </div>
          ))}
        </div>
      </div>

      <div className="outcomes-section">
        <h2>Proven Results</h2>
        <div className="outcomes-grid">
          {outcomes.map((outcome, index) => (
            <div key={index} className="outcome-card">
              <div className="outcome-icon">{outcome.icon}</div>
              <div className="outcome-metric">{outcome.metric}</div>
              <div className="outcome-label">{outcome.label}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="cta-section">
        <h2>Ready to Transform Your Farm?</h2>
        <p>Start using our platform today and see the difference in your crop yield and farm profitability.</p>
        <Link to="/login">
          <button className="cta-button">Get Started Free</button>
        </Link>
      </div>
    </section>
  );
}
