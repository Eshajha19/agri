import React, { useState } from "react";
import { Link } from "react-router-dom";
import { 
  Wifi, 
  BrainCircuit, 
  Sprout, 
  CloudSun, 
  LayoutDashboard, 
  TrendingUp,
  LineChart,
  CircleDollarSign,
  CheckCircle2,
  Users2,
  Check
} from "lucide-react";
import "./How.css";

export default function How() {
  const [activeStep, setActiveStep] = useState(0);

  const steps = [
    {
      icon: <Wifi size={24} />,
      title: "Collect Farm Data",
      desc: "Gather soil condition, crop type, weather patterns, and location-based insights.",
      details: "Our sensors and integrations automatically capture comprehensive farm data including soil moisture, temperature, humidity, and GPS location.",
      benefits: ["Real-time monitoring", "Automated data collection", "Historical tracking"],
      color: "blue",
    },
    {
      icon: <BrainCircuit size={24} />,
      title: "Smart AI Analysis",
      desc: "AI studies the data and generates accurate recommendations for farmers.",
      details: "Advanced machine learning models analyze patterns from thousands of farms to provide personalized insights specific to your region.",
      benefits: ["Predictive analytics", "Pattern recognition", "Custom models"],
      color: "green",
    },
    {
      icon: <Sprout size={24} />,
      title: "Crop Suggestions",
      desc: "Receive the best crop, fertilizer, and irrigation guidance for maximum yield.",
      details: "Get evidence-based recommendations for crop selection, planting schedules, and resource optimization tailored to your farm.",
      benefits: ["Yield optimization", "Cost reduction", "Sustainability"],
      color: "yellow",
    },
    {
      icon: <CloudSun size={24} />,
      title: "Weather Monitoring",
      desc: "Stay updated with rainfall, temperature, and storm alerts in real time.",
      details: "Receive real-time weather alerts, forecasts, and climate insights to help you plan farming activities effectively.",
      benefits: ["Instant alerts", "15-day forecasts", "Storm warnings"],
      color: "purple",
    },
    {
      icon: <LayoutDashboard size={24} />,
      title: "Easy Dashboard Access",
      desc: "View all insights on a clean, mobile-friendly dashboard anytime, anywhere.",
      details: "Access all your farm data, recommendations, and insights from any device with an intuitive, user-friendly interface.",
      benefits: ["Mobile-first design", "Real-time updates", "Offline access"],
      color: "orange",
    },
    {
      icon: <TrendingUp size={24} />,
      title: "Better Farming Results",
      desc: "Improve productivity, reduce waste, and increase profits with smarter decisions.",
      details: "See measurable improvements in crop yield, resource efficiency, and farm profitability within the first season.",
      benefits: ["Higher yields", "Lower costs", "Increased profits"],
      color: "red",
    },
  ];

  const outcomes = [
    { metric: "30-40%", label: "Higher Yield", icon: <LineChart size={32} /> },
    { metric: "25%", label: "Cost Reduction", icon: <CircleDollarSign size={32} /> },
    { metric: "99.9%", label: "Uptime", icon: <CheckCircle2 size={32} /> },
    { metric: "24/7", label: "Support", icon: <Users2 size={32} /> },
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
              className="step-card-wrap fade-up"
              data-step={index + 1}
            >
              <div className="step-card-inner">
                {/* FRONT FACE */}
                <div className="step-card-front">
                  <div className="step-number">0{index + 1}</div>
                  <div className="step-icon">{step.icon}</div>
                  <h3>{step.title}</h3>
                  <p>{step.desc}</p>
                  <div className="flip-hint">Hover to see details</div>
                </div>

                {/* BACK FACE */}
                <div className="step-card-back">
                  <h3>{step.title}</h3>
                  <div className="step-details">{step.details}</div>
                  <div className="step-benefits">
                    {step.benefits.map((benefit, i) => (
                      <span key={i} className="benefit-tag">
                        <Check size={14} className="inline-icon" /> {benefit}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
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
