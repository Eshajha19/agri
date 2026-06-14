import React from "react";
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
  Check,
} from "lucide-react";
import "./How.css";

export default function How() {
  const steps = [
    {
      icon: <Wifi size={28} />,
      title: "Collect Farm Data",
      desc: "Automatically gather soil conditions, weather updates, crop details, and location information.",
      benefits: [
        "Real-time monitoring",
        "Automatic data collection",
        "Historical records",
      ],
    },
    {
      icon: <BrainCircuit size={28} />,
      title: "AI Analyzes the Data",
      desc: "Advanced AI processes your farm information and identifies the best recommendations.",
      benefits: [
        "Predictive analytics",
        "Smart insights",
        "Personalized advice",
      ],
    },
    {
      icon: <Sprout size={28} />,
      title: "Get Crop Recommendations",
      desc: "Receive suggestions for crops, fertilizers, irrigation schedules, and planting strategies.",
      benefits: [
        "Higher yield",
        "Reduced costs",
        "Sustainable farming",
      ],
    },
    {
      icon: <CloudSun size={28} />,
      title: "Monitor Weather",
      desc: "Stay informed with rainfall forecasts, temperature updates, and severe weather alerts.",
      benefits: [
        "Live weather updates",
        "Storm notifications",
        "15-day forecast",
      ],
    },
    {
      icon: <LayoutDashboard size={28} />,
      title: "Track Everything",
      desc: "View recommendations and farm insights from a simple dashboard on any device.",
      benefits: [
        "Mobile friendly",
        "Real-time sync",
        "Easy navigation",
      ],
    },
    {
      icon: <TrendingUp size={28} />,
      title: "Increase Productivity",
      desc: "Make data-driven decisions that improve crop yield while lowering operational costs.",
      benefits: [
        "Higher profits",
        "Better efficiency",
        "Smarter farming",
      ],
    },
  ];

  const outcomes = [
    {
      metric: "30–40%",
      label: "Higher Yield",
      icon: <LineChart size={30} />,
    },
    {
      metric: "25%",
      label: "Cost Reduction",
      icon: <CircleDollarSign size={30} />,
    },
    {
      metric: "99.9%",
      label: "System Uptime",
      icon: <CheckCircle2 size={30} />,
    },
    {
      metric: "24/7",
      label: "Farmer Support",
      icon: <Users2 size={30} />,
    },
  ];

  return (
    <section className="howitworks">
      {/* Header */}

      <div className="howitworks-header">
        <span className="section-tag">How It Works</span>

        <h1>How Our Smart Farming Platform Works</h1>

        <p>
          From collecting farm data to providing AI-powered recommendations,
          our platform helps farmers make better decisions, improve crop
          productivity, and reduce costs in just a few simple steps.
        </p>
      </div>

      {/* Steps */}

      <div className="steps-container">
        <div className="steps">
          {steps.map((step, index) => (
            <div className="step-card" key={index}>
              <div className="step-left">
                <div className="step-number">
                  {(index + 1).toString().padStart(2, "0")}
                </div>

                <div className="step-icon">{step.icon}</div>
              </div>

              <div className="step-content">
                <h3>{step.title}</h3>

                <p>{step.desc}</p>

                <ul className="benefits-list">
                  {step.benefits.map((item, i) => (
                    <li key={i}>
                      <Check size={16} />
                      <span>{item}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Outcomes */}

      <div className="outcomes-section">
        <h2>Expected Benefits</h2>

        <div className="outcomes-grid">
          {outcomes.map((item, index) => (
            <div className="outcome-card" key={index}>
              <div className="outcome-icon">{item.icon}</div>

              <h3>{item.metric}</h3>

              <p>{item.label}</p>
            </div>
          ))}
        </div>
      </div>

      {/* CTA */}

      <div className="cta-section">
        <h2>Ready to Grow Smarter?</h2>

        <p>
          Join thousands of farmers using AI to increase productivity and make
          informed farming decisions every day.
        </p>

        <Link to="/login">
          <button className="cta-button">
            Get Started for Free
          </button>
        </Link>
      </div>
    </section>
  );
}