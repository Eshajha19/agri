import React from "react";
import "./How.css";

export default function How() {
  const steps = [
    {
      icon: "📡",
      title: "Collect Farm Data",
      desc: "Gather soil condition, crop type, weather patterns, and location-based insights.",
      color: "blue",
    },
    {
      icon: "🤖",
      title: "Smart AI Analysis",
      desc: "AI studies the data and generates accurate recommendations for farmers.",
      color: "green",
    },
    {
      icon: "🌱",
      title: "Crop Suggestions",
      desc: "Receive the best crop, fertilizer, and irrigation guidance for maximum yield.",
      color: "yellow",
    },
    {
      icon: "☁️",
      title: "Weather Monitoring",
      desc: "Stay updated with rainfall, temperature, and storm alerts in real time.",
      color: "purple",
    },
    {
      icon: "📱",
      title: "Easy Dashboard Access",
      desc: "View all insights on a clean, mobile-friendly dashboard anytime, anywhere.",
      color: "orange",
    },
    {
      icon: "🚜",
      title: "Better Farming Results",
      desc: "Improve productivity, reduce waste, and increase profits with smarter decisions.",
      color: "red",
    },
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

      <div className="steps">
        {steps.map((step, index) => (
          <div
            key={index}
            className="step-card fade-up"
            data-step={index + 1}
          >
            <div className="step-number">0{index + 1}</div>

            <div className="step-icon">{step.icon}</div>

            <div className="step-content">
              <h3>{step.title}</h3>
              <p>{step.desc}</p>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
