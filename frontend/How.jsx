import React from "react";
import "./How.css";

export default function How() {
  const steps = [
    {
      icon: "📡",
      title: "Data Collection",
      desc: "We gather live weather, soil, and market data from trusted sources.",
    },
    {
      icon: "🤖",
      title: "AI Processing",
      desc: "Our AI analyzes the data and generates personalized advice.",
    },
    {
      icon: "📱",
      title: "Easy Access",
      desc: "Farmers get insights through mobile-friendly dashboards & voice support.",
    },
    {
      icon: "🌐",
      title: "Multilingual",
      desc: "Available in English, Hindi, Marathi, and regional languages.",
    },
    {
      icon: "📶",
      title: "Offline Access",
      desc: "Guidance is stored so you can use it even without internet.",
    },
    {
      icon: "🚜",
      title: "Smarter Farming",
      desc: "Better crop planning, water saving, and increased profits.",
    },
  ];

  return (
    <section className="howitworks">
      <div className="howitworks-header">
        <h1>🚀 How It Works</h1>
        <p>From data to decisions — simple steps to empower farmers.</p>
      </div>

      <div className="steps">
        {steps.map((step, index) => (
          <div
            key={index}
            className="step-card fade-up"
            data-step={index + 1}
          >
            <div className="step-icon">{step.icon}</div>
            <h3>{step.title}</h3>
            <p>{step.desc}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
