import React, { useState } from "react";
import "./FAQ.css";

export default function Faq() {
  const [activeIndex, setActiveIndex] = useState(null);

  const faqs = [
    {
      q: "🌾 What is <span className=\"notranslate\">Fasal Saathi</span>?",
      a: "<span className=\"notranslate\">Fasal Saathi</span> is an AI-powered agricultural assistant that helps farmers with crop recommendations, soil analysis, weather updates, and yield optimization."
    },
    {
      q: "🤖 How does the crop recommendation system work?",
      a: "It uses machine learning models that analyze soil nutrients (NPK), pH levels, weather conditions, and regional data to suggest the most suitable crops."
    },
    {
      q: "🌦️ How accurate is the weather data?",
      a: "Weather data is fetched from trusted APIs like OpenWeatherMap, providing real-time forecasts and alerts tailored to your location."
    },
    {
      q: "🧪 What parameters are used in soil analysis?",
      a: "Soil analysis considers Nitrogen (N), Phosphorus (P), Potassium (K), pH level, moisture, and other nutrients to give recommendations."
    },
    {
      q: "💧 Can it help with irrigation planning?",
      a: "Yes, <span className=\"notranslate\">Fasal Saathi</span> suggests optimal irrigation schedules based on weather forecasts and soil moisture data."
    },
    {
      q: "🌱 Which crops are supported?",
      a: "The platform supports a wide range of crops including rice, wheat, maize, pulses, and more based on regional compatibility."
    },
    {
      q: "🔐 Is my personal and farm data safe?",
      a: "Yes, we use Firebase Authentication and secure cloud storage to ensure your data is protected and private."
    },
    {
      q: "📱 Can I use <span className=\"notranslate\">Fasal Saathi</span> on mobile devices?",
      a: "Absolutely! The platform is fully responsive and works seamlessly on smartphones, tablets, and desktops."
    },
    {
      q: "🌐 Does the platform support multiple languages?",
      a: "Multi-language support is planned to make the platform accessible to farmers across different regions."
    },
    {
      q: "📊 Does it provide yield prediction?",
      a: "Yes, AI models analyze inputs to estimate potential crop yield and help farmers plan better."
    },
    {
      q: "💊 Does it suggest fertilizers and pesticides?",
      a: "Yes, based on soil health and crop type, it provides personalized fertilizer and pesticide recommendations."
    },
    {
      q: "⚡ Do I need internet to use it?",
      a: "Currently yes, but offline/PWA support is planned for low-connectivity areas."
    }
  ];

  const toggleFAQ = (index) => {
    setActiveIndex(activeIndex === index ? null : index);
  };

  return (
    <div className="faq-page">
      <h1>Frequently Asked Questions</h1>
        <p className="faq-subtitle">
          Everything you need to know about using <span className="notranslate\">Fasal Saathi</span> effectively
        </p>

      <div className="faq-container">
        {faqs.map((faq, index) => (
          <div key={index} className="faq-item">
            <div
              className={`faq-question ${activeIndex === index ? "active" : ""}`}
              onClick={() => toggleFAQ(index)}
            >
              {faq.q}
              <span className="faq-toggle">
                {activeIndex === index ? "-" : "+"}
              </span>
            </div>

            <div
              className={`faq-answer ${
                activeIndex === index ? "show" : ""
              }`}
            >
              {faq.a}
            </div>
          </div>
        ))}
      </div>

      {/* Extra Section */}
      <div className="faq-contact">
        <h3>Still have questions?</h3>
        <p>We're here to help you with your farming journey.</p>
        <a href="/contact" className="faq-btn">
          Contact Support
        </a>
      </div>
    </div>
  );
}