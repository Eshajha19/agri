import React from "react";
import "./Legal.css";

export default function FAQ() {
  const faqs = [
    {
      question: "What is Fasal Saathi?",
      answer: "Fasal Saathi is an AI-powered agricultural advisory platform that provides personalized recommendations for crop selection, soil management, and farming practices based on your farm data."
    },
    {
      question: "How accurate are the recommendations?",
      answer: "Our AI models are trained on extensive agricultural data and provide guidance based on proven farming practices. However, recommendations should be validated with local agricultural experts."
    },
    {
      question: "Is my farm data secure?",
      answer: "Yes, we use Firebase for secure data storage with encryption. Your data is never sold to third parties and is only used to provide better recommendations."
    },
    {
      question: "Can I use Fasal Saathi offline?",
      answer: "Currently, Fasal Saathi requires an internet connection to access our AI models and weather data. Offline functionality may be added in future updates."
    },
    {
      question: "How do I update my farm information?",
      answer: "You can update your farm data through the chat interface or by accessing your profile settings. Keep your information current for the best recommendations."
    }
  ];

  return (
    <div className="legal-page">
      <h1>Frequently Asked Questions</h1>
      <p className="last-updated">Last Updated: April 2026</p>

      <div className="faq-list">
        {faqs.map((faq, index) => (
          <section key={index} className="faq-item">
            <h2>{faq.question}</h2>
            <p>{faq.answer}</p>
          </section>
        ))}
      </div>

      <section>
        <h2>Still have questions?</h2>
        <p>
          If you couldn't find the answer you're looking for, please contact us
          through the Contact page or use our chat support.
        </p>
      </section>
import React, { useState } from "react";
import "./FAQ.css";

export default function FAQ() {
  const [activeIndex, setActiveIndex] = useState(null);

  const faqs = [
    {
      q: "🌾 What is Fasal Saathi?",
      a: "Fasal Saathi is an AI-powered agricultural assistant that helps farmers with crop recommendations, soil analysis, weather updates, and yield optimization."
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
      a: "Yes, Fasal Saathi suggests optimal irrigation schedules based on weather forecasts and soil moisture data."
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
      q: "📱 Can I use Fasal Saathi on mobile devices?",
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
        Everything you need to know about using Fasal Saathi effectively
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
        <p>We’re here to help you with your farming journey.</p>
        <a href="/contact" className="faq-btn">
          Contact Support
        </a>
      </div>
    </div>
  );
}