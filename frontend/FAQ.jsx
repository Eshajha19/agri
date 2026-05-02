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
  },

  {
    q: "📍 Does Fasal Saathi work for my region?",
    a: "Yes, the system uses location-based data and regional agricultural patterns to provide recommendations tailored to your area."
  },
  {
    q: "🧠 Is AI replacing traditional farming knowledge?",
    a: "No, Fasal Saathi complements farmers’ experience by providing data-driven insights alongside traditional knowledge."
  },
  {
    q: "📈 How can it increase my crop yield?",
    a: "By optimizing crop selection, irrigation, fertilizer usage, and timing based on data, it helps maximize productivity."
  },
  {
    q: "🕒 How often is the data updated?",
    a: "Weather and environmental data are updated in real-time, while AI models improve continuously with new inputs."
  },
  {
    q: "💰 Is Fasal Saathi free to use?",
    a: "Basic features are free, while advanced analytics and premium insights may be introduced in future versions."
  },
  {
    q: "👨‍🌾 Can small farmers use this platform?",
    a: "Yes, Fasal Saathi is designed to be simple and accessible for farmers of all scales."
  },
  {
    q: "📊 Can I track my farm history?",
    a: "Yes, future updates will allow farmers to track past crops, yields, and soil data for better planning."
  },
  {
    q: "🚜 Does it support modern farming techniques?",
    a: "Yes, it provides insights aligned with precision agriculture, smart irrigation, and sustainable practices."
  },
  {
    q: "🌍 Is it useful for organic farming?",
    a: "Yes, recommendations can be adapted for organic farming practices, including natural fertilizers and pest control."
  },
  {
    q: "🔔 Will I get alerts or notifications?",
    a: "Yes, the system can notify you about weather changes, pest risks, and important farming actions."
  },
  {
    q: "🧾 Do I need technical knowledge to use it?",
    a: "No, the interface is designed to be simple and user-friendly, even for non-technical users."
  },
  {
    q: "📡 Can it work in low-network areas?",
    a: "The current version requires internet, but offline capabilities are planned for rural areas with limited connectivity."
  },
  {
    q: "🧪 How do I input soil data?",
    a: "You can manually enter soil values or integrate data from soil testing labs if available."
  },
  {
    q: "🤝 Can I connect with experts or other farmers?",
    a: "Future updates aim to include community and expert support features for collaboration and guidance."
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
        <p>We're here to help you with your farming journey.</p>
        <a href="/contact" className="faq-btn">
          Contact Support
        </a>
      </div>
    </div>
  );
}
