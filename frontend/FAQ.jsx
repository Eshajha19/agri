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
import { HelpCircle, ChevronDown, ChevronUp, MessageCircle, Mail, Phone } from "lucide-react";
import "./FAQ.css";

const FAQ_DATA = [
  {
    question: "How do I get crop recommendations?",
    answer: "Navigate to the Advisor section and enter your soil type, location, and season. Our AI will suggest the best crops for you based on real-time data and historical trends."
  },
  {
    question: "Is Fasal Saathi available in my local language?",
    answer: "Yes! We support 12 Indian languages including Hindi, Bengali, Tamil, Telugu, Marathi, and more. You can easily switch your preferred language using the selector in the navigation bar."
  },
  {
    question: "How accurate are the weather forecasts?",
    answer: "We integrate with trusted meteorological sources to provide hyper-local weather insights. While forecasts are highly accurate, we recommend checking regularly for any sudden changes."
  },
  {
    question: "Can I use Fasal Saathi without an internet connection?",
    answer: "Fasal Saathi is designed with offline-first capabilities. Basic features and cached data are available even without internet, and your updates will sync automatically once you're back online."
  },
  {
    question: "How does the Soil Analysis tool work?",
    answer: "The tool allows you to input NPK (Nitrogen, Phosphorus, Potassium) values from your soil test report. It then analyzes these values to provide a health score and specific fertilizer recommendations."
  },
  {
    question: "How do I report a problem or bug?",
    answer: "You can reach out to us through the Contact Us page or join our Farmer Community. Our technical team monitors these channels 24/7 to ensure a seamless experience for all farmers."
  }
];

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
      <div className="faq-container">
        <div className="faq-header">
          <div className="faq-icon-wrap">
            <HelpCircle size={40} className="faq-main-icon" />
          </div>
          <h1>Frequently Asked Questions</h1>
          <p className="faq-subtitle">Find quick answers to common queries about Fasal Saathi and modern farming.</p>
        </div>

        <div className="faq-list">
          {FAQ_DATA.map((item, index) => (
            <div 
              key={index} 
              className={`faq-item ${activeIndex === index ? "active" : ""}`}
            >
              <button 
                className="faq-question" 
                onClick={() => toggleFAQ(index)}
                aria-expanded={activeIndex === index}
              >
                <span>{item.question}</span>
                {activeIndex === index ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
              </button>
              <div className={`faq-answer ${activeIndex === index ? "show" : ""}`}>
                <p>{item.answer}</p>
              </div>
            </div>
          ))}
        </div>

        <div className="faq-footer">
          <h3>Still have questions?</h3>
          <p>We're here to help you grow. Contact our support team directly.</p>
          <div className="faq-contact-grid">
            <div className="faq-contact-card">
              <Mail size={24} />
              <span>Email Support</span>
              <p>support@fasalsaathi.com</p>
            </div>
            <div className="faq-contact-card">
              <Phone size={24} />
              <span>Call Us</span>
              <p>+91 98765 43210</p>
            </div>
            <div className="faq-contact-card">
              <MessageCircle size={24} />
              <span>Community</span>
              <p>Join the Discussion</p>
            </div>
          </div>
          <a href="/contact" className="faq-btn">Contact Us Now</a>
        </div>
      </div>
    </div>
  );
}