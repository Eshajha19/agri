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