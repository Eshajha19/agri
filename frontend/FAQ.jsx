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
    </div>
  );
}