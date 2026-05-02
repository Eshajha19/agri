import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import {
  FaSeedling,
  FaSun,
  FaBrain,
  FaHandHoldingWater,
  FaChartLine,
  FaPhoneAlt,
  FaQuoteLeft,
  FaFlask,
  FaLeaf,
  FaLock,
  FaGlobe,
  FaLandmark,
  FaCalendarAlt,
  FaCommentDots,
  FaCalculator,
  FaUsers,
  FaBug,
  FaArrowRight,
  FaBook,
  FaShieldAlt
} from "react-icons/fa";
import WeatherAlertBar from "./weather/WeatherAlertBar";
import WeatherQuickWidget from "./weather/WeatherQuickWidget";
import "./Home.css";

const features = [
    {
      icon: <FaBrain />,
      title: "AI-Powered Predictions",
      desc: "Smart crop yield predictions using advanced machine learning algorithms",
      category: "Analytics",
      link: "/advisor"
    },
    {
      icon: <FaSun />,
      title: "Weather Insights",
      desc: "Real-time weather forecasts and custom alerts tailored for your farm",
      category: "Monitoring",
      link: "/dashboard"
    },
    {
      icon: <FaHandHoldingWater />,
      title: "Smart Irrigation",
      desc: "Optimize water usage with AI-driven irrigation recommendations",
      category: "Optimization",
      link: "/advisor"
    },
    {
      icon: <FaChartLine />,
      title: "Yield Optimization",
      desc: "Maximize your harvest with data-driven farming strategies",
      category: "Analytics",
      link: "/advisor"
    },
    {
      icon: <FaFlask />,
      title: "Soil Analysis",
      desc: "Comprehensive soil health monitoring and nutrient level analysis",
      category: "Monitoring",
      link: "/soil-guide"
    },
    {
      icon: <FaLeaf />,
      title: "Crop Recommendations",
      desc: "Get crop suggestions based on soil profile and regional climate",
      category: "Recommendations",
      link: "/crop-guide"
    },
    {
      icon: <FaChartLine />,
      title: "Fertilizer Guidance",
      desc: "Personalized fertilizer and pesticide recommendations",
      category: "Recommendations",
      link: "/advisor"
    },
    {
      icon: <FaBug />,
      title: "Disease Awareness",
      desc: "Learn about common crop diseases, their symptoms, and effective remedies",
      category: "Education",
      link: "/disease-awareness"
    },
    {
      icon: <FaCalendarAlt />,
      title: "Seasonal Crop Planner",
      desc: "Plan your yearly farming cycles with optimized crop rotation schedules",
      category: "Planning",
      link: "/crop-planner"
    },
    {
      icon: <FaLock />,
      title: "Secure & Private",
      desc: "Enterprise-grade security with Firebase authentication",
      category: "Protection",
      link: "/login"
    },
    {
      icon: <FaPhoneAlt />,
      title: "Emergency Helpline",
      desc: "Instant access to agriculture support, weather emergencies, and insurance helplines",
      category: "Support",
      link: "/helpline"
    },
    {
      icon: <FaBook />,
      title: "Agri Glossary",
      desc: "Learn common farming terms like Mulching, Drip Irrigation, and more in simple language",
      category: "Learning",
      link: "/glossary"
    },
    {
      icon: <FaShieldAlt />,
      title: "AI Risk Index",
      desc: "Advanced vulnerability scoring for weather, disease, and market instability",
      category: "Analytics",
      link: "/risk-index"
    },
  ];

const stats = [
  { target: 50, suffix: "K+", label: "Farmers Helped" },
  { target: 120, suffix: "+", label: "Crop Types" },
  { target: 98, suffix: "%", label: "Accuracy" },
  { target: 24, suffix: "/7", label: "Support" },
];

const testimonials = [
  { name: "Ramesh Kumar", location: "Maharashtra", text: <><span className="notranslate">Fasal Saathi</span> helped me increase my rice yield by 30% this season!</> },
  { name: "Lakshmi Devi", location: "Tamil Nadu", text: "The weather predictions are accurate. I plan my irrigation accordingly." },
  { name: "Suresh Patel", location: "Gujarat", text: "Best AI farming assistant. Simple to use even for elderly farmers." },
];

// ─── Pre-generated stable bird data (avoids Math.random() on every render) ───
const BIRD_DATA = Array.from({ length: 7 }, (_, i) => ({
  id: i,
  width: 28 + (i * 3) % 12,
  height: 14 + (i * 2) % 8,
  className: `bird bird-${i + 1}`,
}));

// ─── Cloud component at module scope (never remounts during stats animation) ──
const Cloud = ({ className = "" }) => (
  <svg
    className={className}
    width="260"
    height="140"
    viewBox="0 0 260 140"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
  >
    <defs>
      <radialGradient id="cloudBlue" cx="50%" cy="50%" r="50%">
        <stop offset="0%" stopColor="#e0f2ff" stopOpacity="0.98" />
        <stop offset="100%" stopColor="#bfdbfe" stopOpacity="0.85" />
      </radialGradient>
      <radialGradient id="cloudWhite" cx="50%" cy="50%" r="50%">
        <stop offset="0%" stopColor="#f8fafc" stopOpacity="0.97" />
        <stop offset="100%" stopColor="#e2e8f0" stopOpacity="0.80" />
      </radialGradient>
      <filter id="cloudBlur">
        <feGaussianBlur stdDeviation="2" />
      </filter>
    </defs>
    <ellipse cx="130" cy="90" rx="110" ry="42" fill="var(--cloud-fill, url(#cloudBlue))" />
    <ellipse cx="90" cy="72" rx="62" ry="44" fill="var(--cloud-fill, url(#cloudBlue))" />
    <ellipse cx="160" cy="68" rx="54" ry="40" fill="var(--cloud-fill, url(#cloudBlue))" />
    <ellipse cx="120" cy="58" rx="46" ry="36" fill="var(--cloud-fill, url(#cloudBlue))" />
    <ellipse cx="130" cy="90" rx="108" ry="40" fill="none" stroke="var(--cloud-outline, #b6e0ff)" strokeWidth="1.5" strokeOpacity="0.4" />
  </svg>
);

// ─── Birds component at module scope ─────────────────────────────────────────
const Birds = () => (
  <div className="birds-anim-wrap" aria-hidden="true">
    {BIRD_DATA.map(({ id, width, height, className }) => (
      <svg
        key={id}
        className={className}
        width={width}
        height={height}
        viewBox="0 0 28 14"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        <path
          d="M1 7 Q7 1 14 7 Q21 1 27 7"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          fill="none"
        />
      </svg>
    ))}
  </div>
);

// ─── Home component ───────────────────────────────────────────────────────────
export default function Home({ user }) {
  const [statValues, setStatValues] = useState(stats.map(() => 1));

  // Scroll to top on reload
  useEffect(() => {
    if (typeof window === "undefined") return;
    const navigationEntry = performance.getEntriesByType("navigation")?.[0];
    if (navigationEntry?.type === "reload") {
      window.scrollTo(0, 0);
    }
  }, []);

  // Animate stats counters
  useEffect(() => {
    const duration = 1400;
    const startTime = performance.now();
    let rafId = 0;

    const animateStats = (currentTime) => {
      const progress = Math.min((currentTime - startTime) / duration, 1);
      setStatValues(stats.map((stat) => Math.max(1, Math.floor(stat.target * progress))));
      if (progress < 1) {
        rafId = requestAnimationFrame(animateStats);
      }
    };

    rafId = requestAnimationFrame(animateStats);
    return () => cancelAnimationFrame(rafId);
  }, []);

  return (
    <div className="home">
      <WeatherAlertBar />
      <div className="home-weather-relative-wrap">
        <WeatherQuickWidget />
      </div>
      <section className="hero-section highlight-light">
        {/* Real background image layer with overlay for contrast */}
        <div className="hero-bg-image" aria-hidden="true">
          <div className="hero-bg-overlay" />
        </div>

        {/* Cloud Animation Layer */}
        <div className="clouds-anim-wrap" aria-hidden="true">
          {/* Wrapper divs carry the scale so cloud-float keyframes aren't overridden */}
          <div className="cloud-wrapper cloud-wrapper-1"><Cloud className="cloud cloud-1" /></div>
          <div className="cloud-wrapper cloud-wrapper-2"><Cloud className="cloud cloud-2" /></div>
          <div className="cloud-wrapper cloud-wrapper-3"><Cloud className="cloud cloud-3" /></div>
        </div>

        {/* Birds Animation Layer */}
        <Birds />

        <div className="hero-bg">
          <div className="hero-pattern"></div>
        </div>

        <div className="hero-content">
          <div className="hero-copy">
            <div className="hero-badge">
              <FaSeedling /> AI-Powered Farming Assistant
            </div>
            <h1 className="hero-title">
              Smart Farming with <span className="highlight">AI</span>
            </h1>
            <p className="hero-subtitle">
              Get AI-driven crop recommendations, weather insights, and yield predictions
              to maximize your agricultural productivity.
            </p>
            <div className="hero-buttons">
              {/* 
                Primary CTA Link:
                Directs the user to either the AI advisor or the login page.
                We provide a clear aria-label so screen readers understand the context.
              */}
              <Link
                to={user ? "/advisor" : "/login"}
                className="btn-primary"
                aria-label={user ? "Get started with AI Advisor" : "Log in to get started with AI Advisor"}
              >
                <span className="notranslate">Get Started</span>
              </Link>

              {/* 
                Secondary Info Link:
                Explains the app's functionality.
              */}
              <Link 
                to="/how-it-works" 
                className="btn-secondary"
                aria-label="Learn how the AI advisor helps farmers"
              >
                Learn More
              </Link>
            </div>
          </div>
          <div className="hero-stats">
            {stats.map((stat, index) => (
              <div key={index} className="stat-item">
                <span className="stat-number">
                  {statValues[index]}
                  {stat.suffix}
                </span>
                <span className="stat-label">{stat.label}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="hero-visual">
          <Link to="/dashboard" className="floating-card card-1">
            <FaSun className="card-icon" />
            <span>28°C - Sunny</span>
          </Link>
          <Link to="/advisor" className="floating-card card-2">
            <FaSeedling className="card-icon" />
            <span>Yield: +30%</span>
          </Link>
          <Link to="/advisor" className="floating-card card-3">
            <FaHandHoldingWater className="card-icon" />
            <span>Optimal Irrigation</span>
          </Link>
          <Link to="/disease-awareness" className="floating-card card-4">
            <FaBug className="card-icon" />
            <span>Disease Alerts</span>
          </Link>
        </div>
      </section>

      <section className="features-section">
        <div className="section-header">
          <h2>Powerful Features for Modern Farming</h2>
          <p>Everything you need to succeed in agriculture</p>
        </div>
        <div className="features-grid">
          {features.map((feature, index) => (
            <Link to={feature.link || "/"} key={index} className="feature-card-link">
              <div className="feature-card">
                <div className="feature-category">{feature.category}</div>
                <div className="feature-icon">{feature.icon}</div>
                <h3>{feature.title}</h3>
                <p>{feature.desc}</p>
                <div className="feature-card-footer">
                  <span>Learn more</span>
                  <FaArrowRight size={16} />
                </div>
              </div>
            </Link>
          ))}
        </div>
       </section>

       {/* Contributors Section */}
       <section className="contributors-home-section">
          <div className="section-header">
            <h2>🌟 Our Contributors</h2>
            <p className="subtitle">Meet the amazing people behind <span className="notranslate">Fasal Saathi</span></p>
          </div>
         <div className="contributors-home-card">
           <div className="contributors-home-content">
             <div className="contributors-info">
               <h3>Built by the Community</h3>
                <p>
                  <span className="notranslate" translate="no">Fasal Saathi</span> is made possible by passionate developers, designers,
                  and farmers from around the world. Join our open-source community!
                </p>
               <div className="contributors-stats">
                 <div className="stat">
                   <span className="stat-number">25+</span>
                   <span className="stat-label">Contributors</span>
                 </div>
                 <div className="stat">
                   <span className="stat-number">💚</span>
                   <span className="stat-label">Open Source</span>
                 </div>
                 <div className="stat">
                   <span className="stat-number">🌍</span>
                   <span className="stat-label">Global Community</span>
                 </div>
               </div>
                <Link 
                  to="/contributors" 
                  className="btn btn-primary"
                  aria-label="View all people who contributed to this project"
                >
                  <FaUsers aria-hidden="true" /> View All Contributors
                </Link>
             </div>
             <div className="contributors-illustration">
               <FaUsers className="big-icon" aria-hidden="true" />
             </div>
           </div>
         </div>
       </section>

        <section className="how-section">
         <div className="section-header">
           <h2><span className="notranslate">How It Works</span></h2>
           <p>Three simple steps to smarter farming</p>
         </div>
        <div className="steps-container">
          <div className="step">
            <div className="step-number">1</div>
            <h3>Enter Farm Details</h3>
            <p>Input your crop type, area, and farming conditions</p>
          </div>
          <div className="step-connector"></div>
          <div className="step">
            <div className="step-number">2</div>
            <h3>AI Analysis</h3>
            <p>Our ML models analyze your data instantly</p>
          </div>
          <div className="step-connector"></div>
          <div className="step">
            <div className="step-number">3</div>
            <h3>Get Recommendations</h3>
            <p>Receive personalized farming advice</p>
          </div>
        </div>
        <Link 
          to={user ? "/advisor" : "/login"} 
          className="btn-primary"
          aria-label={user ? "Try the AI advisor now" : "Log in to try the AI advisor"}
        >
          Try It Now
        </Link>
      </section>

      <section className="testimonials-section">
        <div className="section-header">
          <h2>What Farmers Say</h2>
          <p>Real experiences from real farmers</p>
        </div>
        <div className="testimonials-grid">
          {testimonials.map((testimonial, index) => (
            <div key={index} className="testimonial-card">
              <FaQuoteLeft className="quote-icon" />
              <p className="testimonial-text">{testimonial.text}</p>
              <div className="testimonial-author">
                 <div className="author-avatar">{testimonial.name[0]}</div>
                <div className="author-info">
                  <span className="author-name"><span className="notranslate">{testimonial.name}</span></span>
                  <span className="author-location">{testimonial.location}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="cta-section">
        <h2>Ready to Transform Your Farm?</h2>
        <p>Join thousands of farmers already benefiting from AI-powered agriculture</p>
        <Link 
          to={user ? "/advisor" : "/login"} 
          className="btn-primary"
          aria-label={user ? "Start a free consultation with the AI" : "Log in to start a free consultation"}
        >
          Start Free Consultation
        </Link>
      </section>

      <footer className="home-footer">
        <div className="footer-content">
          <div className="footer-grid">
            <div className="footer-section">
               <div className="footer-brand">
                 <FaSeedling className="footer-logo" />
                 <span className="notranslate">Fasal Saathi</span>
               </div>
              <p className="footer-description">
                AI-powered agricultural advisor helping farmers with crop planning,
                weather insights, irrigation, and yield optimization.
              </p>
              <div className="footer-contact">
                <FaPhoneAlt />
                <span>+91 98765 43210</span>
              </div>
            </div>
            <div className="footer-section">
              {/* 
                ACCESSIBILITY BEST PRACTICE: Footer Navigation
                Using clear, descriptive link text and aria-labels for footer links.
              */}
               <h4>Quick Links</h4>
              <Link to="/" aria-label="Go to Home Page"><span className="notranslate">Home</span></Link>
              <Link to="/advisor" aria-label="Consult the AI Advisor"><span className="notranslate">Advisor</span></Link>
              <Link to="/how-it-works" aria-label="How Fasal Saathi helps you"><span className="notranslate">How It Works</span></Link>
              <Link to="/schemes" aria-label="View Government Schemes for farmers"><span className="notranslate">Govt Schemes</span></Link>
              <Link to="/dashboard" aria-label="Go to your farming dashboard"><span className="notranslate">Dashboard</span></Link>
              <Link to="/calendar" aria-label="View your farming activity calendar"><span className="notranslate">Activity Calendar</span></Link>
              <Link to="/market-prices" aria-label="Check latest market prices for crops"><span className="notranslate">Market Prices</span></Link>
              <Link to="/community" aria-label="Join the community discussion"><span className="notranslate">Community</span></Link>
              <Link to="/share-feedback" aria-label="Share your thoughts with us"><span className="notranslate">Share Feedback</span></Link>
            </div>
            <div className="footer-section">
              <h4>Resources</h4>
              <Link to="/crop-guide" aria-label="View the Crop Guide"><span className="notranslate">Crop Guide</span></Link>
              <Link to="/weather" aria-label="Check weather updates"><span className="notranslate">Weather Updates</span></Link>
              <Link to="/soil-analysis" aria-label="Get soil analysis insights"><span className="notranslate">Soil Analysis</span></Link>
              <Link to="/faq" aria-label="Frequently Asked Questions"><span className="notranslate">FAQs</span></Link>
            </div>
             <div className="footer-section">
               <h4>Company</h4>
               <Link to="/about" aria-label="Learn about Fasal Saathi"><span className="notranslate">About Us</span></Link>
               <Link to="/contact" aria-label="Contact our support team"><span className="notranslate">Contact</span></Link>
               <Link to="/privacy-policy" aria-label="Read our Privacy Policy"><span className="notranslate">Privacy Policy</span></Link>
               <Link to="/terms" aria-label="Read our Terms of Service"><span className="notranslate">Terms of Service</span></Link>
             </div>
          </div>
          <div className="footer-bottom">
            <div className="footer-socials">
              <FaGlobe />
              <span>Available Across India</span>
            </div>
           <p className="footer-copyright">
              © 2026 <span className="notranslate" translate="no">Fasal Saathi</span>. All rights reserved. MIT Licensed.
           </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
