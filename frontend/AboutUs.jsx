import { useState } from "react";
import { Link } from "react-router-dom";
import {
  FaLeaf, FaBullseye, FaEye, FaHeart, FaSeedling,
  FaBrain, FaSun, FaHandHoldingWater, FaChartLine,
  FaShieldAlt, FaRocket, FaGlobe,
  FaGithub, FaLinkedin, FaTwitter, FaArrowRight,
} from "react-icons/fa";
import "./AboutUs.css";

const VALUES = [
  { icon: <FaHeart />, title: "Farmer First", desc: "Every feature we build starts with one question: does this make a farmer's life better?", color: "#ef4444" },
  { icon: <FaBrain />, title: "AI for Good", desc: "We harness machine learning to give small farmers the same edge as large agribusinesses.", color: "#6366f1" },
  { icon: <FaGlobe />, title: "Inclusive Access", desc: "Available in 12 Indian languages so no farmer is left behind due to a language barrier.", color: "#10b981" },
  { icon: <FaShieldAlt />, title: "Trust & Privacy", desc: "Your farm data is yours. We never sell or misuse the information you share with us.", color: "#f59e0b" },
];

const FEATURES = [
  { icon: <FaBrain />, title: "AI Crop Advisor", desc: "Smart recommendations based on soil, season, and location." },
  { icon: <FaSun />, title: "Weather Insights", desc: "Real-time forecasts and custom alerts for your farm." },
  { icon: <FaHandHoldingWater />, title: "Smart Irrigation", desc: "Optimize water usage with AI-driven guidance." },
  { icon: <FaChartLine />, title: "Market Prices", desc: "Live mandi prices to help you sell at the right time." },
  { icon: <FaSeedling />, title: "Soil Analysis", desc: "Comprehensive soil health and nutrient monitoring." },
  { icon: <FaRocket />, title: "Offline Mode", desc: "Core features work without internet — syncs when back online." },
];

const TIMELINE = [
  { year: "2022", title: "Idea Born", desc: "Fasal Saathi started as a college project to solve real problems faced by Indian farmers." },
  { year: "2023", title: "First Prototype", desc: "Launched MVP with crop recommendations and basic weather alerts for 3 states." },
  { year: "2024", title: "Multi-language Launch", desc: "Expanded to 12 Indian languages and onboarded 10,000+ farmers." },
  { year: "2025", title: "AI Integration", desc: "Integrated machine learning models for yield prediction and soil analysis." },
  { year: "2026", title: "Growing Strong", desc: "50,000+ farmers, community features, and expanding to more regions." },
];

const TEAM = [
  { name: "Esha Jha", role: "Founder & Lead Developer", github: "#", linkedin: "#", twitter: "#", initial: "E" },
  { name: "Ankit Kumar", role: "Frontend Engineer", github: "#", linkedin: "#", twitter: "#", initial: "A" },
  { name: "Priya Sharma", role: "AI/ML Engineer", github: "#", linkedin: "#", twitter: "#", initial: "P" },
  { name: "Rahul Verma", role: "Backend Developer", github: "#", linkedin: "#", twitter: "#", initial: "R" },
];

export default function AboutUs() {
  const [openFaq, setOpenFaq] = useState(null);
  void openFaq; void setOpenFaq;

  return (
    <div className="about-page">
      <div className="about-blob about-blob-1" />
      <div className="about-blob about-blob-2" />

      {/* HERO */}
      <section className="about-hero">
        <div className="about-hero-badge"><FaLeaf /> About Fasal Saathi</div>
        <h1>Empowering Farmers<br /><span className="about-hero-highlight">With Smart Technology</span></h1>
        <p>Fasal Saathi is an AI-powered agricultural platform built to give every Indian farmer access to the tools, insights, and knowledge they need to grow better and earn more.</p>
        <div className="about-hero-actions">
          <Link to="/advisor" className="about-btn-primary"><FaRocket /> Try the Advisor</Link>
          <Link to="/contact" className="about-btn-outline"><FaArrowRight /> Contact Us</Link>
        </div>
        <div className="about-hero-scroll">
          <span>Scroll to explore</span>
          <div className="scroll-mouse"><div className="scroll-wheel" /></div>
        </div>
      </section>

      {/* MISSION & VISION */}
      <div className="about-section">
        <div className="about-mv-grid">
          <div className="about-mv-card mission">
            <div className="about-mv-icon"><FaBullseye /></div>
            <h2>Our Mission</h2>
            <p>To democratize agricultural intelligence by providing every Indian farmer — regardless of land size, literacy, or language — with AI-powered tools that improve crop yield, reduce losses, and increase income.</p>
          </div>
          <div className="about-mv-card vision">
            <div className="about-mv-icon"><FaEye /></div>
            <h2>Our Vision</h2>
            <p>A future where no farmer makes a decision without data. Where technology bridges the gap between traditional farming wisdom and modern science, creating a resilient and prosperous agricultural India.</p>
          </div>
        </div>
      </div>

      {/* WHAT WE DO */}
      <div className="about-section about-intro">
        <div className="about-intro-text">
          <div className="about-section-badge">What We Do</div>
          <h2>Your Digital Farming Companion</h2>
          <p>Fasal Saathi (meaning "Crop Companion") is a free, multilingual platform that combines AI, real-time data, and agricultural expertise to help farmers make smarter decisions every day.</p>
          <p>From choosing the right crop for your soil to knowing when to irrigate, from checking live mandi prices to getting government scheme updates — Fasal Saathi is the one-stop solution for modern Indian farming.</p>
          <Link to="/crop-guide" className="about-btn-primary" style={{ width: "fit-content" }}>
            <FaSeedling /> Explore Crop Guide
          </Link>
        </div>
        <div className="about-intro-visual">
          <div className="about-visual-card">
            <div className="visual-card-header">
              <FaBrain className="visual-icon" />
              <span>AI Crop Advisor</span>
            </div>
            <div className="visual-bars">
              {[80, 65, 90, 55, 75].map((w, i) => (
                <div key={i} className="visual-bar-row">
                  <div className="visual-bar" style={{ width: `${w}%`, animationDelay: `${i * 0.15}s` }} />
                </div>
              ))}
            </div>
            <div className="visual-footer">
              <span className="visual-tag">🌾 Wheat</span>
              <span className="visual-tag">🌽 Maize</span>
              <span className="visual-tag">🍅 Tomato</span>
            </div>
          </div>
        </div>
      </div>

      {/* VALUES */}
      <div className="about-section">
        <div className="about-section-header">
          <div className="about-section-badge">Our Values</div>
          <h2>What Drives Us</h2>
          <p>The principles that guide every decision we make at Fasal Saathi.</p>
        </div>
        <div className="about-values-grid">
          {VALUES.map((v, i) => (
            <div className="about-value-card" key={i} style={{ "--value-color": v.color }}>
              <div className="about-value-icon" style={{ background: `${v.color}18`, color: v.color }}>{v.icon}</div>
              <h3>{v.title}</h3>
              <p>{v.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* FEATURES */}
      <div className="about-features-section">
        <div className="about-section-header">
          <div className="about-section-badge">Platform Features</div>
          <h2>Everything a Farmer Needs</h2>
          <p>A complete toolkit built for the realities of Indian agriculture.</p>
        </div>
        <div className="about-features-grid">
          {FEATURES.map((f, i) => (
            <div className="about-feature-card" key={i}>
              <div className="about-feature-icon">{f.icon}</div>
              <h3>{f.title}</h3>
              <p>{f.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* TIMELINE */}
      <div className="about-section">
        <div className="about-section-header">
          <div className="about-section-badge">Our Journey</div>
          <h2>How We Got Here</h2>
          <p>From a college project to a platform serving farmers across India.</p>
        </div>
        <div className="about-timeline">
          <div className="timeline-line" />
          {TIMELINE.map((t, i) => (
            <div className={`about-timeline-item ${i % 2 === 0 ? "left" : "right"}`} key={i}>
              <div className="timeline-dot" />
              <div className="timeline-card">
                <span className="timeline-year">{t.year}</span>
                <h3>{t.title}</h3>
                <p>{t.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* TEAM */}
      <div className="about-section">
        <div className="about-section-header">
          <div className="about-section-badge">The Team</div>
          <h2>People Behind Fasal Saathi</h2>
          <p>A passionate group of developers and agri-tech enthusiasts.</p>
        </div>
        <div className="about-team-grid">
          {TEAM.map((m, i) => (
            <div className="about-team-card" key={i}>
              <div className="team-avatar">{m.initial}</div>
              <h3>{m.name}</h3>
              <p>{m.role}</p>
              {/* TODO: Replace href="#" with real social profile links */}
              <div className="team-socials">
                <a href={m.github} aria-label="GitHub" target="_blank" rel="noopener noreferrer"><FaGithub /></a>
                <a href={m.linkedin} aria-label="LinkedIn" target="_blank" rel="noopener noreferrer"><FaLinkedin /></a>
                <a href={m.twitter} aria-label="Twitter" target="_blank" rel="noopener noreferrer"><FaTwitter /></a>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* CTA */}
      <section className="about-cta">
        <div className="about-cta-inner">
          <FaLeaf className="about-cta-icon" />
          <h2>Ready to Farm Smarter?</h2>
          <p>Join farmers already using Fasal Saathi to grow better crops and earn more.</p>
          <div className="about-cta-actions">
            <Link to="/crop-guide" className="about-btn-primary">Get Started Free</Link>
            <Link to="/contact" className="about-btn-outline-white">Talk to Us</Link>
          </div>
        </div>
      </section>
    </div>
  );
}
