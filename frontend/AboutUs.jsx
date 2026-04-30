import { useState } from "react";
import { Link } from "react-router-dom";
import {
  FaLeaf, FaBullseye, FaEye, FaHeart, FaSeedling,
  FaBrain, FaSun, FaHandHoldingWater, FaChartLine,
  FaShieldAlt, FaRocket, FaGlobe,
  FaGithub, FaLinkedin, FaTwitter, FaArrowRight,
  FaCrown,
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
  { year: "2022", title: "Idea Born", desc: "Fasal Saathi started as a college project to solve real problems faced by Indian farmers.", emoji: "💡" },
  { year: "2023", title: "First Prototype", desc: "Launched MVP with crop recommendations and basic weather alerts for 3 states.", emoji: "🚀" },
  { year: "2024", title: "Multi-language Launch", desc: "Expanded to 12 Indian languages and onboarded 10,000+ farmers.", emoji: "🌍" },
  { year: "2025", title: "AI Integration", desc: "Integrated machine learning models for yield prediction and soil analysis.", emoji: "🤖" },
  { year: "2026", title: "Growing Strong", desc: "50,000+ farmers, community features, and expanding to more regions.", emoji: "🌱" },
];

const TEAM = [
    { name: "Esha Jha", role: "Owner & Backend+Frontend Developer", github: "#", linkedin: "#", twitter: "#", initial: "E", isOwner: true },
    { name: "Eshita Jha", role: "Owner & Research Analyst", github: "#", linkedin: "#", twitter: "#", initial: "E", isOwner: true },
    { name: "Shreyshi Sriwastav", role: "Owner & ML Engineer", github: "#", linkedin: "#", twitter: "#", initial: "S", isOwner: true },
    { name: "Annsu Choudhary", role: "Owner & Backend Developer", github: "#", linkedin: "#", twitter: "#", initial: "A", isOwner: true },
    { name: "Devanshi Singh", role: "Owner & Research Analyst", github: "#", linkedin: "#", twitter: "#", initial: "D", isOwner: true },
    { name: "Bhargavi Kanojiya", role: "Owner & ML Engineer", github: "#", linkedin: "#", twitter: "#", initial: "B", isOwner: true },
];

const FARM_ICONS = [];

export default function AboutUs() {
  const [openFaq, setOpenFaq] = useState(null);
  void openFaq; void setOpenFaq;

  return (
    <div className="about-page">
      <div className="about-blob about-blob-1" />
      <div className="about-blob about-blob-2" />

      {/* ── HERO — editorial style like reference ── */}
      <section className="about-hero-editorial">
        <div className="about-hero-left">
          <h1 className="about-hero-title">
            Hey, we are<br /><span className="about-hero-highlight">Fasal Saathi</span>
          </h1>
          <p className="about-hero-body">
            We're an <strong>AI-powered agricultural platform</strong> built to empower Indian farmers with smart technology, real-time insights, and practical knowledge. Our goal is to make farming more efficient and profitable by helping farmers choose the right crops, monitor soil health, access weather updates, and track live market prices — all in one simple, multilingual platform that works even in low-connectivity areas.
          </p>
          <p className="about-hero-body">
            We are driven by a mission to make <strong>agricultural intelligence accessible to every farmer</strong>, regardless of land size or literacy level. By combining traditional farming wisdom with modern AI, we aim to reduce losses, improve yields, and increase income while ensuring <strong>trust, privacy, and inclusivity</strong> at every step.
          </p>
          <div className="about-hero-actions">
            <Link to="/advisor" className="about-btn-primary"><FaRocket /> Try the Advisor</Link>
            <Link to="/contact" className="about-btn-outline"><FaArrowRight /> Contact Us</Link>
          </div>
        </div>

        {/* Right — tractor image */}
        <div className="about-hero-right">
          <div className="about-hero-image-wrap">
            <img
              src="/tractor.png"
              alt="Modern farming tractor"
              className="about-tractor-img"
            />
          </div>
        </div>
      </section>

      {/* ── FARM STRIP removed ── */}

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
              {[
                { label: "🌾 Wheat", w: 80 },
                { label: "🌽 Maize", w: 65 },
                { label: "🍅 Tomato", w: 90 },
                { label: "🌿 Spinach", w: 55 },
                { label: "🥔 Potato", w: 75 },
              ].map((item, i) => (
                <div key={i} className="visual-bar-labeled">
                  <span className="visual-bar-label">{item.label}</span>
                  <div className="visual-bar-row">
                    <div className="visual-bar" style={{ width: `${item.w}%`, animationDelay: `${i * 0.15}s` }} />
                  </div>
                  <span className="visual-bar-pct">{item.w}%</span>
                </div>
              ))}
            </div>
            <div className="visual-footer">
              <span className="visual-tag">🚜 Tractor Ready</span>
              <span className="visual-tag">💧 Irrigated</span>
              <span className="visual-tag">�️ Good Weather</span>
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
              <div className="timeline-dot">{t.emoji}</div>
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
                <h3><span className="notranslate">{m.name}</span></h3>
                <p><span className="notranslate">{m.role}</span></p>
               {m.isOwner && (
                 <div className="founder-badge">
                   <FaCrown /> Owner
                 </div>
               )}
               {/* TODO: Replace href="#" with real social profile links */}
               <div className="team-socials">
                 {m.github && m.github !== "#" && <a href={m.github} aria-label="GitHub" target="_blank" rel="noopener noreferrer"><FaGithub /></a>}
                 {m.linkedin && m.linkedin !== "#" && <a href={m.linkedin} aria-label="LinkedIn" target="_blank" rel="noopener noreferrer"><FaLinkedin /></a>}
                 {m.twitter && m.twitter !== "#" && <a href={m.twitter} aria-label="Twitter" target="_blank" rel="noopener noreferrer"><FaTwitter /></a>}
               </div>
             </div>
           ))}
        </div>
      </div>

      {/* CTA */}
      <section className="about-cta">
        <div className="about-cta-inner">
          <div className="about-cta-farm-row">
            <span>🌾</span><span>🚜</span><span>🐄</span><span>🌽</span><span>👩‍🌾</span>
          </div>
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
