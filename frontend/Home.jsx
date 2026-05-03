import React from "react";
import { Link } from "react-router-dom";
import {
  FaBrain, FaChartLine, FaHandHoldingWater, FaLeaf, FaLock,
  FaGlobe, FaCalendarAlt, FaBug, FaArrowRight, FaBook,
  FaShieldAlt, FaSun, FaFlask, FaPhoneAlt, FaQuoteLeft,
  FaSeedling, FaUsers
} from "react-icons/fa";

import {
  LineChart, Line, XAxis, Tooltip, ResponsiveContainer
} from "recharts";

import WeatherAlertBar from "./weather/WeatherAlertBar";
import WeatherQuickWidget from "./weather/WeatherQuickWidget";
import "./Home.css";

/* =========================
   DATA CONFIG
========================= */

const features = [
  { icon: <FaBrain />, title: "AI Predictions", desc: "ML yield insights", category: "Analytics", link: "/advisor" },
  { icon: <FaSun />, title: "Weather Insights", desc: "Forecast + alerts", category: "Monitoring", link: "/dashboard" },
  { icon: <FaHandHoldingWater />, title: "Smart Irrigation", desc: "Save water smartly", category: "Optimization", link: "/advisor" },
  { icon: <FaFlask />, title: "Soil Analysis", desc: "Know your soil", category: "Monitoring", link: "/soil-guide" },
  { icon: <FaLeaf />, title: "Crop Recommendation", desc: "Best crop suggestions", category: "Recommendations", link: "/crop-guide" },
  { icon: <FaBug />, title: "Disease Detection", desc: "Identify crop issues", category: "Education", link: "/disease-awareness" },
  { icon: <FaCalendarAlt />, title: "Crop Planner", desc: "Plan seasons", category: "Planning", link: "/crop-planner" },
  { icon: <FaShieldAlt />, title: "Risk Index", desc: "AI risk score", category: "Analytics", link: "/risk-index" },
];

const stats = [
  { target: 50, suffix: "K+", label: "Farmers Helped" },
  { target: 120, suffix: "+", label: "Crop Types" },
  { target: 98, suffix: "%", label: "Accuracy" },
  { target: 24, suffix: "/7", label: "Support" },
];

const cropData = [
  { year: "2019", rice: 2722, wheat: 3440 },
  { year: "2020", rice: 2717, wheat: 3521 },
  { year: "2021", rice: 2798, wheat: 3537 },
  { year: "2022", rice: 2838, wheat: 3521 },
  { year: "2023", rice: 2882, wheat: 3559 },
];

const testimonials = [
  { name: "Ramesh Kumar", location: "Maharashtra", text: "Yield increased by 30%!" },
  { name: "Lakshmi Devi", location: "Tamil Nadu", text: "Weather predictions are accurate." },
  { name: "Suresh Patel", location: "Gujarat", text: "Best AI farming assistant." },
];

/* =========================
   MAIN COMPONENT
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
          d="M2 7 C5 2 9 2 14 7 C19 2 23 2 26 7"
          stroke="#2d3748"
          strokeWidth="3"
          strokeLinecap="round"
          fill="none"
        />
        <path
          d="M8 5 L10 3"
          stroke="#2d3748"
          strokeWidth="2.5"
          strokeLinecap="round"
        />
      </svg>
    ))}
  </div>
);

export default function Home({ user }) {

  const [statValues, setStatValues] = React.useState([0, 0, 0, 0]);
  const [filter, setFilter] = React.useState("All");
  const [activeTestimonial, setActiveTestimonial] = React.useState(0);

  /* -------- Stats Animation -------- */
  React.useEffect(() => {
    const interval = setInterval(() => {
      setStatValues(prev =>
        prev.map((val, i) => {
          const target = stats[i].target;
          return val < target ? val + 2 : target;
        })
      );
    }, 40);
    return () => clearInterval(interval);
  }, []);

  /* -------- Testimonials Slider -------- */
  React.useEffect(() => {
    const interval = setInterval(() => {
      setActiveTestimonial(prev => (prev + 1) % testimonials.length);
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  const filteredFeatures =
    filter === "All"
      ? features
      : features.filter(f => f.category === filter);

  const t = testimonials[activeTestimonial];

  return (
    <div className="home">

      <WeatherAlertBar />
      <WeatherQuickWidget />

      {/* ================= HERO ================= */}
      <section className="hero-section">

        <div className="hero-content">
          <h1>
            {user ? `Welcome ${user.name || "Farmer"} 👋` : "Smart Farming with AI"}
          </h1>

          <p>
            AI-powered agriculture insights, weather tracking, and smart recommendations.
          </p>

          <div className="hero-buttons">
            <Link to={user ? "/advisor" : "/login"} className="btn-primary">
              Get Started
            </Link>
            <Link to="/how-it-works" className="btn-secondary">
              Learn More
            </Link>
          </div>

          {/* Stats */}
          <div className="hero-stats">
            {stats.map((s, i) => (
              <div key={i} className="stat">
                <h2>{statValues[i]}{s.suffix}</h2>
                <p>{s.label}</p>
              </div>
            ))}
          </div>
        </div>

      </section>

      {/* ================= CHART ================= */}
      <section className="insights-section">
        <h2>📊 Crop Trends</h2>

        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={cropData}>
            <XAxis dataKey="year" />
            <Tooltip />
            <Line dataKey="rice" strokeWidth={3} />
            <Line dataKey="wheat" strokeWidth={3} />
          </LineChart>
        </ResponsiveContainer>
      </section>

      {/* ================= AI INSIGHTS ================= */}
      <section className="ai-insights">
        <h2>🤖 AI Insights</h2>

        <div className="insight-grid">
          <div className="insight-card">🌾 Wheat highest production</div>
          <div className="insight-card">📈 Urad fastest growth</div>
          <div className="insight-card">⚠️ Ragi declining</div>
        </div>
      </section>

      {/* ================= FEATURES ================= */}
      <section className="features-section">
        <h2>Features</h2>

        {/* FILTER */}
        <div className="filters">
          {["All", "Analytics", "Monitoring", "Planning"].map(cat => (
            <button key={cat} onClick={() => setFilter(cat)}>
              {cat}
            </button>
          ))}
        </div>

        {/* GRID */}
        <div className="features-grid">
          {filteredFeatures.map((f, i) => (
            <Link key={i} to={f.link} className="feature-card">
              <div className="icon">{f.icon}</div>
              <h3>{f.title}</h3>
              <p>{f.desc}</p>
              <FaArrowRight />
            </Link>
          ))}
        </div>
      </section>

      {/* ================= HOW IT WORKS ================= */}
      <section className="how-section">
        <h2>How It Works</h2>

        <div className="steps">
          <div>1. Enter Farm Details</div>
          <div>2. AI Analysis</div>
          <div>3. Get Recommendations</div>
        </div>
      </section>

      {/* ================= TESTIMONIAL ================= */}
      <section className="testimonial-section">
        <FaQuoteLeft />
        <p>{t.text}</p>
        <h4>{t.name} ({t.location})</h4>
      </section>

      {/* ================= CONTRIBUTORS ================= */}
      <section className="contributors">
        <h2>Our Contributors</h2>

        <div className="contributors-box">
          <FaUsers size={40} />
          <p>25+ Developers contributing globally</p>
          <Link to="/contributors" className="btn-primary">
            View Contributors
          </Link>
        </div>
      </section>

      {/* ================= CTA ================= */}
      <section className="cta-section">
        <h2>Ready to Transform Farming?</h2>

        <Link to="/advisor" className="btn-primary">
          Start Now
        </Link>
      </section>

      {/* ================= STICKY CTA ================= */}
      <div className="sticky-cta">
        <Link to="/advisor">🚀 AI Advisor</Link>
      </div>

      {/* ================= FOOTER ================= */}
      <footer className="footer">
        <div className="footer-grid">

          <div>
            <FaSeedling />
            <h3>Fasal Saathi</h3>
            <p>AI-powered farming assistant</p>
          </div>

          <div>
            <h4>Links</h4>
            <Link to="/">Home</Link>
            <Link to="/advisor">Advisor</Link>
            <Link to="/dashboard">Dashboard</Link>
          </div>

          <div>
            <h4>Contact</h4>
            <p><FaPhoneAlt /> +91 98765 43210</p>
            <p><FaGlobe /> India</p>
          </div>

        </div>

        <p className="copyright">
          © 2026 Fasal Saathi
        </p>
      </footer>

    </div>
  );
}  

const Cloud = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ width: '100px', height: '100px', opacity: 0.8 }}>
    <path d="M17.5 19a4.5 4.5 0 0 0 2.9-7.7A7 7 0 0 0 7 9a5.2 5.2 0 0 0-4 8.5" />
    <path d="M7 19h10.5" />
  </svg>
);

const Birds = () => (
  <div className="birds-anim-wrap" aria-hidden="true" style={{ position: 'absolute', top: '10%', left: '0', width: '100%', height: '100%', pointerEvents: 'none' }}>
    <svg className="bird bird-1" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ width: '30px', height: '30px', position: 'absolute', left: '20%', animation: 'fly 10s linear infinite' }}>
      <path d="M2 12s3-3 6-3 4 3 4 3" />
      <path d="M12 12s3-3 6-3 4 3 4 3" />
    </svg>
  </div>
);  
