import React from "react";
import { Link } from "react-router-dom";
import {
  FaBrain, FaChartLine, FaHandHoldingWater, FaLeaf, FaLock,
  FaGlobe, FaCalendarAlt, FaUsers, FaBug, FaArrowRight,
  FaBook, FaShieldAlt, FaSun, FaFlask, FaPhoneAlt,
  FaQuoteLeft, FaSeedling, FaChevronUp
} from "react-icons/fa";

import {
  LineChart, Line, XAxis, Tooltip, ResponsiveContainer
} from "recharts";

import WeatherAlertBar from "./weather/WeatherAlertBar";
import WeatherQuickWidget from "./weather/WeatherQuickWidget";
import "./Home.css";

/* ================= DATA ================= */

const cropData = [
  { year: "2019", Rice: 2722, Wheat: 3440 },
  { year: "2020", Rice: 2717, Wheat: 3521 },
  { year: "2021", Rice: 2798, Wheat: 3537 },
  { year: "2022", Rice: 2838, Wheat: 3521 },
  { year: "2023", Rice: 2882, Wheat: 3559 },
];

const features = [
  { icon: <FaBrain />, title: "AI Predictions", desc: "ML crop insights", category: "Analytics", link: "/advisor" },
  { icon: <FaSun />, title: "Weather Insights", desc: "Forecast & alerts", category: "Monitoring", link: "/dashboard" },
  { icon: <FaHandHoldingWater />, title: "Smart Irrigation", desc: "Water optimization", category: "Optimization", link: "/advisor" },
  { icon: <FaFlask />, title: "Soil Analysis", desc: "Soil health tracking", category: "Monitoring", link: "/soil-guide" },
  { icon: <FaLeaf />, title: "Crop Recommendation", desc: "Best crop selection", category: "Recommendations", link: "/crop-guide" },
  { icon: <FaBug />, title: "Disease Awareness", desc: "Detect crop diseases", category: "Education", link: "/disease-awareness" },
];

const stats = [
  { target: 50, suffix: "K+", label: "Farmers Helped" },
  { target: 120, suffix: "+", label: "Crop Types" },
  { target: 98, suffix: "%", label: "Accuracy" },
  { target: 24, suffix: "/7", label: "Support" },
];

const testimonials = [
  { name: "Ramesh Kumar", location: "Maharashtra", text: "Yield increased by 30%" },
  { name: "Lakshmi Devi", location: "Tamil Nadu", text: "Weather predictions are accurate" },
  { name: "Suresh Patel", location: "Gujarat", text: "Best AI farming assistant" },
];

/* ================= COMPONENT ================= */

export default function Home({ user }) {

  const [statValues, setStatValues] = React.useState([0,0,0,0]);
  const [filter, setFilter] = React.useState("All");
  const [showTop, setShowTop] = React.useState(false);

  /* Stats Animation */
  React.useEffect(() => {
    const interval = setInterval(() => {
      setStatValues(prev =>
        prev.map((val, i) => {
          const target = stats[i].target;
          return val < target ? val + Math.ceil(target/25) : target;
        })
      );
    }, 60);
    return () => clearInterval(interval);
  }, []);

  /* Scroll button */
  React.useEffect(() => {
    const handleScroll = () => setShowTop(window.scrollY > 300);
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const filteredFeatures =
    filter === "All"
      ? features
      : features.filter(f => f.category === filter);

  return (
    <div className="home">

      <WeatherAlertBar />
      <WeatherQuickWidget />

      {/* HERO */}
      <section className="hero-section">
        <h1>
          {user ? `Welcome ${user.name || "Farmer"} 👋` : "Smart Farming with AI"}
        </h1>
        <p>AI-powered agriculture insights & recommendations</p>

        <div className="hero-buttons">
          <Link to={user ? "/advisor" : "/login"} className="btn-primary">
            Get Started
          </Link>
          <Link to="/how-it-works" className="btn-secondary">
            Learn More
          </Link>
        </div>

        <div className="hero-stats">
          {stats.map((s,i) => (
            <div key={i}>
              <h2>{statValues[i]}{s.suffix}</h2>
              <p>{s.label}</p>
            </div>
          ))}
        </div>
      </section>

      {/* CHART */}
      <section className="insights-section">
        <h2>📊 Crop Trends</h2>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={cropData}>
            <XAxis dataKey="year" />
            <Tooltip />
            <Line dataKey="Rice" strokeWidth={3} />
            <Line dataKey="Wheat" strokeWidth={3} />
          </LineChart>
        </ResponsiveContainer>
      </section>

      {/* AI INSIGHTS */}
      <section className="ai-insights">
        <h2>🤖 AI Insights</h2>
        <div className="insight-grid">
          <div>🌾 Wheat highest production</div>
          <div>📈 Urad fastest growth</div>
          <div>⚠️ Ragi declining</div>
        </div>
      </section>

      {/* FEATURES */}
      <section className="features-section">
        <h2>Features</h2>

        <div className="feature-filters">
          {["All","Analytics","Monitoring","Education"].map(cat => (
            <button key={cat} onClick={()=>setFilter(cat)}>
              {cat}
            </button>
          ))}
        </div>

        <div className="features-grid">
          {filteredFeatures.map((f,i)=>(
            <Link key={i} to={f.link} className="feature-card">
              {f.icon}
              <h3>{f.title}</h3>
              <p>{f.desc}</p>
              <FaArrowRight />
            </Link>
          ))}
        </div>
      </section>

      {/* TESTIMONIAL */}
      <section className="testimonials-section">
        <h2>What Farmers Say</h2>
        {testimonials.map((t,i)=>(
          <div key={i} className="testimonial-card">
            <FaQuoteLeft />
            <p>{t.text}</p>
            <h4>{t.name} ({t.location})</h4>
          </div>
        ))}
      </section>

      {/* CTA */}
      <section className="cta-section">
        <h2>Ready to Transform Farming?</h2>
        <Link to="/advisor" className="btn-primary">
          Start Now
        </Link>
      </section>

      {/* STICKY CTA */}
      <div className="sticky-cta">
        <Link to="/advisor">🚀 AI Advice</Link>
      </div>

      {/* SCROLL TOP */}
      {showTop && (
        <button className="scroll-top"
          onClick={()=>window.scrollTo({top:0,behavior:"smooth"})}>
          <FaChevronUp />
        </button>
      )}

      {/* FOOTER */}
      <footer className="home-footer">
        <h3><FaSeedling /> Fasal Saathi</h3>
        <p>AI-powered farming assistant</p>
        <p><FaPhoneAlt /> +91 98765 43210</p>
        <p><FaGlobe /> India</p>
      </footer>

    </div>
  );
}
