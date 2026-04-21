import React, { useEffect } from "react";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";

import Advisor from "./Advisor";
import Home from "./Home";
import Resources from "./Resources";
import CropGuide from "./CropGuide";
import CropProfitCalculator from "./CropProfitCalculator";
import {
  FaHome,
  FaComments,
  FaInfoCircle,
  FaLeaf,
  FaBars,
  FaTimes,
  FaCalculator,
} from "react-icons/fa";
import How from "./How";
import { NavLink } from "react-router-dom";
import { useUiStore } from "./stores/uiStore";

import "./App.css";
import "./themes/sunlight.css";

/* ---------------- LANGUAGE ---------------- */

const LANGUAGE_OPTIONS = [
  { value: "en", label: "🌍 English" },
  { value: "hi", label: "🇮🇳 हिंदी" },
  { value: "mr", label: "🇮🇳 मराठी" },
  { value: "bn", label: "🇮🇳 বাংলা" },
  { value: "ta", label: "🇮🇳 தமிழ்" },
  { value: "te", label: "🇮🇳 తెలుగు" },
  { value: "gu", label: "🇮🇳 ગુજરાતી" },
  { value: "pa", label: "🇮🇳 ਪੰਜਾਬੀ" },
  { value: "kn", label: "🇮🇳 ಕನ್ನಡ" },
  { value: "ml", label: "🇮🇳 മലയാളം" },
  { value: "or", label: "🇮🇳 ଓଡ଼ିଆ" },
  { value: "as", label: "🇮🇳 অসমীয়া" },
];

const getInitialLanguage = () => {
  try {
    const stored = localStorage.getItem("preferredLanguage");
    return LANGUAGE_OPTIONS.some((l) => l.value === stored)
      ? stored
      : "en";
  } catch {
    return "en";
  }
};

/* ---------------- GOOGLE TRANSLATE CONTROL ---------------- */

const applyGoogleTranslate = (lang) => {
  const el = document.querySelector(".goog-te-combo");
  if (!el) return false;

  el.value = lang;
  el.dispatchEvent(new Event("change"));
  return true;
};

const syncLanguage = (lang, setLang) => {
  setLang(lang);
  localStorage.setItem("preferredLanguage", lang);
  applyGoogleTranslate(lang);
};

/* APP COMPONENT */

function App() {
  const {
    preferredLang,
    setPreferredLang,
    isNavOpen,
    setNavOpen,
    theme,
    setTheme,
    isAccessibilityMode,
    farmerName,
    setFarmerName,
    inputName,
    setInputName,
  } = useUiStore();

  /* Apply theme changes */
  useEffect(() => {
    document.documentElement.classList.toggle("theme-dark", theme === "dark");
  }, [theme]);

  /* Apply accessibility / sunlight mode */
  useEffect(() => {
    document.documentElement.classList.toggle("sunlight", isAccessibilityMode);
  }, [isAccessibilityMode]);

  /* Apply language changes with Google Translate */
  useEffect(() => {
    if (applyGoogleTranslate(preferredLang)) return;

    const id = setInterval(() => {
      if (applyGoogleTranslate(preferredLang)) clearInterval(id);
    }, 300);

    return () => clearInterval(id);
  }, [preferredLang]);

  /* LOGIN handlers */
  const handleLogin = (e) => {
    e.preventDefault();

    if (!inputName.trim()) {
      alert("Name is required");
      return;
    }

    setFarmerName(inputName);
    setInputName("");
    window.location.href = "/";
  };

  const handleLogout = () => {
    setFarmerName("");
    window.location.href = "/";
  };

  const handleThemeToggle = () => {
    setTheme(theme === "dark" ? "light" : "dark");
  };

  const handleLangChange = (e) => {
    setPreferredLang(e.target.value);
  };

  const handleNavToggle = () => {
    setNavOpen(!isNavOpen);
  };

  /* UI */

  return (
    <Router>
      <div className={`app ${theme === "dark" ? "theme-dark" : ""}`}>

        {/* NAVBAR */}
        <nav className="navbar">
          <div className="nav-left">
            <FaLeaf className="icon" />
            <Link to="/" className="brand">
              Fasal Saathi
            </Link>
          </div>

          <ul className={`nav-center ${isNavOpen ? "active" : ""}`}>
            <li>
              <Link to="/" onClick={() => setNavOpen(false)}>
                <FaHome /> Home
              </Link>
            </li>
            <li>
              <Link to="/advisor" onClick={() => setNavOpen(false)}>
                <FaComments /> Chat
              </Link>
            </li>
            <li>
              <Link to="/how-it-works" onClick={() => setNavOpen(false)}>
                <FaInfoCircle /> How It Works
              </Link>
            </li>
            <li>
              <Link to="/crop-guide" onClick={() => setNavOpen(false)}>
                <FaLeaf className="icon" /> Crop Guide
              </Link>
            </li>
            <li>
              <Link to="/profit-calculator" onClick={() => setNavOpen(false)}>
                <FaCalculator /> Profit Calculator
              </Link>
            </li>
          </ul>

          <div className="nav-right">
            <button onClick={handleThemeToggle}>
              {theme === "dark" ? "☀️" : "🌙"}
            </button>

            <select
              className="lang-select notranslate"
              value={preferredLang}
              onChange={handleLangChange}
            >
              {LANGUAGE_OPTIONS.map((l) => (
                <option key={l.value} value={l.value}>
                  {l.label}
                </option>
              ))}
            </select>

            <div className="nav-user">
              {farmerName ? (
                <>
                  👋 {farmerName}
                  <button onClick={handleLogout}>Change User</button>
                </>
              ) : (
                <Link to="/login">Get Started</Link>
              )}
            </div>
          </div>

          <button
            className="hamburger"
            onClick={handleNavToggle}
          >
            {isNavOpen ? <FaTimes /> : <FaBars />}
          </button>
        </nav>

        {/* ROUTES */}
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/advisor" element={<Advisor />} />
          <Route path="/how-it-works" element={<How />} />
          <Route path="/profit-calculator" element={<CropProfitCalculator />} />

          <Route
            path="/login"
            element={
              <div className="login-page">
                <div className="login-card">
                  <h2>👨‍🌾 Farmer Login</h2>

                  <form onSubmit={handleLogin}>
                    <input
                      type="text"
                      placeholder="Enter your name"
                      value={inputName}
                      onChange={(e) => setInputName(e.target.value)}
                    />

                    <button type="submit">Login</button>
                  </form>
                </div>
              </div>
            }
          />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
