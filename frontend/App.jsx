import React, { useEffect, useState } from "react";
import { Link, Navigate, Route, Routes, useLocation } from "react-router-dom";
import { Toaster } from "react-hot-toast";
import { useTranslation } from "react-i18next";
import {
  FaBars,
  FaComments,
  FaGlobe,
  FaHome,
  FaInfoCircle,
  FaLeaf,
  FaSeedling,
  FaTachometerAlt,
  FaTimes,
  FaUsers,
} from "react-icons/fa";

import AdminFeedback from "./AdminFeedback";
import Advisor from "./Advisor";
import Auth from "./Auth";
import Calendar from "./FarmingCalendar";
import Contributors from "./Contributors";
import CropGuide from "./CropGuide";
import CropProfitCalculator from "./CropProfitCalculator";
import Dashboard from "./Dashboard";
import Feedback from "./Feedback";
import FarmingMap from "./FarmingMap";
import GovernmentSchemes from "./GovernmentSchemes";
import How from "./How";
import Home from "./Home";
import MarketPrices from "./MarketPrices";
import ProfileSetup from "./ProfileSetup";
import Resources from "./Resources";

import "./App.css";
import "./themes/sunlight.css";

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

function App() {
  const { i18n } = useTranslation();
  const [isNavOpen, setIsNavOpen] = useState(false);
  const [isDarkTheme, setIsDarkTheme] = useState(() => {
    try {
      return localStorage.getItem("theme") === "dark";
    } catch {
      return false;
    }
  });
  const location = useLocation();

  useEffect(() => {
    document.documentElement.classList.toggle("theme-dark", isDarkTheme);
    localStorage.setItem("theme", isDarkTheme ? "dark" : "light");
  }, [isDarkTheme]);

  const handleLanguageChange = (newLang) => {
    i18n.changeLanguage(newLang);
    localStorage.setItem("preferredLanguage", newLang);
  };

  useEffect(() => {
    setIsNavOpen(false);
  }, [location.pathname]);

  return (
    <div className={`app ${isDarkTheme ? "theme-dark" : ""}`}>
      <Toaster position="top-right" />

      <nav className="navbar">
        <div className="nav-left">
          <FaLeaf className="icon" />
          <Link to="/" className="brand">
            Fasal Saathi
          </Link>
        </div>

        <ul className={`nav-center ${isNavOpen ? "active" : ""}`}>
          <li>
            <Link to="/">
              <FaHome /> Home
            </Link>
          </li>
          <li>
            <Link to="/advisor">
              <FaComments /> Advisor
            </Link>
          </li>
          <li>
            <Link to="/how-it-works">
              <FaInfoCircle /> How It Works
            </Link>
          </li>
          <li>
            <Link to="/crop-guide">
              <FaSeedling /> Crop Guide
            </Link>
          </li>
          <li>
            <Link to="/dashboard">
              <FaTachometerAlt /> Dashboard
            </Link>
          </li>
          <li>
            <Link to="/contributors">
              <FaUsers /> Contributors
            </Link>
          </li>
        </ul>

        <div className="nav-right">
          <button
            type="button"
            className="theme-toggle"
            onClick={() => setIsDarkTheme((value) => !value)}
            aria-label="Toggle theme"
          >
            {isDarkTheme ? "☀" : "☾"}
          </button>

          <select
            className="lang-select notranslate"
            value={i18n.language}
            onChange={(event) => handleLanguageChange(event.target.value)}
          >
            {LANGUAGE_OPTIONS.map((language) => (
              <option key={language.value} value={language.value}>
                {language.label}
              </option>
            ))}
          </select>

          <Link to="/login" className="btn-get-started">
            Get Started
          </Link>
        </div>

        <button
          type="button"
          className="hamburger"
          onClick={() => setIsNavOpen((value) => !value)}
          aria-label="Toggle menu"
        >
          {isNavOpen ? <FaTimes /> : <FaBars />}
        </button>
      </nav>

      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/advisor" element={<Advisor />} />
        <Route path="/how-it-works" element={<How />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/crop-guide" element={<CropGuide />} />
        <Route path="/profit-calculator" element={<CropProfitCalculator />} />
        <Route path="/farming-map" element={<FarmingMap />} />
        <Route path="/schemes" element={<GovernmentSchemes />} />
        <Route path="/resources" element={<Resources />} />
        <Route path="/contributors" element={<Contributors />} />
        <Route path="/calendar" element={<Calendar />} />
        <Route path="/share-feedback" element={<Feedback />} />
        <Route path="/admin/feedback" element={<AdminFeedback />} />
        <Route path="/market-prices" element={<MarketPrices />} />
        <Route path="/login" element={<Auth />} />
        <Route path="/profile-setup" element={<ProfileSetup user={null} profileCompleted={false} />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </div>
  );
}

export default App;
