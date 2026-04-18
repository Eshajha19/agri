import React, { useEffect } from "react";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";

import Advisor from "./Advisor";
import Home from "./Home";
import Resources from "./Resources";
import CropGuide from "./CropGuide";
import { ToastContainer } from "react-toastify";
import useNotifications from "./Notifications";
import {
  FaHome,
  FaComments,
  FaInfoCircle,
  FaLeaf,
  FaBars,
  FaTimes,
} from "react-icons/fa";
import How from "./How";
import { NavLink } from "react-router-dom";
import { useUiStore } from "./stores/uiStore";

import "./App.css";
import LanguageDropdown from "./LanguageDropdown";

/* ---------------- LANGUAGE ---------------- */

const LANGUAGE_OPTIONS = [
  { value: "en", label: "🌍 English", englishName: "english" },
  { value: "hi", label: "🇮🇳 हिंदी", englishName: "hindi" },
  { value: "mr", label: "🇮🇳 मराठी", englishName: "marathi" },
  { value: "bn", label: "🇮🇳 বাংলা", englishName: "bengali" },
  { value: "ta", label: "🇮🇳 தமிழ்", englishName: "tamil" },
  { value: "te", label: "🇮🇳 తెలుగు", englishName: "telugu" },
  { value: "gu", label: "🇮🇳 ગુજરાતી", englishName: "gujarati" },
  { value: "pa", label: "🇮🇳 ਪੰਜਾਬੀ", englishName: "punjabi" },
  { value: "kn", label: "🇮🇳 ಕನ್ನಡ", englishName: "kannada" },
  { value: "ml", label: "🇮🇳 മലയാളം", englishName: "malayalam" },
  { value: "or", label: "🇮🇳 ଓଡ଼ିଆ", englishName: "odia" },
  { value: "as", label: "🇮🇳 অসমীয়া", englishName: "assamese" },
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
  const [preferredLang, setPreferredLang] = useState(getInitialLanguage);
  const [isOpen, setIsOpen] = useState(false);
  const [sunlight, setSunlight] = useState(false); 
  useNotifications();

  const [theme, setTheme] = useState(() => {
    return localStorage.getItem("theme") || "light";
  });

  const [name, setName] = useState(localStorage.getItem("farmerName") || "");
  const [inputName, setInputName] = useState("");

  /* ---------------- THEME ---------------- */
  useEffect(() => {
    document.documentElement.classList.toggle("theme-dark", theme === "dark");
  }, [theme]);

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
              <Link to="/crop-guide" onClick={() => setIsOpen(false)}>
                <FaLeaf className="icon" /> Crop Guide
              </Link>
            </li>
      <Link to="/resources" onClick={() => setIsOpen(false)}>
                Resources
      </Link>
      </li>
          </ul>

          <div className="nav-right">
            <button onClick={handleThemeToggle}>
              {theme === "dark" ? "☀️" : "🌙"}
            </button>

            <LanguageDropdown
              options={LANGUAGE_OPTIONS}
              value={preferredLang}
              onChange={(val) => syncLanguage(val, setPreferredLang)}
            />

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
          <Route path="/crop-guide" element={<CropGuide />} />
          <Route path="/resources" element={<Resources />} />

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

       <ToastContainer />
    </Router>
  );
}

export default App;
