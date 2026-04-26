import React, { useState, useEffect } from "react";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import "./App.css";
import Advisor from "./Advisor";
import Home from "./Home";
import Terms from "./Terms";
import Privacy from "./Privacy";
import How from "./How";
import FAQ from "./FAQ";
import {
  FaHome,
  FaComments,
  FaInfoCircle,
  FaLeaf,
  FaBars,
  FaTimes,
} from "react-icons/fa";

function App() {
  const [preferredLang, setPreferredLang] = useState(getInitialPreferredLanguage);
  const [isOpen, setIsOpen] = useState(false);
  const [themeAnimNonce, setThemeAnimNonce] = useState(0);

  const getInitialTheme = () => {
    try {
      const stored = localStorage.getItem("theme");
      if (stored === "light" || stored === "dark") return stored;
    } catch {
      // ignore
    }
    return window.matchMedia &&
      window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light";
  };

  const [theme, setTheme] = useState(getInitialTheme);

  const [name, setName] = useState(localStorage.getItem("farmerName") || "");
  const [inputName, setInputName] = useState("");

  const handleThemeToggle = () => {
    setTheme((t) => (t === "dark" ? "light" : "dark"));
    setThemeAnimNonce((n) => n + 1);
  };


  const handleLogin = (e) => {
    e.preventDefault();

    if (!inputName.trim()) {
      alert("Name is required");
      return;
    }

    localStorage.setItem("farmerName", inputName);

    setName(inputName);

    setInputName("");
    window.location.href = "/";
  };

  const handleLogout = () => {
    localStorage.removeItem("farmerName");
    setName("");
    window.location.href = "/";
  };

  useEffect(() => {
    const root = document.documentElement;
    root.classList.toggle("theme-dark", theme === "dark");
    try {
      localStorage.setItem("theme", theme);
    } catch {
      // ignore
    }
  }, [theme]);

  useEffect(() => {
    if (applyLanguageToGoogleTranslate(preferredLang)) {
      return;
    }

    let attempts = 0;
    const maxAttempts = 20;
    const retryId = window.setInterval(() => {
      attempts += 1;
      const applied = applyLanguageToGoogleTranslate(preferredLang);
      if (applied || attempts >= maxAttempts) {
        window.clearInterval(retryId);
      }
    }, 300);

    return () => {
      window.clearInterval(retryId);
    };
  }, [preferredLang]);

  return (
    <Router>
      <div className="app">

        {/* Navbar */}
        {/* NAVBAR */}
        <nav className="navbar">
          <div className="nav-left">
            <FaLeaf className="icon" />
            <Link to="/" className="brand">
              Fasal Saathi
            </Link>
          </div>

          <ul className={`nav-center ${isOpen ? "active" : ""}`}>
            <li>
              <Link to="/" onClick={() => setIsOpen(false)}>
                <FaHome className="icon" /> Home
              </Link>
            </li>
            <li>
              <Link to="/advisor" onClick={() => setIsOpen(false)}>
                <FaComments className="icon" /> Chat
              </Link>
            </li>
            <li>
              <Link to="/how-it-works" onClick={() => setIsOpen(false)}>
                <FaInfoCircle className="icon" /> How It Works
              </Link>
            </li>
          </ul>

          <div className="nav-right">
            <button
              type="button"
              onClick={handleThemeToggle}
              className="theme-toggle"
              aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
              aria-pressed={theme === "dark"}
              title={theme === "dark" ? "Light mode" : "Dark mode"}
            >
              <span key={themeAnimNonce} className="theme-toggle-icon">
                {theme === "dark" ? "☀️" : "🌙"}
              </span>
            </button>

            {/* Language Dropdown */}
            {/* LANGUAGE SELECT */}
            <select
              className="lang-select"
              value={preferredLang}
              onChange={(e) => {
                syncPreferredLanguage(e.target.value, setPreferredLang);
              }}
            >
              {LANGUAGE_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>

            {/* USER */}
            <div className="nav-user">
              {name ? (
                <>
                  👋 Welcome, {name}!
                  <button className="logout-btn" onClick={handleLogout}>
                    Logout
                  </button>
                </>
              ) : (
                <Link to="/login" onClick={() => setIsOpen(false)}>
                  Login
                </Link>
              )}
            </div>
          </div>

          <button className="hamburger" onClick={() => setIsOpen(!isOpen)}>
            {isOpen ? <FaTimes /> : <FaBars />}
          </button>
        </nav>

        {/* ROUTES */}
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/advisor" element={<Advisor />} />
          <Route path="/how-it-works" element={<How />} />
          
  {/* NEW ROUTES */}
  <Route path="/faq" element={<FAQ />} />
  <Route path="/terms" element={<Terms />} />
  <Route path="/privacy-policy" element={<Privacy />} />

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

                    <select
                      value={preferredLang}
                      onChange={(e) => {
                        syncPreferredLanguage(e.target.value, setPreferredLang);
                      }}
                      style={{ marginBottom: "18px" }}
                    >
                      {LANGUAGE_OPTIONS.map((option) => (
                        <option key={option.value} value={option.value}>
                          {option.label}
                        </option>
                      ))}
                    </select>

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
