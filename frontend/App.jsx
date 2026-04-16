import React, { useState, useRef, useEffect } from "react";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import "./App.css";
import Advisor from "./Advisor";
import How from "./How";
import Home from "./Home";
import GoogleTranslate from "./GoogleTranslate";
import {
  FaHome,
  FaComments,
  FaInfoCircle,
  FaLeaf,
  FaBars,
  FaTimes,
} from "react-icons/fa";

function App() {
  const [loginLang, setLoginLang] = useState("");
  const [showAlert, setShowAlert] = useState(true);
  const [isOpen, setIsOpen] = useState(false);
  const [sunlight, setSunlight] = useState(false);

  const [name, setName] = useState(localStorage.getItem("farmerName") || "");
  const [inputName, setInputName] = useState("");
  const [preferredLang, setPreferredLang] = useState(
    localStorage.getItem("preferredLanguage") || ""
  );

  const videoRef = useRef(null);

  // Google Translate auto-apply
  useEffect(() => {
    if (!preferredLang) return;

    const applyLang = () => {
      const select = document.querySelector(".goog-te-combo");
      if (!select) return false;

      if (select.value !== preferredLang) {
        select.value = preferredLang;
        select.dispatchEvent(new Event("change"));
      }
      return true;
    };

    if (applyLang()) return;

    const observer = new MutationObserver(() => {
      if (applyLang()) observer.disconnect();
    });

    observer.observe(document.body, { childList: true, subtree: true });

    return () => observer.disconnect();
  }, [preferredLang]);

  const handleLogin = (e) => {
    e.preventDefault();

    if (!inputName.trim()) {
      alert("Name is required");
      return;
    }

    if (!loginLang) {
      alert("Please select a language");
      return;
    }

    localStorage.setItem("farmerName", inputName);
    localStorage.setItem("preferredLanguage", loginLang);

    setName(inputName);
    setPreferredLang(loginLang);

    setInputName("");
    window.location.href = "/";
  };

  const handleLogout = () => {
    localStorage.removeItem("farmerName");
    localStorage.removeItem("preferredLanguage");
    setName("");
    setPreferredLang("en");
    window.location.href = "/";
  };

  return (
    <Router>
      {/* ✅ FIXED ROOT */}
      <div className={sunlight ? "app sunlight" : "app"}>
        {/* Google Translate */}
        <GoogleTranslate lang={preferredLang} />

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
              onClick={() => setSunlight(!sunlight)}
              className="sunlight-toggle"
            >
              {sunlight ? "👁️ Normal View" : "☀️ Sunlight Mode"}
            </button>

            {/* LANGUAGE SELECT */}
            <select
              className="lang-select notranslate"
              translate="no"
              value={preferredLang}
              onChange={(e) => {
                const lang = e.target.value;
                setPreferredLang(lang);
                localStorage.setItem("preferredLanguage", lang);
              }}
            >
              <option value="">Select Language</option>
              <option value="en">🌍 English</option>
              <option value="hi">🇮🇳 हिंदी</option>
              <option value="mr">🇮🇳 मराठी</option>
              <option value="bn">🇮🇳 বাংলা</option>
              <option value="ta">🇮🇳 தமிழ்</option>
              <option value="te">🇮🇳 తెలుగు</option>
              <option value="gu">🇮🇳 ગુજરાતી</option>
              <option value="pa">🇮🇳 ਪੰਜਾਬੀ</option>
              <option value="kn">🇮🇳 ಕನ್ನಡ</option>
              <option value="ml">🇮🇳 മലയാളം</option>
              <option value="or">🇮🇳 ଓଡ଼ିଆ</option>
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

        {/* ALERT */}
        {showAlert && (
          <div className="alert-bar">
            🌧️ Weather Alert: Heavy rainfall expected in parts of Maharashtra this evening.
            <button className="close-btn" onClick={() => setShowAlert(false)}>
              <FaTimes />
            </button>
          </div>
        )}

        {/* ROUTES */}
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/advisor" element={<Advisor />} />
          <Route path="/how-it-works" element={<How />} />

          <Route
            path="/login"
            element={
              <div className="login-page">
                <div className="login-card">
                  <h2>👨‍🌾 Farmer Login</h2>
                  <p>Welcome! Please provide your details to continue.</p>

                  <form onSubmit={handleLogin}>
                    <input
                      type="text"
                      placeholder="Enter your name"
                      value={inputName}
                      onChange={(e) => setInputName(e.target.value)}
                    />

                    {/* CLEAN LOGIN LANGUAGE SELECT */}
                    <select
                      value={loginLang}
                      onChange={(e) => setLoginLang(e.target.value)}
                    >
                      <option value="">Select Language</option>
                      <option value="en">🌍 English</option>
                      <option value="hi">🇮🇳 हिंदी (Hindi)</option>
                      <option value="mr">🇮🇳 मराठी (Marathi)</option>
                      <option value="bn">🇮🇳 বাংলা (Bengali)</option>
                      <option value="ta">🇮🇳 தமிழ் (Tamil)</option>
                      <option value="te">🇮🇳 తెలుగు (Telugu)</option>
                      <option value="gu">🇮🇳 ગુજરાતી (Gujarati)</option>
                      <option value="pa">🇮🇳 ਪੰਜਾਬੀ (Punjabi)</option>
                      <option value="kn">🇮🇳 ಕನ್ನಡ (Kannada)</option>
                      <option value="ml">🇮🇳 മലയാളം (Malayalam)</option>
                      <option value="or">🇮🇳 ଓଡ଼ିଆ (Odia)</option>
                    </select>

                    <button type="submit">Login</button>
                  </form>

                  <p className="login-note">
                    Your preferences will be saved for future visits.
                  </p>
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
