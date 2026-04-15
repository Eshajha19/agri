import React, { useState, useRef, useEffect } from "react";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import "./App.css";
import Advisor from "./Advisor";
import How from "./How";
import Home from "./Home";
import {
  FaHome,
  FaComments,
  FaInfoCircle,
  FaLeaf,
  FaBars,
  FaTimes,
} from "react-icons/fa";

function App() {
  const [showAlert, setShowAlert] = useState(true);
  const [isOpen, setIsOpen] = useState(false);
  const [sunlight, setSunlight] = useState(false); 

  const [name, setName] = useState(localStorage.getItem("farmerName") || "");
  const [inputName, setInputName] = useState("");
  const [preferredLang, setPreferredLang] = useState(
    localStorage.getItem("preferredLanguage") || "en"
  );


  const handleLogin = (e) => {
    e.preventDefault();
    if (inputName.trim() && preferredLang) {
      localStorage.setItem("farmerName", inputName);
      localStorage.setItem("preferredLanguage", preferredLang);
      setName(inputName);
      setInputName("");
      window.location.href = "/";
    }
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
      <div className={sunlight ? "app sunlight" : "app"}>
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
              aria-label="Toggle High Contrast Sunlight Mode"
            >
              {sunlight ? "👁️ Normal View" : "☀️ Sunlight Mode"}
            </button>

            <select
              className="lang-select"
              value={preferredLang}
              onChange={(e) => {
                const lang = e.target.value;
                setPreferredLang(lang);
                localStorage.setItem("preferredLanguage", lang);
              }}
            >
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

        {showAlert && (
          <div className="alert-bar">
            🌧️ Weather Alert: Heavy rainfall expected in parts of Maharashtra this evening.
            <button className="close-btn" onClick={() => setShowAlert(false)}>
              <FaTimes />
            </button>
          </div>
        )}

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
                      required
                    />
                    <select
                      value={preferredLang}
                      onChange={(e) => setPreferredLang(e.target.value)}
                      required
                    >
                      <option value="en">English</option>
                      <option value="hi">Hindi</option>
                      <option value="mr">Marathi</option>
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