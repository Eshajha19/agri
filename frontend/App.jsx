import React, { useState, useEffect } from "react";
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from "react-router-dom";
import "./App.css";
import Advisor from "./Advisor";
import How from "./How";
import Home from "./Home";
import { FaHome, FaComments, FaInfoCircle, FaLeaf, FaBars, FaTimes } from "react-icons/fa";

// 🔹 ScrollToTop component
function ScrollToTop() {
  const { pathname } = useLocation();
  useEffect(() => {
    window.scrollTo(0, 0);
  }, [pathname]);
  return null;
}

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

  // ✅ NEW STATES
  const [search, setSearch] = useState("");
  const [isLangOpen, setIsLangOpen] = useState(false);

  const languageOptions = [
    { code: "en", label: "🌍 English" },
    { code: "hi", label: "🇮🇳 हिंदी" },
    { code: "mr", label: "🇮🇳 मराठी" },
    { code: "bn", label: "🇮🇳 বাংলা" },
    { code: "ta", label: "🇮🇳 தமிழ்" },
    { code: "te", label: "🇮🇳 తెలుగు" },
    { code: "gu", label: "🇮🇳 ગુજરાતી" },
    { code: "pa", label: "🇮🇳 ਪੰਜਾਬੀ" },
    { code: "kn", label: "🇮🇳 ಕನ್ನಡ" },
    { code: "ml", label: "🇮🇳 മലയാളം" },
    { code: "or", label: "🇮🇳 ଓଡ଼ିଆ" },
  ];

  const filteredLanguages = languageOptions.filter((lang) =>
  lang.label.toLowerCase().includes(search.toLowerCase()) ||
  lang.code.toLowerCase().includes(search.toLowerCase()) ||
  lang.label.toLowerCase().replace(/[^a-z]/g, "").includes(search.toLowerCase())
);

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
      <ScrollToTop />
      <div className={sunlight ? "app sunlight" : "app"}>

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

            {/* ✅ SEARCHABLE LANGUAGE DROPDOWN */}
            <div className="lang-dropdown">
              <div
                className="lang-selected"
                onClick={() => setIsLangOpen(!isLangOpen)}
              >
                {preferredLang
                  ? languageOptions.find((l) => l.code === preferredLang)?.label
                  : "Select Language"}
              </div>

              {isLangOpen && (
                <div className="lang-menu">
                  <input
                    type="text"
                    placeholder="Search language..."
                    value={search}
                    onChange={(e) => setSearch(e.target.value)}
                    className="lang-search"
                  />

                  {filteredLanguages.length > 0 ? (
                    filteredLanguages.map((lang) => (
                      <div
                        key={lang.code}
                        className="lang-item"
                        onClick={() => {
                          setPreferredLang(lang.code);
                          localStorage.setItem("preferredLanguage", lang.code);
                          setIsLangOpen(false);
                          setSearch("");
                        }}
                      >
                        {lang.label}
                      </div>
                    ))
                  ) : (
                    <div className="lang-item">No language found</div>
                  )}
                </div>
              )}
            </div>

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

                  <form onSubmit={handleLogin}>
                    <input
                      type="text"
                      placeholder="Enter your name"
                      value={inputName}
                      onChange={(e) => setInputName(e.target.value)}
                    />

                    <select
                      value={loginLang}
                      onChange={(e) => setLoginLang(e.target.value)}
                    >
                      <option value="">Select Language</option>
                      <option value="en">English</option>
                      <option value="hi">Hindi</option>
                      <option value="mr">Marathi</option>
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