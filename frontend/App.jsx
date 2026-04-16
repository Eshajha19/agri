import React, { useState, useEffect, useRef } from "react";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import "./App.css";
import Advisor from "./Advisor";
import How from "./How";
import Home from "./Home";
import { FaHome, FaComments, FaInfoCircle, FaLeaf, FaBars, FaTimes } from "react-icons/fa";
import { useLocation } from "react-router-dom";

// 🔹 ScrollToTop component to fix navigation positioning
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

  // Languages array for dropdowns
  const languages = [
    { code: "en", name: "English", flag: "🌍" },
    { code: "hi", name: "Hindi", flag: "🇮🇳" },
    { code: "mr", name: "Marathi", flag: "🇮🇳" },
    { code: "bn", name: "Bengali", flag: "🇮🇳" },
    { code: "ta", name: "Tamil", flag: "🇮🇳" },
    { code: "te", name: "Telugu", flag: "🇮🇳" },
    { code: "gu", name: "Gujarati", flag: "🇮🇳" },
    { code: "pa", name: "Punjabi", flag: "🇮🇳" },
    { code: "kn", name: "Kannada", flag: "🇮🇳" },
    { code: "ml", name: "Malayalam", flag: "🇮🇳" },
    { code: "or", name: "Odia", flag: "🇮🇳" },
  ];

  // Searchable Language Select Component
  const SearchableLanguageSelect = ({ value, onChange }) => {
    const [searchTerm, setSearchTerm] = useState("");
    const [isDropdownOpen, setIsDropdownOpen] = useState(false);
    const dropdownRef = useRef(null);

    // Filter languages based on search term
    const filteredLanguages = languages.filter(lang =>
      lang.name.toLowerCase().includes(searchTerm.toLowerCase())
    );

    // Get selected language name
    const selectedLanguage = languages.find(lang => lang.code === value);

    // Close dropdown when clicking outside
    useEffect(() => {
      const handleClickOutside = (event) => {
        if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
          setIsDropdownOpen(false);
        }
      };
      document.addEventListener("mousedown", handleClickOutside);
      return () => document.removeEventListener("mousedown", handleClickOutside);
    }, []);

    return (
      <div className="searchable-dropdown" ref={dropdownRef}>
        <div
          className="dropdown-header"
          onClick={() => setIsDropdownOpen(!isDropdownOpen)}
        >
          {selectedLanguage ? `${selectedLanguage.flag} ${selectedLanguage.name}` : "Select Language"}
          <span className="dropdown-arrow">{isDropdownOpen ? "▲" : "▼"}</span>
        </div>
        {isDropdownOpen && (
          <div className="dropdown-content">
            <input
              type="text"
              placeholder="Search languages..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="search-input"
              onClick={(e) => e.stopPropagation()}
            />
            <ul className="language-list">
              {filteredLanguages.length > 0 ? (
                filteredLanguages.map(lang => (
                  <li
                    key={lang.code}
                    onClick={() => {
                      onChange(lang.code);
                      setIsDropdownOpen(false);
                      setSearchTerm("");
                    }}
                    className={value === lang.code ? "selected" : ""}
                  >
                    {lang.flag} {lang.name}
                  </li>
                ))
              ) : (
                <li className="no-results">No results found</li>
              )}
            </ul>
          </div>
        )}
      </div>
    );
  };


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
              onClick={() => setSunlight(!sunlight)}
              className="sunlight-toggle"
            >
              {sunlight ? "👁️ Normal View" : "☀️ Sunlight Mode"}
            </button>

            {/* Language Dropdown */}
            {/* LANGUAGE SELECT */}
            <select
              className="lang-select"
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

                  <form onSubmit={handleLogin}>
                    <input
                      type="text"
                      placeholder="Enter your name"
                      value={inputName}
                      onChange={(e) => setInputName(e.target.value)}
                    />

                    <SearchableLanguageSelect
                      value={loginLang}
                      onChange={setLoginLang}
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
