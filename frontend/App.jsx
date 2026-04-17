import React, { useState, useEffect } from "react";
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from "react-router-dom";
import "./App.css";
import Advisor from "./Advisor";
import How from "./How";
import Home from "./Home";
import { FaHome, FaComments, FaInfoCircle, FaLeaf, FaBars, FaTimes } from "react-icons/fa";

// Scroll to top
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

  const [search, setSearch] = useState("");
  const [isLangOpen, setIsLangOpen] = useState(false);

  // ✅ Language list with aliases (FIXED SEARCH)
  const languageOptions = [
    { code: "en", label: "🌍 English", aliases: ["english", "en"] },
    { code: "hi", label: "🇮🇳 हिंदी", aliases: ["hindi", "hi"] },
    { code: "mr", label: "🇮🇳 मराठी", aliases: ["marathi", "mr"] },
    { code: "bn", label: "🇮🇳 বাংলা", aliases: ["bengali", "bn"] },
    { code: "ta", label: "🇮🇳 தமிழ்", aliases: ["tamil", "ta"] },
    { code: "te", label: "🇮🇳 తెలుగు", aliases: ["telugu", "te"] },
    { code: "gu", label: "🇮🇳 ગુજરાતી", aliases: ["gujarati", "gu"] },
    { code: "pa", label: "🇮🇳 ਪੰਜਾਬੀ", aliases: ["punjabi", "pa", "pu"] },
    { code: "kn", label: "🇮🇳 ಕನ್ನಡ", aliases: ["kannada", "kn"] },
    { code: "ml", label: "🇮🇳 മലയാളം", aliases: ["malayalam", "ml"] },
    { code: "or", label: "🇮🇳 ଓଡ଼ିଆ", aliases: ["odia", "or"] },
  ];

  const filteredLanguages = languageOptions.filter((lang) => {
    const text = search.toLowerCase();
    return (
      lang.label.toLowerCase().includes(text) ||
      lang.aliases.some((alias) => alias.includes(text))
    );
  });

  const handleLogin = (e) => {
    e.preventDefault();

    if (!inputName.trim()) return alert("Name is required");
    if (!loginLang) return alert("Please select a language");

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
            <Link to="/" className="brand">Fasal Saathi</Link>
          </div>

          <ul className={`nav-center ${isOpen ? "active" : ""}`}>
            <li><Link to="/" onClick={() => setIsOpen(false)}><FaHome /> Home</Link></li>
            <li><Link to="/advisor" onClick={() => setIsOpen(false)}><FaComments /> Chat</Link></li>
            <li><Link to="/how-it-works" onClick={() => setIsOpen(false)}><FaInfoCircle /> How It Works</Link></li>
          </ul>

          <div className="nav-right">
            <button onClick={() => setSunlight(!sunlight)} className="sunlight-toggle">
              {sunlight ? "👁️ Normal" : "☀️ Sunlight"}
            </button>

            {/* LANGUAGE DROPDOWN */}
            <div className="lang-dropdown">

              <button
                className="lang-selected"
                onClick={() => setIsLangOpen(!isLangOpen)}
                aria-haspopup="listbox"
                aria-expanded={isLangOpen}
              >
                {preferredLang
                  ? languageOptions.find((l) => l.code === preferredLang)?.label
                  : "Select Language"}
              </button>

              {isLangOpen && (
                <div className="lang-menu" role="listbox">

                  <input
                    type="text"
                    placeholder="Search language..."
                    value={search}
                    onChange={(e) => setSearch(e.target.value)}
                    className="lang-search"
                  />

                  {filteredLanguages.length > 0 ? (
                    filteredLanguages.map((lang) => (
                      <button
                        key={lang.code}
                        className="lang-item"
                        onClick={() => {
                          setPreferredLang(lang.code);
                          localStorage.setItem("preferredLanguage", lang.code);
                          setIsLangOpen(false);
                          setSearch("");
                        }}
                        role="option"
                        aria-selected={preferredLang === lang.code}
                      >
                        {lang.label}
                      </button>
                    ))
                  ) : (
                    <div className="lang-item">No results found</div>
                  )}
                </div>
              )}
            </div>

            {/* USER */}
            <div className="nav-user">
              {name ? (
                <>
                  👋 {name}
                  <button className="logout-btn" onClick={handleLogout}>Logout</button>
                </>
              ) : (
                <Link to="/login">Login</Link>
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
            🌧️ Heavy rainfall expected today
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

          <Route path="/login" element={
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
          }/>
        </Routes>
      </div>
    </Router>
  );
}

export default App;