import React, { useEffect, useState } from "react";
import { BrowserRouter as Router, Link, Route, Routes } from "react-router-dom";
import { FaBars, FaComments, FaHome, FaInfoCircle, FaLeaf, FaTimes } from "react-icons/fa";
import Advisor from "./Advisor";
import GoogleTranslate from "./GoogleTranslate";
import Home from "./Home";
import How from "./How";
import WeatherAlertBar from "./WeatherAlertBar";
import "./App.css";

function App() {
  const [isOpen, setIsOpen] = useState(false);
  const [name, setName] = useState(localStorage.getItem("farmerName") || "");
  const [inputName, setInputName] = useState("");
  const [preferredLang, setPreferredLang] = useState(
    localStorage.getItem("preferredLanguage") || "en"
  );

  useEffect(() => {
    if (!preferredLang) {
      return;
    }

    const interval = setInterval(() => {
      const select = document.querySelector(".goog-te-combo");
      if (select) {
        select.value = preferredLang;
        select.dispatchEvent(new Event("change"));
        clearInterval(interval);
      }
    }, 500);

    return () => clearInterval(interval);
  }, [preferredLang]);

  const handleLogin = (event) => {
    event.preventDefault();
    if (!inputName.trim() || !preferredLang) {
      return;
    }

    localStorage.setItem("farmerName", inputName);
    localStorage.setItem("preferredLanguage", preferredLang);
    setName(inputName);
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

  const applyLanguage = (language) => {
    setPreferredLang(language);
    localStorage.setItem("preferredLanguage", language);

    const select = document.querySelector(".goog-te-combo");
    if (select) {
      select.value = language;
      select.dispatchEvent(new Event("change"));
    }
  };

  return (
    <Router>
      <div className="app">
        <GoogleTranslate lang={preferredLang} />

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
            <select
              className="lang-select"
              value={preferredLang}
              onChange={(event) => applyLanguage(event.target.value)}
            >
              <option value="en">English</option>
              <option value="hi">Hindi</option>
              <option value="mr">Marathi</option>
              <option value="bn">Bengali</option>
              <option value="ta">Tamil</option>
              <option value="te">Telugu</option>
              <option value="gu">Gujarati</option>
              <option value="pa">Punjabi</option>
              <option value="kn">Kannada</option>
              <option value="ml">Malayalam</option>
              <option value="or">Odia</option>
            </select>

            <div className="nav-user">
              {name ? (
                <>
                  Welcome, {name}!
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

          <button className="hamburger" onClick={() => setIsOpen((open) => !open)}>
            {isOpen ? <FaTimes /> : <FaBars />}
          </button>
        </nav>

        <WeatherAlertBar />

        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/advisor" element={<Advisor />} />
          <Route path="/how-it-works" element={<How />} />
          <Route
            path="/login"
            element={
              <div className="login-page">
                <div className="login-card">
                  <h2>Farmer Login</h2>
                  <p>Welcome! Please provide your details to continue.</p>
                  <form onSubmit={handleLogin}>
                    <input
                      type="text"
                      placeholder="Enter your name"
                      value={inputName}
                      onChange={(event) => setInputName(event.target.value)}
                      required
                    />

                    <select
                      value={preferredLang}
                      onChange={(event) => setPreferredLang(event.target.value)}
                      required
                    >
                      <option value="en">English</option>
                      <option value="hi">Hindi</option>
                      <option value="mr">Marathi</option>
                      <option value="bn">Bengali</option>
                      <option value="ta">Tamil</option>
                      <option value="te">Telugu</option>
                      <option value="gu">Gujarati</option>
                      <option value="pa">Punjabi</option>
                      <option value="kn">Kannada</option>
                      <option value="ml">Malayalam</option>
                      <option value="or">Odia</option>
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
