import React, { useEffect } from "react";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import { Toaster } from "react-hot-toast";

import Advisor from "./Advisor";
import Home from "./Home";
import Resources from "./Resources";
import CropGuide from "./CropGuide";
import CropProfitCalculator from "./CropProfitCalculator";
import FarmingMap from "./FarmingMap";
import {
  FaHome,
  FaComments,
  FaInfoCircle,
  FaLeaf,
  FaBars,
  FaTimes,
  FaCalculator,
  FaMap,
} from "react-icons/fa";
import { ToastContainer } from "react-toastify";

import Advisor from "./Advisor";
import Home from "./Home";
import Resources from "./Resources";
import CropGuide from "./CropGuide";
import How from "./How";
import Dashboard from "./Dashboard";
import Auth from "./Auth";
import ProfileSetup from "./ProfileSetup";
import LanguageDropdown from "./LanguageDropdown";
import useNotifications from "./Notifications";
import Schemes from "./GovernmentSchemes";
import Feedback from "./Feedback";
import AdminFeedback from "./AdminFeedback";
import Calendar from "./FarmingCalendar";
import MarketPrices from "./MarketPrices";

import { auth, db, isFirebaseConfigured } from "./lib/firebase";

import "./App.css";
import "./themes/sunlight.css";

/* ---------------- LANGUAGE ---------------- */

const LANGUAGE_OPTIONS = [
  { value: "en", label: "🌍 English", englishName: "english" },
  { value: "hi", label: "🇮🇳 हिंदी", englishName: "hindi" },
  { value: "mr", label: "🇮🇳 मराठी", englishName: "marathi" },
  { value: "bn", label: "🇮🇳 বাংলা", englishName: "bengali" },
  { value: "ta", label: "🇮🇳 தமிழ்", englishName: "tamil" },
  { value: "te", label: "🇮🇳 తెలుగు", englishName: "telugu" },
  { value: "gu", label: "🇮🇳 ગુજરાତି", englishName: "gujarati" },
  { value: "pa", label: "🇮🇳 ਪੰਜਾਬੀ", englishName: "punjabi" },
  { value: "kn", label: "🇮🇳 ಕನ್ನಡ", englishName: "kannada" },
  { value: "ml", label: "🇮🇳 മലയാളം", englishName: "malayalam" },
  { value: "or", label: "🇮🇳 ଓଡ଼ିଆ", englishName: "odia" },
  { value: "as", label: "🇮🇳 অসমীয়া", englishName: "assamese" },
];

const getInitialLanguage = () => {
  try {
    const stored = localStorage.getItem("preferredLanguage");
    return LANGUAGE_OPTIONS.some((l) => l.value === stored) ? stored : "en";
  } catch {
    return "en";
  }
};

const setGoogleTranslateCookie = (lang) => {
  try {
    const cookieValue = encodeURIComponent(`/en/${lang}`);
    document.cookie = `googtrans=${cookieValue}; path=/;`;
    const hostname = window.location.hostname;
    if (hostname) {
      document.cookie = `googtrans=${cookieValue}; domain=.${hostname}; path=/;`;
    }
  } catch {
    // Ignore if cookies are blocked
  }
};

const applyGoogleTranslate = (lang) => {
  document.cookie = `googtrans=/en/${lang}; path=/`;
  window.location.reload();
};

const syncLanguage = (lang, setLang) => {
  setLang(lang);
  localStorage.setItem("preferredLanguage", lang);
  applyGoogleTranslate(lang);
};

function App() {
  const [preferredLang, setPreferredLang] = useState(getInitialLanguage);
  const [isOpen, setIsOpen] = useState(false);
  const [user, setUser] = useState(null);
  const [userData, setUserData] = useState(null);
  const [profileCompleted, setProfileCompleted] = useState(true);
  const [loading, setLoading] = useState(true);
  const [showScorecard, setShowScorecard] = useState(false);
  const location = useLocation();

  useNotifications();

   /* ---------------- THEME SYSTEM ---------------- */
   const [isDarkTheme, setIsDarkTheme] = useState(() => {
     try {
       return (localStorage.getItem("theme") || "light") === "dark";
     } catch {
       return false;
     }
   });

   useEffect(() => {
     document.documentElement.classList.toggle("theme-dark", isDarkTheme);
     localStorage.setItem("theme", isDarkTheme ? "dark" : "light");
   }, [isDarkTheme]);

   const handleThemeToggle = () => {
     setIsDarkTheme(!isDarkTheme);
   };


   useEffect(() => {
     setGoogleTranslateCookie(preferredLang);
   }, [preferredLang]);

  /* LOGIN handlers */
  const handleLogin = (e) => {
    e.preventDefault();

    if (!inputName.trim()) {
      alert("Name is required");
      return;
    }

  useEffect(() => {
    if (!isFirebaseConfigured()) {
      setLoading(false);
      return;
    }
    const unsubscribeAuth = onAuthStateChanged(auth, (currentUser) => {
      setUser(currentUser);
      if (currentUser) {
        const unsubscribeDoc = onSnapshot(doc(db, "users", currentUser.uid), (userDoc) => {
          if (userDoc.exists()) {
            const data = userDoc.data();
            setUserData(data);
            setProfileCompleted(data.profileCompleted === true);
          } else {
            setUserData(null);
            setProfileCompleted(false);
          }
          setLoading(false);
        }, (error) => {
          console.error("Firestore sync error:", error);
          setLoading(false);
        });
        return () => unsubscribeDoc();
      } else {
        setUserData(null);
        setProfileCompleted(true);
        setLoading(false);
      }
    });
    return () => unsubscribeAuth();
  }, []);


  /* ---------------- OFFLINE STATUS ---------------- */
  /* ---------------- AUTH STATE LISTENER ---------------- */



  const [isOffline, setIsOffline] = useState(!navigator.onLine);

   useEffect(() => {
     const handleNetworkChange = () => setIsOffline(!navigator.onLine);
     window.addEventListener("online", handleNetworkChange);
     window.addEventListener("offline", handleNetworkChange);

     const interval = setInterval(handleNetworkChange, 1000);

     return () => {
       window.removeEventListener("online", handleNetworkChange);
       window.removeEventListener("offline", handleNetworkChange);
       clearInterval(interval);
     };
   }, []);

  return (
    <div className={`app ${isDarkTheme ? "theme-dark" : ""}`}>
      {isOffline && (
        <div className="offline-banner">
          You are currently offline. Running in offline mode using local data.
        </div>
      )}

      <nav className="navbar">
        <div className="nav-left">
          <FaLeaf className="icon" />
          <Link to="/" className="brand">Fasal Saathi</Link>
        </div>

        <ul className={`nav-center ${isOpen ? "active" : ""}`}>
          <li><Link to="/" onClick={() => setIsOpen(false)}><FaHome /> Home</Link></li>
          <li><Link to="/advisor" onClick={() => setIsOpen(false)}><FaComments /> Chat</Link></li>
          <li><Link to="/how-it-works" onClick={() => setIsOpen(false)}><FaInfoCircle /> How It Works</Link></li>
          <li><Link to="/crop-guide" onClick={() => setIsOpen(false)}><FaLeaf className="icon" /> Crop Guide</Link></li>
          <li><Link to="/resources" onClick={() => setIsOpen(false)}>Resources</Link></li>
          <li><Link to="/dashboard" onClick={() => setIsOpen(false)}><FaTachometerAlt /> Dashboard</Link></li>

        </ul>

        <div className="nav-right">
          <button onClick={handleThemeToggle} className="theme-toggle" aria-label="Toggle Theme">
            {isDarkTheme ? "☀️" : "🌙"}
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
          <Route
            path="/farming-map"
            element={
              <div className="page-container">
                <FarmingMap />
              </div>
            }
          />
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
            ) : (
              <Link to="/login" className="btn-get-started">Get Started</Link>
            )}
          </div>
        </div>

        <button className="hamburger" onClick={() => setIsOpen(!isOpen)} aria-label="Toggle Menu">
          {isOpen ? <FaTimes /> : <FaBars />}
        </button>
      </nav>

      {!loading && user && !user.emailVerified && !showScorecard && location.pathname !== "/login" && (
        <div className="verification-overlay">
          <div className="verification-card">
            <div className="verify-icon">✉️</div>
            <h2>Verify Your Email</h2>
            <p>We've sent a link to <b>{user.email}</b>.<br /> Please verify your email to unlock all features.</p>
            <button
              onClick={() => {
                auth.currentUser.reload().then(() => window.location.reload());
              }}
              className="btn-refresh"
            >
              I've Verified My Email
            </button>
            <button onClick={handleLogout} className="btn-logout-simple">Sign Out</button>
          </div>
        </div>
      )}

      {!loading && user && user.emailVerified && !profileCompleted && location.pathname !== "/profile-setup" && (
        <Navigate to="/profile-setup" />
      )}

      <Routes>
        <Route path="/" element={<Home user={user} />} />
        <Route path="/advisor" element={<Advisor />} />
        <Route path="/how-it-works" element={<How />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/crop-guide" element={<CropGuide />} />
        <Route path="/schemes" element={<Schemes />} />
        <Route path="/resources" element={<Resources />} />
        <Route path="/login" element={<Auth />} />
        <Route path="/profile-setup" element={<ProfileSetup user={user} profileCompleted={profileCompleted} />} />
        <Route path="/calendar" element={<Calendar />} />
        <Route path="/share-feedback" element={<Feedback />} />
        <Route path="/admin/feedback" element={<AdminFeedback />} />
        <Route path="/market-prices" element={<MarketPrices />} />
      </Routes>

      <ToastContainer position="bottom-right" />
    </div>
  );
}

export default App;
