import React, { useEffect, useState } from "react";
import { Routes, Route, Link, Navigate, useLocation } from "react-router-dom";

import { ToastContainer } from "react-toastify";

import {
  FaHome,
  FaComments,
  FaInfoCircle,
  FaLeaf,
  FaBars,
  FaTimes,
  FaCalculator,
  FaMap,
  FaTachometerAlt,
} from "react-icons/fa";

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
import Loader from "./Loader";
import FarmingMap from "./FarmingMap";
import CropProfitCalculator from "./CropProfitCalculator";

import { syncOfflineRequests } from "./lib/syncOfflineRequests";
import { auth, db, isFirebaseConfigured } from "./lib/firebase";
import { auth, db, isFirebaseConfigured, doc, onSnapshot } from "./lib/firebase";
import { onAuthStateChanged, signOut } from "firebase/auth";

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
  const [showScorecard] = useState(false);
  const location = useLocation();

  const farmerName = userData?.name || user?.displayName || "";

  const handleLangChange = (e) => {
    syncLanguage(e.target.value, setPreferredLang);
  };

  const handleNavToggle = () => {
    setIsOpen(!isOpen);
  };

  const handleLogout = async () => {
    try {
      await signOut(auth);
      setUser(null);
      setUserData(null);
      setProfileCompleted(true);
    } catch (error) {
      console.error("Logout error:", error);
    }
  };

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
     const handleNetworkChange = () => {
       const offline = !navigator.onLine;
       setIsOffline(offline);
       if (!offline) {
         syncOfflineRequests();
       }
     };
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
      {loading && <Loader fullPage={true} message="Initializing Fasal Saathi..." />}
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

          <div className="nav-user" onClick={() => setShowScorecard(!showScorecard)}>
            {loading ? (
              <div className="nav-loader-mini"></div>
            ) : user ? (
              <div className="user-profile-trigger">
                <div className="profile-main">
                  <span className="profile-name">{userData?.displayName || user.email?.split('@')[0]}</span>
                  <FaChevronDown className={`chevron ${showScorecard ? 'open' : ''}`} />
                </div>

                {showScorecard && userData && (
                  <div className="profile-scorecard" onClick={(e) => e.stopPropagation()}>
                    <div className="scorecard-header">
                      <div className="scorecard-avatar">{userData.displayName?.[0] || 'F'}</div>
                      <h3>{userData.displayName}</h3>
                      <p>{userData.email}</p>
                    </div>
                    <div className="scorecard-body">
                      {[
                        { label: "Primary Crop", value: userData.cropType },
                        { label: "Language", value: LANGUAGE_OPTIONS.find(l => l.value === userData.language)?.label || userData.language },
                        { label: "Location", value: userData.address || "Fetching..." }
                      ].map((item, i) => (
                        <div key={i} className="score-item">
                          <label>{item.label}</label>
                          <span>{item.value}</span>
                        </div>
                      ))}
                    </div>
                    <div className="scorecard-footer">
                      <button onClick={handleLogout} className="btn-logout-alt">Sign Out</button>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <Link to="/login" className="btn-get-started">Get Started</Link>
            )}
          </div>

          <button
            className="hamburger"
            onClick={handleNavToggle}
          >
            {isOpen ? <FaTimes /> : <FaBars />}
          </button>
        </div>
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
        <Route path="/farming-map" element={<FarmingMap />} />
        <Route path="/profit-calculator" element={<CropProfitCalculator />} />
      </Routes>

      <ToastContainer position="bottom-right" />
    </div>
  );
}

export default App;
