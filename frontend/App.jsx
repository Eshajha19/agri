import React, { useEffect, useState, useRef } from "react";
import { Routes, Route, Link, Navigate, useLocation } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import { useFloating, flip, shift, offset, autoUpdate } from "@floating-ui/react";
import {
  FaComments,
  FaLeaf,
  FaTachometerAlt,
  FaTimes,
  FaBars,
  FaChevronDown,
  FaChevronUp,
  FaWhatsapp,
  FaInfoCircle,
  FaBook,
  FaShieldAlt,
  FaBolt,
} from "react-icons/fa";
import { usePerformanceStore } from "./stores/performanceStore";

// Components
import AdminFeedback from "./AdminFeedback";
import Advisor from "./Advisor";
import Auth from "./Auth";
import Calendar from "./FarmingCalendar";
import Contributors from "./Contributors";
import CropGuide from "./CropGuide";
import CropProfitCalculator from "./CropProfitCalculator";
import Dashboard from "./Dashboard";
import Feedback from "./Feedback";
import FarmingMap from "./FarmingMap";
import Schemes from "./GovernmentSchemes";
import How from "./How";
import Home from "./Home";
import MarketPrices from "./MarketPrices";
import Loader from "./Loader";
import Community from "./Community";
import ContactUs from "./ContactUs";
import AboutUs from "./AboutUs";
import LanguageDropdown from "./LanguageDropdown";
import useNotifications from "./Notifications";
import ProfileSetup from "./ProfileSetup";
import QRTraceability from "./QRTraceability";
import Resources from "./Resources";
import SeasonalCropPlanner from "./SeasonalCropPlanner";
import SoilGuide from "./SoilGuide";
import CropDiseaseAwareness from "./CropDiseaseAwareness";
import CropRotation from "./CropRotation";
import Helpline from "./Helpline";
import Glossary from "./Glossary";
import RiskIndex from "./RiskIndex";
import Blog from "./Blog";
import BlogDetail from "./BlogDetail";
import FAQ from "./FAQ";
import NotFound from "./NotFound";
import PrivacyPolicy from "./PrivacyPolicy";
import Terms from "./Terms";
import SoilAnalysis from "./SoilAnalysis";
import SeedVerifier from "./SeedVerifier";
import { SkipLink } from "./NavigationManager";

// Libs
import { auth, db, isFirebaseConfigured, doc, onSnapshot } from "./lib/firebase";
import { onAuthStateChanged, signOut } from "firebase/auth";

// CSS
import "./App.css";
import "./themes/sunlight.css";

const LANGUAGE_OPTIONS = [
  { value: "en", label: "English" },
  { value: "hi", label: "हिन्दी" },
  { value: "bn", label: "বাংলা" },
  { value: "te", label: "తెలుగు" },
  { value: "ta", label: "தமிழ்" },
  { value: "mr", label: "मराठी" },
  { value: "gu", label: "ગુજરાતી" },
  { value: "kn", label: "ಕನ್ನಡ" },
  { value: "ml", label: "മലയാളം" },
  { value: "pa", label: "ਪੰਜਾਬੀ" },
  { value: "or", label: "ଓଡ଼ିଆ" },
  { value: "as", label: "অসমীয়া" },
];

const getInitialLanguage = () => {
  try {
    return localStorage.getItem("preferredLanguage") || "en";
  } catch {
    return "en";
  }
};

function App() {
  const scorecardRef = useRef(null);
  const [settings, setSettings] = useState({ language: getInitialLanguage() });
  const [isOpen, setIsOpen] = useState(false);
  const [user, setUser] = useState(null);
  const [userData, setUserData] = useState(null);
  const [profileCompleted, setProfileCompleted] = useState(true);
  const [loading, setLoading] = useState(true);
  const [showScorecard, setShowScorecard] = useState(false);
  const [showMoreMenu, setShowMoreMenu] = useState(false);
  const [isOffline, setIsOffline] = useState(!navigator.onLine);
  const [showScrollTop, setShowScrollTop] = useState(false);
  
  const { liteMode, setLiteMode, detectAndSetLiteMode } = usePerformanceStore();

  useEffect(() => {
    detectAndSetLiteMode();
  }, []);

  const { i18n } = useTranslation();
  const location = useLocation();

  // Auth & Profile Sync
  useEffect(() => {
    if (!isFirebaseConfigured()) {
      setLoading(false);
      return;
    }
    const unsubscribeAuth = onAuthStateChanged(auth, (currentUser) => {
      setUser(currentUser);
      if (currentUser) {
        localStorage.setItem("userId", currentUser.uid);
        const unsubscribeDoc = onSnapshot(
          doc(db, "users", currentUser.uid),
          (userDoc) => {
            if (userDoc.exists()) {
              const data = userDoc.data();
              setUserData(data);
              setProfileCompleted(data.profileCompleted === true);
            } else {
              setUserData(null);
              setProfileCompleted(false);
            }
            setLoading(false);
          },
          (error) => {
            console.error("Firestore sync error:", error);
            setLoading(false);
          }
        );
        return () => unsubscribeDoc();
      } else {
        setUserData(null);
        setProfileCompleted(true);
        setLoading(false);
      }
    });
    return () => unsubscribeAuth();
  }, []);

  // Theme Sync
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

  // Online/Offline detection
  useEffect(() => {
    const handleOnline = () => setIsOffline(false);
    const handleOffline = () => setIsOffline(true);
    window.addEventListener("online", handleOnline);
    window.addEventListener("offline", handleOffline);
    return () => {
      window.removeEventListener("online", handleOnline);
      window.removeEventListener("offline", handleOffline);
    };
  }, []);

  // Scroll to Top logic
  useEffect(() => {
    const handleScroll = () => {
      setShowScrollTop(window.scrollY > 300);
    };
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  // Language init
  useEffect(() => {
    const lang = getInitialLanguage();
    i18n.changeLanguage(lang);
  }, [i18n]);

  // Click outside scorecard
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (scorecardRef.current && !scorecardRef.current.contains(event.target)) {
        setShowScorecard(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleNavToggle = () => setIsOpen(!isOpen);
  const handleThemeToggle = () => setIsDarkTheme(!isDarkTheme);
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
  const scrollToTop = () => window.scrollTo({ top: 0, behavior: "smooth" });

  useNotifications();

  return (
    <div className={`app ${isDarkTheme ? "theme-dark" : ""} ${liteMode ? "lite-mode" : ""}`}>
      <SkipLink />
      
      {loading && <Loader fullPage={true} message={<span className="notranslate">Initializing Fasal Saathi...</span>} />}
      
      {isOffline && (
        <div className="offline-banner" role="alert">
          You are currently offline. Running in offline mode using local data.
        </div>
      )}

      <nav className={`navbar ${isOpen ? "menu-open" : ""}`} role="navigation" aria-label="Main Navigation">
        <div className="nav-left">
          <Link to="/" className="brand" aria-label="Fasal Saathi Home">
            <FaLeaf className="brand-icon" />
            <span className="notranslate" translate="no">Fasal Saathi</span>
          </Link>
        </div>

        <ul className={`nav-center ${isOpen ? "active" : ""}`}>
          <li><Link to="/" onClick={() => setIsOpen(false)}>Home</Link></li>
          <li><Link to="/how-it-works" onClick={() => setIsOpen(false)}>Works</Link></li>
          <li><Link to="/crop-guide" onClick={() => setIsOpen(false)}>Guide</Link></li>
          <li><Link to="/resources" onClick={() => setIsOpen(false)}>Resources</Link></li>
          <li><Link to="/crop-planner" onClick={() => setIsOpen(false)}>Planner</Link></li>
        </ul>

        <div className="nav-right">
          <button onClick={handleThemeToggle} className="theme-toggle" aria-label="Toggle Theme">
            {isDarkTheme ? "☀️" : "🌙"}
          </button>

          <button 
            onClick={(e) => { e.stopPropagation(); setShowMoreMenu(!showMoreMenu); }} 
            className={`more-menu-toggle ${showMoreMenu ? 'active' : ''}`} 
            aria-label="More Options"
          >
            <span className="notranslate">More</span>
            <FaChevronDown className="chevron" />
          </button>

          {showMoreMenu && (
            <div className="more-dropdown" onClick={(e) => e.stopPropagation()} role="menu">
              <div className="dropdown-links">
                <div className="language-selector-section">
                  <label className="language-label">Language:</label>
                  <LanguageDropdown 
                    options={LANGUAGE_OPTIONS}
                    value={settings.language}
                    onChange={(lang) => {
                      setSettings({ ...settings, language: lang });
                      i18n.changeLanguage(lang);
                      localStorage.setItem("preferredLanguage", lang);
                    }}
                  />
                </div>
                <div className="performance-toggle-section">
                  <button 
                    className={`lite-mode-toggle ${liteMode ? 'active' : ''}`}
                    onClick={() => setLiteMode(!liteMode)}
                    role="menuitem"
                  >
                    <div className="toggle-info">
                      <FaBolt className="zap-icon" />
                      <span>Lite Mode {liteMode ? "ON" : "OFF"}</span>
                    </div>
                    <div className="toggle-switch">
                      <div className="switch-handle" />
                    </div>
                  </button>
                </div>
                <Link to="/dashboard" onClick={() => setShowMoreMenu(false)} role="menuitem"><FaTachometerAlt /> Dashboard</Link>
                <Link to="/community" onClick={() => setShowMoreMenu(false)} role="menuitem"><FaComments /> Community</Link>
                <Link to="/disease-awareness" onClick={() => setShowMoreMenu(false)} role="menuitem"><FaLeaf /> Awareness</Link>
                <Link to="/risk-index" onClick={() => setShowMoreMenu(false)} role="menuitem"><FaShieldAlt /> Risk Index</Link>
                <Link to="/glossary" onClick={() => setShowMoreMenu(false)} role="menuitem"><FaBook /> Glossary</Link>
                <Link to="/about" onClick={() => setShowMoreMenu(false)} role="menuitem"><FaInfoCircle /> About Us</Link>
                <Link to="/contact" onClick={() => setShowMoreMenu(false)} role="menuitem"><FaInfoCircle /> Contact</Link>
              </div>
            </div>
          )}

          <div className="nav-user" ref={scorecardRef}>
            {!loading && user ? (
              <div className="user-profile-trigger" onClick={() => { setShowScorecard(!showScorecard); setShowMoreMenu(false); }}>
                <div className="profile-main">
                  <span className="profile-name">{userData?.displayName || user.email?.split('@')?.[0] || "Farmer"}</span>
                  <FaChevronDown className={`chevron ${showScorecard ? 'open' : ''}`} />
                </div>

                {showScorecard && (
                  <div className="profile-scorecard" onClick={(e) => e.stopPropagation()}>
                    <div className="scorecard-header">
                      <div className="scorecard-avatar">{userData?.displayName?.[0] || 'F'}</div>
                      <h3>{userData?.displayName || "Farmer"}</h3>
                      <p>{user.email}</p>
                    </div>
                    <div className="scorecard-body">
                      <div className="score-item"><label>Primary Crop</label><span>{userData?.cropType || "N/A"}</span></div>
                      <div className="score-item"><label>Location</label><span>{userData?.address || "N/A"}</span></div>
                      <div className="score-item">
                        <label>Language</label>
                        <span>{LANGUAGE_OPTIONS.find(l => l.value === (userData?.language || settings.language))?.label || "English"}</span>
                      </div>
                    </div>
                    <div className="scorecard-footer">
                      <button onClick={handleLogout} className="btn-logout-alt">Sign Out</button>
                    </div>
                  </div>
                )}
              </div>
            ) : !loading && (
              <Link to="/login" className="btn-get-started" aria-label="Get Started">
                <span className="notranslate">Get Started</span>
              </Link>
            )}
          </div>
        </div>

        <button className="hamburger" onClick={handleNavToggle} aria-label="Toggle Menu">
          {isOpen ? <FaTimes /> : <FaBars />}
        </button>
      </nav>

      {!loading && user && !user.emailVerified && location.pathname !== "/login" && (
        <div className="verification-overlay">
          <div className="verification-card">
            <div className="verify-icon">✉️</div>
            <h2>Verify Your Email</h2>
            <p>We've sent a link to <b>{user.email}</b>.<br /> Please verify your email to unlock all features.</p>
            <button onClick={() => window.location.reload()} className="btn-refresh">I've Verified My Email</button>
            <button onClick={handleLogout} className="btn-logout-simple">Sign Out</button>
          </div>
        </div>
      )}

      {!loading && user && user.emailVerified && !profileCompleted && location.pathname !== "/profile-setup" && (
        <Navigate to="/profile-setup" replace />
      )}

      <main id="main-content" tabIndex="-1" style={{ outline: 'none' }}>
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
          <Route path="/community" element={<Community />} />
          <Route path="/soil-analysis" element={<SoilAnalysis />} />
          <Route path="/faq" element={<FAQ />} />
          <Route path="/terms" element={<Terms />} />
          <Route path="/privacy-policy" element={<PrivacyPolicy />} />
          <Route path="/contributors" element={<Contributors />} />
          <Route path="/trace/:id" element={<QRTraceability />} />
          <Route path="/contact" element={<ContactUs />} />
          <Route path="/about" element={<AboutUs />} />
          <Route path="/crop-planner" element={<SeasonalCropPlanner />} />
          <Route path="/soil-guide" element={<SoilGuide />} />
          <Route path="/disease-awareness" element={<CropDiseaseAwareness />} />
          <Route path="/helpline" element={<Helpline />} />
          <Route path="/glossary" element={<Glossary />} />
          <Route path="/risk-index" element={<RiskIndex />} />
          <Route path="/crop-rotation" element={<CropRotation />} />
          <Route path="/seed-verifier" element={<SeedVerifier />} />
          <Route path="/blog" element={<Blog />} />
          <Route path="/blog/:id" element={<BlogDetail />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </main>

      {/* Floating Buttons */}
      <Link to="/advisor" className="floating-chat-btn" aria-label="Open AI Advisor Chat">
        <FaComments size={28} aria-hidden="true" />
      </Link>

      <a 
        href="https://wa.me/14155238886?text=I%20want%20to%20start%20the%20conversation" 
        target="_blank" 
        rel="noopener noreferrer" 
        className="whatsapp-float"
        title="Chat with WhatsApp Bot"
      >
        <FaWhatsapp />
        <span className="tooltip">Chat with Bot</span>
      </a>

      {showScrollTop && (
        <button className="scroll-to-top" onClick={scrollToTop} aria-label="Scroll to top">
          <FaChevronUp size={24} />
        </button>
      )}

      <ToastContainer position="bottom-right" />
    </div>
  );
}

export default App;
