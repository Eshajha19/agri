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
  FaUserSecret,
  FaFileInvoiceDollar,
  FaHome,
  FaSun,
  FaMoon,
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
import FarmFinance from "./FarmFinance";
import YieldPredictor from "./YieldPredictor";
import Footer from "./components/Footer";
import { SkipLink } from "./NavigationManager";
import { useTheme } from "./ThemeContext";

// Libs
import { auth, db, isFirebaseConfigured, doc, onSnapshot, setDoc } from "./lib/firebase";
import { onAuthStateChanged, signOut } from "firebase/auth";

// CSS
import "./App.css";

const LANGUAGE_OPTIONS = [
  { value: "en", label: "🌍 English", englishName: "english" },
  { value: "hi", label: "🇮🇳 हिंदी", englishName: "hindi" },
  { value: "mr", label: "🇮🇳 मराठी", englishName: "marathi" },
  { value: "bn", label: "🇮🇳 বাংলা", englishName: "bengali" },
  { value: "ta", label: "🇮🇳 தமிழ்", englishName: "tamil" },
  { value: "te", label: "🇮🇳 తెలుగు", englishName: "telugu" },
  { value: "gu", label: "🇮🇳 ગુજરાતી", englishName: "gujarati" },
  { value: "pa", label: "🇮🇳 ਪੰਜਾਬੀ", englishName: "punjabi" },
  { value: "kn", label: "🇮🇳 ಕನ್ನಡ", englishName: "kannada" },
  { value: "ml", label: "🇮🇳 മലയാളം", englishName: "malayalam" },
  { value: "or", label: "🇮🇳 ଓଡ଼ିଆ", englishName: "odia" },
  { value: "as", label: "🇮🇳 অসমীয়া", englishName: "assamese" },
];

const getInitialLanguage = () => {
  // Always default to English when the user enters the site
  return "en";
};

/**
 * Helper to apply Google Translate selection to the hidden widget
 */
const applyGoogleTranslate = (langCode) => {
  try {
    const select = document.querySelector(".goog-te-combo");
    if (select) {
      if (select.value !== langCode) {
        select.value = langCode;
        select.dispatchEvent(new Event("change", { bubbles: true }));
      }
      return true;
    }
  } catch (e) {
    console.error("GT Apply Error:", e);
  }
  return false;
};

const GuestBanner = ({ onSignUp }) => (
  <div className="guest-banner">
    <div className="guest-banner-content">
      <FaUserSecret className="banner-icon" />
      <span>
        <strong>Guest Session Active:</strong> Explore the platform freely! 
        <Link to="/auth" className="banner-link"> Sign Up</Link> to save your progress permanently.
      </span>
    </div>
  </div>
);

function App() {
  const scorecardRef = useRef(null);
  const [preferredLang, setPreferredLang] = useState(getInitialLanguage);
  const [isOpen, setIsOpen] = useState(false);
  const { theme, toggleTheme } = useTheme();
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
  }, [detectAndSetLiteMode]);

  const { i18n } = useTranslation();
  const location = useLocation();

  useNotifications();

  /* ---------------- THEME SYSTEM (Moved to ThemeProvider) ---------------- */

  /* ---------------- LANGUAGE AUTO-TRANS ---------------- */
  useEffect(() => {
    if (applyGoogleTranslate(preferredLang)) return;
    const id = setInterval(() => {
      if (applyGoogleTranslate(preferredLang)) clearInterval(id);
    }, 300);
    return () => clearInterval(id);
  }, [preferredLang]);

  /* ---------------- AUTH & FIRESTORE SYNC ---------------- */
  useEffect(() => {
    if (!isFirebaseConfigured()) {
      setLoading(false);
      return;
    }

    // Safety timeout — if Firebase auth never responds (revoked key, network issue),
    // force loading=false so the app doesn't hang forever on the spinner.
    const safetyTimer = setTimeout(() => {
      setLoading(false);
    }, 5000);

    const unsubscribeAuth = onAuthStateChanged(auth, (currentUser) => {
      clearTimeout(safetyTimer);
      setUser(currentUser);
      if (currentUser) {
        const unsubscribeDoc = onSnapshot(doc(db, "users", currentUser.uid), (userDoc) => {
          if (userDoc.exists()) {
            const data = userDoc.data();
            setUserData(data);
            setProfileCompleted(data.profileCompleted === true);
          } else if (currentUser.isAnonymous) {
            setUserData({ displayName: "Guest Farmer", isAnonymous: true });
            setProfileCompleted(true);
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
    return () => { clearTimeout(safetyTimer); unsubscribeAuth(); };
  }, []);

  // E2EE Key Generation Sync
  useEffect(() => {
    if (!user || !isFirebaseConfigured()) return;

    const ensurePublicKey = async () => {
      try {
        let privateJwk = localStorage.getItem(`agri:ecdh_private_${user.uid}`);
        let publicJwk = localStorage.getItem(`agri:ecdh_public_${user.uid}`);
        
        if (!privateJwk || !publicJwk) {
          const { cryptoService } = await import("./utils/cryptoService");
          const keyPair = await cryptoService.generateECDHKeyPair();
          privateJwk = await cryptoService.exportKey(keyPair.privateKey);
          publicJwk = await cryptoService.exportKey(keyPair.publicKey);
          localStorage.setItem(`agri:ecdh_private_${user.uid}`, JSON.stringify(privateJwk));
          localStorage.setItem(`agri:ecdh_public_${user.uid}`, JSON.stringify(publicJwk));
        } else {
          publicJwk = JSON.parse(publicJwk);
        }

        const pubKeyRef = doc(db, "public_keys", user.uid);
        await setDoc(pubKeyRef, { jwk: publicJwk }, { merge: true });
      } catch (error) {
        console.error("Failed to generate/publish ECDH keys globally:", error);
      }
    };

    ensurePublicKey();
  }, [user]);

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
  const handleThemeToggle = toggleTheme;
  const handleLogout = async () => {
    try {
      await signOut(auth);
      window.location.href = "/";
    } catch (error) {
      console.error("Sign out error:", error);
    }
  };
  const scrollToTop = () => window.scrollTo({ top: 0, behavior: "smooth" });

  return (
    <div className={`app ${theme === "dark" ? "theme-dark" : ""} ${liteMode ? "lite-mode" : ""}`}>
      <SkipLink />
      {user?.isAnonymous && <GuestBanner />}

      {loading && <Loader fullPage={true} message={<span className="notranslate">Initializing Fasal Saathi...</span>} />}

      {isOffline && (
        <div className="offline-banner" role="alert">
          You are currently offline. Running in offline mode using local data.
        </div>
      )}

      <nav className={`navbar ${isOpen ? "menu-open" : ""}`} role="navigation" aria-label="Main Navigation">
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
        </ul>

        <div className="nav-right">
          <button onClick={handleThemeToggle} className="theme-toggle" aria-label="Toggle Theme">
            {theme === "dark" ? <FaSun className="theme-toggle-icon" /> : <FaMoon className="theme-toggle-icon" />}
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
                    value={preferredLang}
                    onChange={(lang) => {
                      setPreferredLang(lang);
                      i18n.changeLanguage(lang);
                      localStorage.setItem("agri:preferredLanguage", lang);
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
                {userData?.role === "admin" && (
                  <Link to="/admin/feedback" onClick={() => setShowMoreMenu(false)} role="menuitem"><FaShieldAlt /> Feedback Admin</Link>
                )}
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
                  <span className="profile-name">👋 {userData?.displayName || user.email?.split('@')[0]}</span>
                  <FaChevronDown className={`chevron ${showScorecard ? 'open' : ''}`} />
                </div>

                {showScorecard && userData && (
                  <div className="profile-scorecard" onClick={(e) => e.stopPropagation()}>
                    <div className="scorecard-header">
                      <div className="scorecard-avatar">{userData.displayName?.[0] || 'F'}</div>
                      <h3>{userData.displayName}</h3>
                      <p>{userData.email || user.email}</p>
                    </div>
                    <div className="scorecard-body">
                      {[
                        { label: "🌾 Primary Crop", value: userData.cropType || "N/A" },
                        { label: "🌐 Language", value: LANGUAGE_OPTIONS.find(l => l.value === (userData.language || preferredLang))?.label || preferredLang },
                        { label: "📍 Location", value: userData.address || "Fetching..." }
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
        </div>

        <button className="hamburger" onClick={() => setIsOpen(!isOpen)} aria-label="Toggle Menu">
          {isOpen ? <FaTimes /> : <FaBars />}
        </button>
      </nav>

      {/* VERIFICATION GUARD */}
      {!loading && user && !user.isAnonymous && !user.emailVerified && !showScorecard && location.pathname !== "/login" && (
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

      {/* PROFILE COMPLETION GUARD */}
      {!loading && user && (user.isAnonymous || user.emailVerified) && !profileCompleted && location.pathname !== "/profile-setup" && (
        <Navigate to="/profile-setup" />
      )}

      <main id="main-content" tabIndex="-1" style={{ outline: 'none' }}>
        <Routes>
          <Route path="/" element={<Home user={user} />} />
          <Route path="/advisor" element={<Advisor userData={userData} />} />
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
          <Route path="/farm-finance" element={<FarmFinance />} />
          <Route path="/yield-predictor" element={<YieldPredictor />} />
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
      <Footer />
    </div>
  );
}

export default App;
