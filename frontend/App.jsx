import React, { useEffect, useState, useRef } from "react";
import { Routes, Route, Link, Navigate, useLocation } from "react-router-dom";
import { SkipLink } from "./NavigationManager";

import { ToastContainer } from "react-toastify";
import { useFloating, flip, shift, offset, autoUpdate } from "@floating-ui/react";
  import {
  FaHome,
  FaComments,
  FaGlobe,
  FaHome,
  FaInfoCircle,
  FaLeaf,
  FaSeedling,
  FaTachometerAlt,
  FaTimes,
  FaCalculator,
  FaMap,
  FaTachometerAlt,
  FaChevronDown,
  FaChevronUp,
  FaWhatsapp,
  FaUser,
  FaBook,
  FaShieldAlt,
} from "react-icons/fa";

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
import GovernmentSchemes from "./GovernmentSchemes";
import How from "./How";
import Home from "./Home";
import MarketPrices from "./MarketPrices";
import Loader from "./Loader";
import FarmingMap from "./FarmingMap";
import CropProfitCalculator from "./CropProfitCalculator";
import Community from "./Community";
import ContactUs from "./ContactUs";
import AboutUs from "./AboutUs";
import Contributors from "./Contributors";
import SeasonalCropPlanner from "./SeasonalCropPlanner";
import SoilGuide from "./SoilGuide";
import CropDiseaseAwareness from "./CropDiseaseAwareness";
import Helpline from "./Helpline";
import Glossary from "./Glossary";
import RiskIndex from "./RiskIndex";
import Blog from "./Blog";
import BlogDetail from "./BlogDetail";

import FAQ from "./FAQ";
import NotFound from "./NotFound";
import PrivacyPolicy from "./PrivacyPolicy";
import Terms from "./Terms";

import { syncOfflineRequests } from "./lib/syncOfflineRequests";
import { auth, db, isFirebaseConfigured, doc, onSnapshot } from "./lib/firebase";
import { onAuthStateChanged, signOut } from "firebase/auth";

import "./App.css";
import "./themes/sunlight.css";

/* ---------------- LANGUAGE ---------------- */
const LANGUAGE_OPTIONS = [
  { value: "en", label: "English", englishName: "english" },
  { value: "hi", label: "हिन्दी", englishName: "hindi" },
  { value: "bn", label: "বাংলা", englishName: "bengali" },
  { value: "te", label: "తెలుగు", englishName: "telugu" },
  { value: "ta", label: "தமிழ்", englishName: "tamil" },
  { value: "mr", label: "मराठी", englishName: "marathi" },
  { value: "gu", label: "ગુજરાતી", englishName: "gujarati" },
  { value: "kn", label: "ಕನ್ನಡ", englishName: "kannada" },
  { value: "ml", label: "മലയാളം", englishName: "malayalam" },
];

const getInitialLanguage = () => {
  try {
    return localStorage.getItem("preferredLanguage") || "en";
  } catch {
    return "en";
  }
};

const syncLanguage = (lang, setLang) => {
  setLang(lang);
  try {
    localStorage.setItem("preferredLanguage", lang);
  } catch {}
  
  // Trigger Google Translate if available
  if (window.google && window.google.translate && window.google.translate.TranslateElement) {
    try {
      // Find the Google Translate select element
      let select = document.querySelector('.goog-te-combo');
      if (!select) {
        // Try to find it in the hidden translate element
        const gtEl = document.getElementById('google_translate_element');
        if (gtEl) {
          select = gtEl.querySelector('select');
        }
      }
      
      if (select) {
        select.value = lang;
        // Trigger change event to force translation
        select.dispatchEvent(new Event('change', { bubbles: true }));
      }
      
      // Also update the cookie for Google Translate
      if (lang !== 'en') {
        const cookieValue = '/en/' + lang;
        const expires = new Date();
        expires.setFullYear(expires.getFullYear() + 1);
        document.cookie = 'googtrans=' + cookieValue + '; path=/; expires=' + expires.toUTCString();
      } else {
        // Clear cookie for English
        document.cookie = 'googtrans=; path=/; max-age=0';
      }
    } catch (e) {
      console.error('Google Translate sync error:', e);
    }
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
  const [preferredLang, setPreferredLang] = useState(getInitialLanguage);
  const [showMoreMenu, setShowMoreMenu] = useState(false);

  const { floatingStyles } = useFloating({
    placement: "bottom-end",
    middleware: [
      offset(8),
      flip(),
      shift({ padding: 10 })
    ],
    whileElementsMounted: autoUpdate
  });
  const location = useLocation();

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
      if (process.env.NODE_ENV !== 'production') {
        console.error("Logout error:", error);
      }
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
    const handleClickOutside = (event) => {
      if (scorecardRef.current && !scorecardRef.current.contains(event.target)) {
        setShowScorecard(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

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

  const [isOffline, setIsOffline] = useState(false);

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

  const [showScrollTop, setShowScrollTop] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      if (window.scrollY > 300) {
        setShowScrollTop(true);
      } else {
        setShowScrollTop(false);
      }
    };

    handleScroll();
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const scrollToTop = () => {
    window.scrollTo({
      top: 0,
      behavior: "smooth",
    });
  };

  return (
    <div className={`app ${isDarkTheme ? "theme-dark" : ""}`}>
      <SkipLink />

       {loading && <Loader fullPage={true} message={<span className="notranslate">Initializing Fasal Saathi...</span>} />}
       {isOffline && (
         <div className="offline-banner" role="alert">
           You are currently offline. Running in offline mode using local data.
         </div>
       )}

      {/* 
        ACCESSIBILITY IMPROVEMENT: Semantic Navigation
        The 'nav' element defines a landmark region for screen readers.
        Using Link components from react-router-dom ensures client-side routing.
      */}
        <nav className={`navbar ${isOpen ? "menu-open" : ""}`} role="navigation" aria-label="Main Navigation">
          <div className="nav-left">
            {/* 
              Brand Link: Directs users back to the landing page.
              We ensure this is the first link in the tab order for consistency.
            */}
             <Link to="/" className="brand" aria-label="Fasal Saathi Home">
               <FaLeaf className="brand-icon" />
               <span className="notranslate" translate="no">Fasal Saathi</span>
             </Link>
           </div>

        <ul className={`nav-center ${isOpen ? "active" : ""}`}>
          <li>
            <Link
              to="/"
              onClick={() => setIsOpen(false)}
              aria-label="Navigate to Home Page"
            >
              <span className="notranslate">Home</span>
            </Link>
          </li>
          <li>
            <Link
              to="/how-it-works"
              onClick={() => setIsOpen(false)}
              aria-label="Learn how Fasal Saathi works"
            >
              <span className="notranslate">Works</span>
            </Link>
          </li>
          <li>
            <Link
              to="/crop-guide"
              onClick={() => setIsOpen(false)}
              aria-label="View our comprehensive Crop Guide"
            >
              <span className="notranslate">Guide</span>
            </Link>
          </li>
          <li>
            <Link
              to="/resources"
              onClick={() => setIsOpen(false)}
              aria-label="Access farming resources and materials"
            >
              <span className="notranslate">Resources</span>
            </Link>
           </li>
           <li>
             <Link
               to="/about"
               onClick={() => setIsOpen(false)}
               aria-label="Learn about our mission and team"
             >
               <span className="notranslate">About</span>
             </Link>
           </li>
           <li>
             <Link
               to="/contact"
               onClick={() => setIsOpen(false)}
               aria-label="Get in touch with us"
             >
               <span className="notranslate">Contact</span>
             </Link>
           </li>
           <li className="mobile-only-language">
            <div className="language-selector-section">
              <label className="language-label">Language:</label>
              <LanguageDropdown 
                options={LANGUAGE_OPTIONS}
                value={settings.language}
                onChange={(lang) => {
                  setSettings({ ...settings, language: lang });
                  syncLanguage(lang, setPreferredLang);
                }}
              />
            </div>
          </li>
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
                    {/* Language Selector in hamburger menu */}
                    <div className="language-selector-section">
                      <label className="language-label">Language:</label>
                      <LanguageDropdown 
                        options={LANGUAGE_OPTIONS}
                        value={settings.language}
                        onChange={(lang) => {
                          setSettings({ ...settings, language: lang });
                          syncLanguage(lang, setPreferredLang);
                        }}
                      />
                    </div>
                    {/* 
                      Internal App Links:
                      Using Dashboard and Community links within the dropdown.
                      We ensure these are also focusable and labeled correctly.
                    */}
                     <Link
                       to="/dashboard"
                       onClick={() => setShowMoreMenu(false)}
                       role="menuitem"
                       aria-label="Go to Farmer Dashboard"
                     >
                       <FaTachometerAlt aria-hidden="true" /> <span className="notranslate">Dashboard</span>
                     </Link>
                     <Link
                       to="/community"
                       onClick={() => setShowMoreMenu(false)}
                       role="menuitem"
                       aria-label="Join our Community forum"
                     >
                       <FaComments aria-hidden="true" /> <span className="notranslate">Community</span>
                     </Link>
                     <Link
                       to="/glossary"
                       onClick={() => setShowMoreMenu(false)}
                       role="menuitem"
                       aria-label="View Agricultural Glossary"
                     >
                       <FaBook aria-hidden="true" /> <span className="notranslate">Glossary</span>
                     </Link>
                     <Link
                       to="/risk-index"
                       onClick={() => setShowMoreMenu(false)}
                       role="menuitem"
                       aria-label="Access AI Risk Index"
                     >
                       <FaShieldAlt aria-hidden="true" /> <span className="notranslate">Risk Index</span>
                     </Link>
                  </div>
                </div>
              )}

           <div className="nav-user" ref={scorecardRef} onClick={() => { setShowScorecard(!showScorecard); setShowMoreMenu(false); }}>
            {loading ? (
              <div className="nav-loader-mini"></div>
            ) : user ? (
              <div className="user-profile-trigger">
                <div className="profile-main">
                  <span className="profile-name">{userData?.displayName || user.email?.split('@')?.[0] || "Farmer"}</span>
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
              /* 
                Login Link: Prominently displayed for guest users.
                This is often the primary call-to-action in the navbar.
              */
              <Link
                to="/login"
                className="btn-get-started"
                aria-label="Log in or Sign up to get started"
              >
                <span className="notranslate">Get Started</span>
              </Link>
            )}
          </div>
        </div>
        <button
          className="hamburger"
          onClick={handleNavToggle}
        >
          {isOpen ? <FaTimes /> : <FaBars />}
        </button>
      </nav>

        {!loading && user && !user.emailVerified && !showScorecard && window.location.pathname !== "/login" && (
          <div className="verification-overlay">
            <div className="verification-card">
              <div className="verify-icon">✉️</div>
              <h2>Verify Your Email</h2>
              <p>We've sent a link to <b>{user.email}</b>.<br /> Please verify your email to unlock all features.</p>
              <button
                 onClick={() => {
                   auth?.currentUser?.reload().then(() => window.location.reload()).catch(() => window.location.reload());
                 }}
                 className="btn-refresh"
              >
                I've Verified My Email
              </button>
               <button onClick={handleLogout} className="btn-logout-simple">Sign Out</button>
             </div>
           </div>
          )}

        {!loading && user && user.emailVerified && !profileCompleted && window.location.pathname !== "/profile-setup" && (
          <Navigate to="/profile-setup" />
        )}

      {/* 
        MAIN CONTENT AREA:
        We wrap the Routes in a <main> element with an ID of 'main-content'.
        This serves as the target for our SkipLink, allowing users to bypass 
        the header and navigation menu.
      */}
      <main id="main-content" tabIndex="-1" style={{ outline: 'none' }}>
        <Routes>
          <Route path="/" element={<Home user={user} />} />
          <Route path="/advisor" element={<Advisor />} />
          <Route path="/how-it-works" element={<How />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/crop-guide" element={<CropGuide />} />
          <Route path="/schemes" element={<Schemes />} />
          <Route path="/resources" element={<Resources />} />
          <Route path="/contributors" element={<Contributors />} />
          <Route path="/login" element={<Auth />} />
          <Route path="/profile-setup" element={<ProfileSetup user={user} profileCompleted={profileCompleted} />} />
          <Route path="/calendar" element={<Calendar />} />
          <Route path="/share-feedback" element={<Feedback />} />
          <Route path="/admin/feedback" element={<AdminFeedback />} />
          <Route path="/market-prices" element={<MarketPrices />} />
          <Route path="/farming-map" element={<FarmingMap />} />
          <Route path="/profit-calculator" element={<CropProfitCalculator />} />
          <Route path="/community" element={<Community />} />
          <Route path="/contact" element={<ContactUs />} />
          <Route path="/about" element={<AboutUs />} />
          <Route path="/crop-planner" element={<SeasonalCropPlanner />} />
          <Route path="/soil-guide" element={<SoilGuide />} />
          <Route path="/disease-awareness" element={<CropDiseaseAwareness />} />
          <Route path="/helpline" element={<Helpline />} />
           <Route path="/glossary" element={<Glossary />} />
           <Route path="/risk-index" element={<RiskIndex />} />
            <Route path="/blog" element={<Blog />} />
           <Route path="/blog/:id" element={<BlogDetail />} />
           <Route path="/faq" element={<FAQ />} />
          <Route path="/privacy-policy" element={<PrivacyPolicy />} />
          <Route path="/terms" element={<Terms />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </main>

      {/* 
        Floating Action Button:
        A fixed-position link for quick access to the AI advisor.
        We provide a descriptive aria-label and ensure icons are hidden from screen readers
        to avoid redundant announcements.
      */}
      <Link 
        to="/advisor" 
        className="floating-chat-btn" 
        aria-label="Open AI Advisor Chat"
      >
        <FaComments size={28} aria-hidden="true" />
      </Link>

      <a 
        href="https://wa.me/14155238886?text=I%20want%20to%20start%20the%20conversation%20for%20real-time%20data%20sharing" 
        target="_blank" 
        rel="noopener noreferrer" 
        className="whatsapp-float"
        title="Chat with WhatsApp Bot"
      >
        <FaWhatsapp />
        <span className="tooltip">Chat with Bot</span>
      </a>

      {showScrollTop && (
        <button 
          className="scroll-to-top" 
          onClick={scrollToTop}
          aria-label="Scroll to top"
          title="Scroll to top"
        >
          <FaChevronUp size={24} />
        </button>
      )}

      <ToastContainer position="bottom-right" />
    </div>
  );
}

export default App;
