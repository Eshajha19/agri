import React, { useEffect, useState, useRef } from "react";
import { Routes, Route, Link, Navigate, useLocation } from "react-router-dom";
import { SkipLink } from "./NavigationManager";

import { ToastContainer } from "react-toastify";
import { useFloating, flip, shift, offset, autoUpdate } from "@floating-ui/react";
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
  FaChevronDown,
  FaChevronUp,
  FaWhatsapp,
  FaUser,
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
import Community from "./Community";
import ContactUs from "./ContactUs";
import AboutUs from "./AboutUs";
import Contributors from "./Contributors";
import FAQ from "./FAQ";
import NotFound from "./NotFound";

import { syncOfflineRequests } from "./lib/syncOfflineRequests";
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
  { value: "gu", label: "🇮🇳 ગુજરાતી", englishName: "gujarati" },
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

const syncLanguage = (lang, setLang) => {
  setLang(lang);
  localStorage.setItem("preferredLanguage", lang);

  // Set cookie WITHOUT encoding first so Google can read it
  if (lang === 'en') {
    // Clear cookie for English
    document.cookie = 'googtrans=; path=/; max-age=0';
    if (window.location.hostname) {
      document.cookie = 'googtrans=; domain=.' + window.location.hostname + '; path=/; max-age=0';
    }
  } else {
    const rawCookieValue = '/en/' + lang;
    const expires = new Date();
    expires.setFullYear(expires.getFullYear() + 10);
    // Set raw value (no encoding)
    document.cookie = 'googtrans=' + rawCookieValue + '; path=/; expires=' + expires.toUTCString();
    if (window.location.hostname) {
      document.cookie = 'googtrans=' + rawCookieValue + '; domain=.' + window.location.hostname + '; path=/; expires=' + expires.toUTCString();
    }
  }
  // Use a small delay to ensure cookies are set before reload
  setTimeout(() => window.location.reload(), 50);
};

function App() {
  const scorecardRef = useRef(null);
  const [preferredLang, setPreferredLang] = useState(getInitialLanguage);
  const [isOpen, setIsOpen] = useState(false);
  const [user, setUser] = useState(null);
  const [userData, setUserData] = useState(null);
  const [profileCompleted, setProfileCompleted] = useState(true);
  const [loading, setLoading] = useState(true);
  const [showScorecard, setShowScorecard] = useState(false);
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
      <nav className="navbar" role="navigation" aria-label="Main Navigation">
          <div className="nav-left">
            {/* 
              Brand Link: Directs users back to the landing page.
              We ensure this is the first link in the tab order for consistency.
            */}
            <Link to="/" className="brand" aria-label="Fasal Saathi Home">
              <FaLeaf className="brand-icon" />
              <span>Fasal Saathi</span>
            </Link>
          </div>

        <ul className={`nav-center ${isOpen ? "active" : ""}`}>
          {/* 
            Navigation List: Organized as a list (ul) so screen readers 
            can announce the number of items in the menu.
          */}
          <li>
            <Link 
              to="/" 
              onClick={() => setIsOpen(false)}
              aria-label="Navigate to Home Page"
            >
              Home
            </Link>
          </li>
          <li>
            <Link 
              to="/how-it-works" 
              onClick={() => setIsOpen(false)}
              aria-label="Learn how Fasal Saathi works"
            >
              Works
            </Link>
          </li>
          <li>
            <Link 
              to="/crop-guide" 
              onClick={() => setIsOpen(false)}
              aria-label="View our comprehensive Crop Guide"
            >
              Guide
            </Link>
          </li>
          <li>
            <Link 
              to="/resources" 
              onClick={() => setIsOpen(false)}
              aria-label="Access farming resources and materials"
            >
              Resources
            </Link>
          </li>
          <li>
            <Link 
              to="/about" 
              onClick={() => setIsOpen(false)}
              aria-label="Learn about our mission and team"
            >
              About
            </Link>
          </li>
          <li>
            <Link 
              to="/contact" 
              onClick={() => setIsOpen(false)}
              aria-label="Get in touch with us"
            >
              Contact
            </Link>
          </li>
        </ul>

          <div className="nav-right">
            <button onClick={handleThemeToggle} className="theme-toggle" aria-label="Toggle Theme">
              {isDarkTheme ? "☀️" : "🌙"}
            </button>

            <button onClick={() => setShowMoreMenu(!showMoreMenu)} className="more-menu-toggle" aria-label="More Options">
              <FaBars />
            </button>

            {showMoreMenu && (
              <div className="more-dropdown" onClick={(e) => e.stopPropagation()} role="menu">
                 <div className="dropdown-section">
                   <label id="language-label">Language</label>
                   <LanguageDropdown
                     options={LANGUAGE_OPTIONS}
                     value={preferredLang}
                     onChange={(lang) => {
                       syncLanguage(lang, setPreferredLang);
                       setShowMoreMenu(false);
                     }}
                     aria-labelledby="language-label"
                   />
                 </div>
                <div className="dropdown-links">
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
                    <FaTachometerAlt aria-hidden="true" /> Dashboard
                  </Link>
                  <Link 
                    to="/community" 
                    onClick={() => setShowMoreMenu(false)}
                    role="menuitem"
                    aria-label="Join our Community forum"
                  >
                    <FaComments aria-hidden="true" /> Community
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
                Get Started
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
          <Route path="/dashboard" element={!loading && !user ? <Navigate to="/login" state={{ from: location }} replace /> : <Dashboard />} />
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
          <Route path="/faq" element={<FAQ />} />
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
