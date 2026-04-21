import React, { useState, useEffect } from "react";
import { Routes, Route, Link, Navigate, useLocation } from "react-router-dom";
import { onAuthStateChanged, signOut } from "firebase/auth";
import { doc, onSnapshot } from "firebase/firestore";
import { 
  FaHome, 
  FaComments, 
  FaInfoCircle, 
  FaLeaf, 
  FaBars, 
  FaTimes, 
  FaChevronDown 
} from "react-icons/fa";
import { ToastContainer } from "react-toastify";

import Advisor from "./Advisor";
import Home from "./Home";
import Resources from "./Resources";
import CropGuide from "./CropGuide";
import How from "./How";
import Auth from "./Auth";
import ProfileSetup from "./ProfileSetup";
import LanguageDropdown from "./LanguageDropdown";
import useNotifications from "./Notifications";

import { auth, db, isFirebaseConfigured } from "./lib/firebase";

import "./App.css";
import "./themes/sunlight.css";

/* ---------------- LANGUAGE options ---------------- */

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
  const el = document.querySelector(".goog-te-combo");
  if (!el) return false;
  el.value = lang;
  el.dispatchEvent(new Event("change"));
  return true;
};

const syncLanguage = (lang, setLang) => {
  setLang(lang);
  localStorage.setItem("preferredLanguage", lang);
  applyGoogleTranslate(lang);
};

/* ---------------- APP MAIN ---------------- */

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
    try {
      localStorage.setItem("theme", isDarkTheme ? "dark" : "light");
    } catch {
      // ignore
    }
  }, [isDarkTheme]);

  const handleThemeToggle = () => setIsDarkTheme((prev) => !prev);

  /* ---------------- LANGUAGE AUTO-TRANS ---------------- */
  useEffect(() => {
    if (applyGoogleTranslate(preferredLang)) return;
    const id = setInterval(() => {
      if (applyGoogleTranslate(preferredLang)) clearInterval(id);
    }, 300);
    return () => clearInterval(id);
  }, [preferredLang]);

   useEffect(() => {
    setGoogleTranslateCookie(preferredLang);

    if (applyGoogleTranslate(preferredLang)) return;

    const id = setInterval(() => {
      if (applyGoogleTranslate(preferredLang)) clearInterval(id);
    }, 300);

    return () => clearInterval(id);
  }, [preferredLang]);

  /* ---------------- TRANSLATION TOOLBAR DETECTION ---------------- */
  // Detects when Chrome's translation toolbar is present and adjusts layout accordingly
  // This prevents UI layout breaks when browser translation is enabled
  useEffect(() => {
    const cleanupGoogleTranslate = () => {
      const selectors = [
        '.goog-te-banner-frame',
        '.goog-te-balloon-frame', 
        '#goog-gt-tt',
        'iframe[src*="translate.google"]',
        '.VIpgJd-ZVi9od-ORHb-OEVgZj'
      ];
      selectors.forEach(sel => {
        document.querySelectorAll(sel).forEach(el => {
          el.style.display = 'none';
          el.style.visibility = 'hidden';
          el.style.height = '0';
          el.style.width = '0';
          el.style.position = 'absolute';
          el.style.pointerEvents = 'none';
        });
      });
      document.body.style.marginTop = '0';
      document.body.style.paddingTop = '0';
      document.body.style.transform = 'none';
    };

    const detectTranslationToolbar = () => {
      cleanupGoogleTranslate();
      // Multiple detection methods for translation toolbar
      const hasTranslationToolbar =
        // Check for Google Translate banner/frame
        document.querySelector('.goog-te-banner-frame') ||
        document.querySelector('.goog-te-gadget') ||
        document.querySelector('[data-ogpc]') || // Google Translate attribute
        // Check if body has translation-related transforms
        (document.body.style.transform && document.body.style.transform.includes('translateY')) ||
        (document.body.style.marginTop && parseInt(document.body.style.marginTop) > 0) ||
        // Check for translation meta tags
        document.querySelector('meta[name="google-translate-customization"]') ||
        // Check if the page height has changed significantly (toolbar pushes content down)
        (window.innerHeight < window.screen.height * 0.9 && document.documentElement.scrollHeight > window.innerHeight);

      document.documentElement.classList.toggle('has-translation-toolbar', hasTranslationToolbar);
    };

    // Initial check
    detectTranslationToolbar();

    // Run cleanup immediately then check periodically
    cleanupGoogleTranslate();
    const interval = setInterval(() => {
      cleanupGoogleTranslate();
      detectTranslationToolbar();
    }, 500);

    // Check on various events that might indicate translation
    const handleVisibilityChange = () => setTimeout(detectTranslationToolbar, 500);
    const handleFocus = () => setTimeout(detectTranslationToolbar, 200);

    document.addEventListener('visibilitychange', handleVisibilityChange);
    window.addEventListener('focus', handleFocus);
    window.addEventListener('resize', detectTranslationToolbar);

    // Also run cleanup on any user interaction that might trigger translate
    const handleClick = () => cleanupGoogleTranslate();
    const handleScroll = () => cleanupGoogleTranslate();
    document.addEventListener('click', handleClick, true);
    document.addEventListener('scroll', handleScroll, true);

    // Check for DOM changes that might indicate translation
    const observer = new MutationObserver((mutations) => {
      let shouldCheck = false;
      mutations.forEach((mutation) => {
        if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
          Array.from(mutation.addedNodes).forEach((node) => {
            if (node.nodeType === Node.ELEMENT_NODE &&
                (node.classList?.contains('goog-te') ||
                 node.id?.includes('google_translate') ||
                 node.tagName === 'IFRAME')) {
              shouldCheck = true;
            }
          });
        }
        if (mutation.type === 'attributes' &&
            (mutation.attributeName === 'style' || mutation.attributeName === 'class')) {
          shouldCheck = true;
        }
      });
      if (shouldCheck) detectTranslationToolbar();
    });

    observer.observe(document.body, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ['style', 'class', 'id']
    });

return () => {
      clearInterval(interval);
      document.removeEventListener('visibilitychange', handleVisibilityChange);
      document.removeEventListener('focus', handleFocus);
      document.removeEventListener('resize', detectTranslationToolbar);
      document.removeEventListener('click', handleClick, true);
      document.removeEventListener('scroll', handleScroll, true);
      observer.disconnect();
    };
  }, []);

  const handleLogout = async () => {
    if (!isFirebaseConfigured() || !auth) {
      window.location.href = "/";
      return;
    }
    try {
      await signOut(auth);
      window.location.href = "/";
    } catch (error) {
      console.error("Sign out error:", error);
    }
  };

  /* ---------------- AUTH STATE LISTENER ---------------- */
  useEffect(() => {
    if (!isFirebaseConfigured()) {
      setLoading(false);
      return;
    }
    const unsubscribeAuth = auth?.onAuthStateChanged ? auth.onAuthStateChanged((currentUser) => {
      setUser(currentUser);
      if (currentUser) {
        const unsubscribeDoc = db && onSnapshot(doc(db, "users", currentUser.uid), (userDoc) => {
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
        return () => unsubscribeDoc?.();
      } else {
        setUserData(null);
        setProfileCompleted(true);
        setLoading(false);
      }
    }) : () => {};
    return () => unsubscribeAuth();
  }, []);


  /* ---------------- OFFLINE STATUS ---------------- */
  const [isOffline, setIsOffline] = useState(!navigator.onLine);

  useEffect(() => {
    const handleNetworkChange = () => setIsOffline(!navigator.onLine);

    window.addEventListener("online", handleNetworkChange);
    window.addEventListener("offline", handleNetworkChange);
    
    // Polling fallback to detect DevTools offline toggling where the event might be suppressed
    const interval = setInterval(handleNetworkChange, 1000);

    return () => {
      window.removeEventListener("online", handleNetworkChange);
      window.removeEventListener("offline", handleNetworkChange);
      clearInterval(interval);
    };
  }, []);

  return (
    <div className="app">
      {/* OFFLINE INDICATOR */}
      {isOffline && (
        <div className="offline-banner">
          ⚠️ You are currently offline. Running in offline mode using local data.
        </div>
      )}

      {/* PROFESSIONAL NAVBAR */}
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
        </ul>

        <div className="nav-right">
          <button onClick={handleThemeToggle} className="theme-toggle" aria-label="Toggle Theme">
            {isDarkTheme ? "☀️" : "🌙"}
          </button>

          <LanguageDropdown
            options={LANGUAGE_OPTIONS}
            value={preferredLang}
            onChange={(val) => syncLanguage(val, setPreferredLang)}
          />

          <div className="nav-user" onClick={() => setShowScorecard(!showScorecard)}>
            {loading ? (
              <span className="loading-text">Loading...</span>
            ) : user ? (
              <div className="user-profile-trigger">
                <div className="profile-main">
                  <span className="profile-name">👋 {userData?.displayName || user.email?.split('@')[0]}</span>
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
                        { label: "🌾 Primary Crop", value: userData.cropType },
                        { label: "🌐 Language", value: LANGUAGE_OPTIONS.find(l => l.value === userData.language)?.label || userData.language },
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

      {/* PROFILE COMPLETION GUARD */}
      {!loading && user && user.emailVerified && !profileCompleted && location.pathname !== "/profile-setup" && (
        <Navigate to="/profile-setup" />
      )}

      {/* APP ROUTES */}
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/advisor" element={<Advisor />} />
        <Route path="/how-it-works" element={<How />} />
        <Route path="/crop-guide" element={<CropGuide />} />
        <Route path="/resources" element={<Resources />} />
        <Route path="/login" element={<Auth />} />
        <Route path="/profile-setup" element={<ProfileSetup />} />
      </Routes>

      <ToastContainer position="bottom-right" />
    </div>
  );
}

export default App;
