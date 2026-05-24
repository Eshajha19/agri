import React, { Suspense, useEffect, useState, useRef } from "react";
import { Routes, Route, Link, Navigate, useLocation, useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import {
  FaComments,
  FaLeaf,
  FaTachometerAlt,
  FaTimes,
  FaBars,
  FaChevronDown,
  FaChevronUp,
  FaWhatsapp,
  FaBook,
  FaShieldAlt,
  FaBolt,
  FaUserSecret,
  FaFileInvoiceDollar,
  FaTrophy,
  FaUserPlus,
  FaMedal,
  FaCog,
  FaMicrophone,
  FaInfoCircle
} from "react-icons/fa";
import { usePerformanceStore } from "./stores/performanceStore";
import { useBrowserCacheBudget } from "./lib/cacheBudget";
import { cryptoService } from "./utils/cryptoService";
// Components
import Loader from "./Loader";
import LanguageDropdown from "./LanguageDropdown";
import useNotifications from "./Notifications";
import Footer from "./components/Footer";
import { SkipLink } from "./NavigationManager";
import { useTheme } from "./ThemeContext";

// Route-level code splitting
import {
  AdminFeedback,
  Advisor,
  Auth,
  AboutUs,
  Blog,
  BlogDetail,
  Calendar,
  Community,
  Contributors,
  ContactUs,
  CropDiseaseAwareness,
  CropGuide,
  CropProfitCalculator,
  CropRotation,
  Dashboard,
  FAQ,
  FarmFinance,
  FarmingMap,
  FarmingNews,
  Feedback,
  Glossary,
  Helpline,
  Home,
  How,
  Leaderboard,
  MarketPrices,
  NotFound,
  PestDetection,
  PrivacyPolicy,
  ProfileSetup,
  ProfileSettings,
  QRTraceability,
  ReferralHub,
  Resources,
  RiskIndex,
  Schemes,
  SeasonalCropPlanner,
  SeedVerifier,
  SmartFarmAutopilot,
  SoilAnalysis,
  SoilGuide,
  SustainabilityAnalytics,
  Terms,
  YieldPredictor,
  EquipmentManagement,
} from "./routes/lazyPages";

const Weather = React.lazy(() => import("./Weather"));
import VoiceAssistant from "./VoiceAssistant";

/**
 * Thin wrapper so SustainabilityAnalytics (designed as a modal) works as a
 * full standalone route. The onClose prop navigates the user back.
 */
function SustainabilityAnalyticsPage({ userData }) {
  const navigate = useNavigate();
  return <SustainabilityAnalytics userData={userData} onClose={() => navigate(-1)} />
}

// Libs
import { auth, db, isFirebaseConfigured, doc, onSnapshot, setDoc, getDoc } from "./lib/firebase";
import { onAuthStateChanged, signOut } from "firebase/auth";
import { loadAppState, loadUserProfileSnapshot, persistAppState, persistUserProfileSnapshot } from "./lib/offlinePersistence";
import { syncOfflineRequests } from "./lib/syncOfflineRequests";

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

const normalizeUserProfile = (profile) => {
  if (!profile) return profile;

  return {
    ...profile,
    farmArea: profile.farmArea ?? profile.farmSize ?? "",
    irrigationType: profile.irrigationType ?? profile.irrigationMethod ?? "",
  };
};

/**
 * Helper to apply Google Translate selection to the hidden widget
 * Uses MutationObserver for reliable widget detection instead of polling
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

/**
 * Robustly wait for Google Translate widget using MutationObserver
 * Returns a promise that resolves when widget is ready or times out
 */
const waitForGoogleTranslateWidget = (timeoutMs = 15000) => {
  return new Promise((resolve, reject) => {
    const existingWidget = document.querySelector(".goog-te-combo");
    if (existingWidget) {
      resolve(existingWidget);
      return;
    }

    let observer = null;
    const timeoutId = setTimeout(() => {
      if (observer) observer.disconnect();
      reject(new Error("Google Translate widget not found within timeout"));
    }, timeoutMs);

    observer = new MutationObserver((mutations) => {
      const widget = document.querySelector(".goog-te-combo");
      if (widget) {
        clearTimeout(timeoutId);
        observer?.disconnect();
        resolve(widget);
      }
    });

    observer.observe(document.body, {
      childList: true,
      subtree: true,
    });
  });
};

/**
 * Apply translation with robust widget detection
 */
const applyGoogleTranslateRobust = async (langCode, onReady, onError) => {
  try {
    await waitForGoogleTranslateWidget(15000);
    applyGoogleTranslate(langCode);
    onReady?.();
  } catch (error) {
    console.warn("Google Translate widget initialization failed:", error.message);
    onError?.(error);
  }
};

const GuestBanner = () => (
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
  const [preferredLang, setPreferredLang] = useState(() => {
    try {
      return localStorage.getItem("agri:preferredLanguage") || getInitialLanguage();
    } catch {
      return getInitialLanguage();
    }
  });
  const [isOpen, setIsOpen] = useState(false);
  const { theme, toggleTheme, setTheme } = useTheme();
  const [user, setUser] = useState(null);
  const [userData, setUserData] = useState(null);
  const [profileCompleted, setProfileCompleted] = useState(true);
  const [loading, setLoading] = useState(true);
  const [showScorecard, setShowScorecard] = useState(false);
  const [showMoreMenu, setShowMoreMenu] = useState(false);
  const [isOffline, setIsOffline] = useState(!navigator.onLine);
  const [showScrollTop, setShowScrollTop] = useState(false);
  const [scrollProgress, setScrollProgress] = useState(0);

  const { liteMode, setLiteMode, detectAndSetLiteMode } = usePerformanceStore();

  useEffect(() => {
    detectAndSetLiteMode();
  }, [detectAndSetLiteMode]);

  useEffect(() => {
    let cancelled = false;

    const hydrateOfflineState = async () => {
      try {
        const storedState = await loadAppState();
        if (!cancelled && storedState?.preferredLang) {
          setPreferredLang(storedState.preferredLang);
        }
      } catch (error) {
        console.warn("Failed to restore offline app state:", error);
      }
    };

    const syncQueuedRequests = async () => {
      try {
        await syncOfflineRequests();
      } catch (error) {
        console.warn("Offline request sync failed:", error);
      }
    };

    void hydrateOfflineState();
    void syncQueuedRequests();

    const handleOnline = () => {
      setIsOffline(false);
      void syncQueuedRequests();
    };

    const handleOffline = () => setIsOffline(true);

    window.addEventListener("online", handleOnline);
    window.addEventListener("offline", handleOffline);

    return () => {
      cancelled = true;
      window.removeEventListener("online", handleOnline);
      window.removeEventListener("offline", handleOffline);
    };
  }, []);

  useEffect(() => {
    void persistAppState({ preferredLang });
  }, [preferredLang]);

  const { i18n } = useTranslation();
  const location = useLocation();

  useNotifications();
  useBrowserCacheBudget({
    enabled: true,
    usageRatioLimit: liteMode ? 0.72 : 0.85,
  });

  /* ---------------- THEME SYSTEM (Moved to ThemeProvider) ---------------- */

  /* ---------------- LANGUAGE AUTO-TRANS ---------------- */
  useEffect(() => {
    if (applyGoogleTranslate(preferredLang)) return;

    let retries = 0;
    const MAX_RETRIES = 20; // Try for ~6 seconds

    const id = setInterval(() => {
      retries++;
      if (applyGoogleTranslate(preferredLang)) {
        clearInterval(id);
      } else if (retries >= MAX_RETRIES) {
        clearInterval(id);
        console.warn("Google Translate widget initialization timed out or was blocked. Graceful fallback applied.");
      }
    }, 300);

    return () => clearInterval(id);
    const applyTranslation = async () => {
      if (applyGoogleTranslate(preferredLang)) return;

      try {
        await applyGoogleTranslateRobust(
          preferredLang,
          () => console.log("Google Translate initialized successfully"),
          () => console.warn("Google Translate unavailable - using default language")
        );
      } catch (error) {
        console.warn("Translation initialization failed - graceful fallback applied");
      }
    };

    applyTranslation();

    const handleWidgetLoad = () => {
      if (!applyGoogleTranslate(preferredLang)) {
        applyGoogleTranslateRobust(preferredLang);
      }
    };

    const widgetCheckInterval = setInterval(() => {
      if (document.querySelector(".goog-te-combo") && !applyGoogleTranslate(preferredLang)) {
        applyGoogleTranslateRobust(preferredLang);
      }
    }, 2000);

    document.addEventListener("googleTranslateWidgetLoaded", handleWidgetLoad);

    return () => {
      clearInterval(widgetCheckInterval);
      document.removeEventListener("googleTranslateWidgetLoaded", handleWidgetLoad);
    };
  }, [preferredLang]);

  /* ---------------- HIDE GOOGLE TRANSLATE BANNER ---------------- */
  useEffect(() => {
    const hideGoogleTranslateBanner = () => {
      const bannerFrame = document.querySelector(".goog-te-banner-frame");
      if (bannerFrame) {
        bannerFrame.style.display = "none";
      }

      document.body.style.top = "0px";

      const translateElement = document.querySelector(".goog-te-balloon-frame");
      if (translateElement) {
        translateElement.style.display = "none";
      }
    };

    hideGoogleTranslateBanner();

    const interval = setInterval(hideGoogleTranslateBanner, 1000);

    return () => clearInterval(interval);
  }, []);

  /* ---------------- AUTH & FIRESTORE SYNC ---------------- */
  useEffect(() => {
    if (!isFirebaseConfigured()) {
      setLoading(false);
      return;
    }

    const userDocUnsubscribeRef = { current: null };

    const unsubscribeAuth = onAuthStateChanged(auth, (currentUser) => {
      setUser(currentUser);

      const hydrateUserSnapshot = async () => {
        if (!currentUser?.uid) return false;
        try {
          const snapshot = await loadUserProfileSnapshot(currentUser.uid);
          if (snapshot) {
            setUserData(normalizeUserProfile(snapshot));
            setProfileCompleted(snapshot.profileCompleted === true);
            return true;
          }
        } catch (error) {
          console.warn("Failed to restore offline user profile snapshot:", error);
        }
        return false;
      };

      if (currentUser) {
        userDocUnsubscribeRef.current = onSnapshot(doc(db, "users", currentUser.uid), (userDoc) => {
          if (userDoc.exists()) {
            const data = normalizeUserProfile(userDoc.data());
            setUserData(data);
            setProfileCompleted(data.profileCompleted === true);
          } else if (currentUser.isAnonymous) {
            setUserData({ displayName: "Guest Farmer", isAnonymous: true });
            setProfileCompleted(true);
          } else {
            setUserData(null);
            setProfileCompleted(false);
            void hydrateUserSnapshot().finally(() => setLoading(false));
            return;
          }
          setLoading(false);
        }, (error) => {
          console.error("Firestore sync error:", error);
          setUserData(null);
          setProfileCompleted(false);
          void hydrateUserSnapshot().finally(() => setLoading(false));
        });
      } else {
        setUserData(null);
        setProfileCompleted(true);
        setLoading(false);
      }
    });

    return () => {
      unsubscribeAuth();
      if (userDocUnsubscribeRef.current) {
        userDocUnsubscribeRef.current();
      }
    };
  }, []);

  // E2EE Key Generation Sync
  useEffect(() => {
    if (!user || !isFirebaseConfigured()) return;

    const ensurePublicKey = async () => {
      try {
        let { publicJwk } = await cryptoService.ensureKeys(user.uid);

        if (!publicJwk) {
          const publicKeySnap = await getDoc(doc(db, "public_keys", user.uid));
          if (publicKeySnap.exists()) {
            publicJwk = publicKeySnap.data().jwk;
            await cryptoService.savePublicKey(user.uid, publicJwk);
          }
        }

        if (!publicJwk) {
          throw new Error("ECDH public key unavailable after initialization");
        }

        const pubKeyRef = doc(db, "public_keys", user.uid);
        await setDoc(pubKeyRef, { jwk: publicJwk }, { merge: true });
      } catch (error) {
        console.error("Failed to generate/publish ECDH keys globally:", error);
      }
    };

    ensurePublicKey();
  }, [user]);

  useEffect(() => {
    if (!user?.uid || !userData) return;

    void persistUserProfileSnapshot(user.uid, {
      ...normalizeUserProfile(userData),
      profileCompleted,
      savedAt: new Date().toISOString(),
    });
  }, [user?.uid, userData, profileCompleted]);

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
      // Calculate scroll progress
      const totalHeight = document.documentElement.scrollHeight - window.innerHeight;
      const progress = totalHeight > 0 ? (window.scrollY / totalHeight) * 100 : 0;
      setScrollProgress(progress);
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

  const handleThemeToggle = toggleTheme;
  const handleThemeSelect = (nextTheme) => {
    setTheme(nextTheme);
    setShowMoreMenu(false);
  };
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
    <div className={`app ${theme !== "light" ? "theme-dark" : ""} ${theme === "night" ? "theme-night" : ""} ${liteMode ? "lite-mode" : ""}`}>
      {user?.isAnonymous && <GuestBanner />}

      {loading && <Loader fullPage={true} message={<span className="notranslate">Initializing Fasal Saathi...</span>} />}

      {isOffline && (
        <div className="offline-banner" role="alert">
          You are currently offline. Running in offline mode using local data.
        </div>
      )}

      {/* Scroll Progress Bar */}
      <div className="scroll-progress-bar" style={{ width: `${scrollProgress}%` }} aria-hidden="true" />

      <nav className={`navbar ${isOpen ? "menu-open" : ""}`} role="navigation" aria-label="Main Navigation">
        <div className="nav-left">
          <Link to="/" className="brand">Fasal Saathi</Link>
        </div>

        <ul className={`nav-center ${isOpen ? "active" : ""}`}>
          <li><Link to="/" onClick={() => setIsOpen(false)}>Home</Link></li>
          <li><Link to="/about" onClick={() => setIsOpen(false)}>About</Link></li>
          <li><Link to="/how-it-works" onClick={() => setIsOpen(false)}>How It Works</Link></li>
          <li><Link to="/crop-guide" onClick={() => setIsOpen(false)}> Crop Guide</Link></li>
          <li><Link to="/resources" onClick={() => setIsOpen(false)}>Resources</Link></li>
        </ul>

        <div className="nav-right">
          <button onClick={handleThemeToggle} className="theme-toggle" aria-label="Cycle Theme" title={`Current theme: ${theme}`}>
            {theme === "light" ? "🌙" : theme === "dark" ? "☀️" : "🌙"}
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
                      void persistAppState({ preferredLang: lang });
                    }}
                  />
                </div>
                <div className="theme-selector-section">
                  <span className="theme-selector-label">Theme:</span>
                  <div className="theme-option-grid" role="group" aria-label="Theme selection">
                    {[
                      { value: "light", label: "Light", icon: "☀️" },
                      { value: "dark", label: "Dark", icon: "🌙" },
                      { value: "night", label: "Night Light", icon: "🌇" },
                    ].map((option) => (
                      <button
                        key={option.value}
                        type="button"
                        className={`theme-option-button ${theme === option.value ? "active" : ""}`}
                        onClick={() => handleThemeSelect(option.value)}
                        aria-pressed={theme === option.value}
                      >
                        <span className="theme-option-icon" aria-hidden="true">{option.icon}</span>
                        <span>{option.label}</span>
                      </button>
                    ))}
                  </div>
                </div>
                <Link to="/voice-assistant" onClick={() => setShowMoreMenu(false)} role="menuitem"><FaMicrophone /> Voice Assistant</Link>
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
                <Link to="/profile-settings" onClick={() => setShowMoreMenu(false)} role="menuitem"><FaCog /> Profile settings</Link>
                <Link to="/community" onClick={() => setShowMoreMenu(false)} role="menuitem"><FaComments /> Community</Link>
                <Link to="/leaderboard" onClick={() => setShowMoreMenu(false)} role="menuitem"><FaTrophy />Leaderboard</Link>
                <Link to="/referrals" onClick={() => setShowMoreMenu(false)} role="menuitem"><FaUserPlus /> Referrals</Link>
                <Link to="/risk-index" onClick={() => setShowMoreMenu(false)} role="menuitem"><FaShieldAlt /> Risk Index</Link>
                <Link to="/farm-finance" onClick={() => setShowMoreMenu(false)} role="menuitem"><FaFileInvoiceDollar /> Farm Finance</Link>
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
                if (auth.currentUser) {
                  auth.currentUser.reload().then(() => {
                    const refreshedUser = auth.currentUser;
                    setUser({
                      uid: refreshedUser.uid,
                      email: refreshedUser.email,
                      emailVerified: refreshedUser.emailVerified,
                      isAnonymous: refreshedUser.isAnonymous,
                    });
                  }).catch((err) => {
                    console.error("Error reloading user:", err);
                  });
                }
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
        <React.Suspense fallback={<Loader fullPage={true} message={<span className="notranslate">Loading route...</span>} />}>
          <Routes>
            <Route path="/" element={<Home user={user} />} />
            <Route path="/advisor" element={<Advisor userData={userData} />} />
            <Route path="/how-it-works" element={<How />} />
            <Route path="/dashboard" element={<Dashboard userData={userData} />} />
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
            <Route path="/leaderboard" element={<Leaderboard />} />
            <Route path="/referrals" element={<ReferralHub />} />
            <Route path="/soil-analysis" element={<SoilAnalysis />} />
            <Route path="/faq" element={<FAQ />} />
            <Route path="/terms" element={<Terms />} />
            <Route path="/privacy-policy" element={<PrivacyPolicy />} />
            <Route path="/contributors" element={<Contributors />} />
            <Route path="/trace/:id" element={<QRTraceability />} />
            <Route path="/contact" element={<ContactUs />} />
            <Route path="/profile-settings" element={<ProfileSettings user={user} userData={userData} />} />
            <Route path="/about" element={<AboutUs />} />
            <Route path="/crop-planner" element={<SeasonalCropPlanner />} />
            <Route path="/soil-guide" element={<SoilGuide />} />
            <Route path="/disease-awareness" element={<CropDiseaseAwareness />} />
            <Route path="/pest-detection" element={<PestDetection />} />
            <Route path="/equipment-management" element={<EquipmentManagement />} />
            <Route path="/helpline" element={<Helpline />} />
            <Route path="/glossary" element={<Glossary />} />
            <Route path="/risk-index" element={<RiskIndex />} />
            <Route path="/crop-rotation" element={<CropRotation />} />
            <Route path="/seed-verifier" element={<SeedVerifier />} />
            <Route path="/farm-finance" element={<FarmFinance />} />
            <Route path="/farming-news" element={<FarmingNews userData={userData} />} />
            <Route path="/yield-predictor" element={<YieldPredictor />} />
            <Route path="/smart-farm-autopilot" element={<SmartFarmAutopilot />} />
            <Route
              path="/sustainability-analytics"
              element={<SustainabilityAnalyticsPage userData={userData} />}
            />
            <Route path="/blog" element={<Blog />} />
            <Route path="/blog/:id" element={<BlogDetail />} />
            <Route path="/weather" element={<Weather />} />
            <Route path="/voice-assistant" element={<VoiceAssistant />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </React.Suspense>
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
