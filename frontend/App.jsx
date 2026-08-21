import React, { Suspense, useEffect, useState, useRef } from "react";
import { Routes, Route, Link, NavLink, Navigate, useLocation, useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import SprayScheduler from "./SprayScheduler";
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
import usePriceAlerts from "./hooks/usePriceAlerts";
import Footer from "./components/Footer";
import { SkipLink } from "./NavigationManager";
import { useTheme } from "./ThemeContext";
import { SyncBadge } from "./src/components/SyncBadge";
import FarmingMythChecker from "./components/FarmingMythChecker";
import CropComparison from "./components/CropComparison";

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
  PestCalendar,
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
  PredictionExplainer,
  RetrainingPipelineMonitor,
  CropInsuranceClaim
} from "./routes/lazyPages";

const Weather = React.lazy(() => import("./Weather"));
const FeatureDriftMonitor = React.lazy(() => import("./FeatureDriftMonitor"));
import VoiceAssistant from "./VoiceAssistant";
import VoiceMicWidget from "./VoiceMicWidget";

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
import { clearOfflineData, clearOfflineRequests } from "./lib/db";
import { loadAppState, loadUserProfileSnapshot, persistAppState, persistUserProfileSnapshot } from "./lib/offlinePersistence";
import { syncOfflineRequests } from "./lib/syncOfflineRequests";

// CSS
import "./App.css";
import { LANGUAGE_OPTIONS } from "./lib/languageOptions";
import {
  synchronizeTranslation,
  allowUserInterfaceTranslation,
  refreshGoogleTranslation,
  cleanupGoogleTranslate,
} from "./lib/googleTranslate";

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

  const hydrationInProgressRef = useRef(false);
  const offlineSyncInProgressRef = useRef(false);
  const lastPersistedLangRef = useRef(null);
  const restoredSnapshotRef = useRef(false);
  const getStoredLanguagePreference = () => {
    try {
      return localStorage.getItem("agri:preferredLanguage") ||
        sessionStorage.getItem("agri:preferredLanguage");
    } catch {
      return null;
    }
  };

  const { i18n } = useTranslation();
  const { t } = useTranslation();
  const [preferredLang, setPreferredLang] = useState(() => {
    return getStoredLanguagePreference() || getInitialLanguage();
  });
  useEffect(() => {
    if (preferredLang && i18n.language !== preferredLang) {
      i18n.changeLanguage(preferredLang);
    }
  }, [preferredLang, i18n]);
  
  const [isOpen, setIsOpen] = useState(false);
  const { theme, toggleTheme, setTheme } = useTheme();
  const [user, setUser] = useState(null);
  const [userData, setUserData] = useState(null);
  const [profileCompleted, setProfileCompleted] = useState(true);
  const [loading, setLoading] = useState(false);
  const [showScorecard, setShowScorecard] = useState(false);
  const [showMoreMenu, setShowMoreMenu] = useState(false);
  const [isOffline, setIsOffline] = useState(!navigator.onLine);
  const [showScrollTop, setShowScrollTop] = useState(false);
  const [scrollProgress, setScrollProgress] = useState(0);

  const { liteMode, setLiteMode, detectAndSetLiteMode } =
    usePerformanceStore();

  // Price alert WebSocket status for global connection indicator
  const { status: priceAlertStatus } = usePriceAlerts();

  useEffect(() => {
    detectAndSetLiteMode();
  }, []);

  useEffect(() => {
    let cancelled = false;

    const hydrateOfflineState = async () => {
      if (hydrationInProgressRef.current) return;

      hydrationInProgressRef.current = true;

      try {
        const storedState = await loadAppState();

        if (
          !cancelled &&
          storedState &&
          typeof storedState === "object"
        ) {
          if (
            typeof storedState.preferredLang === "string" &&
            storedState.preferredLang.trim()
          ) {
            setPreferredLang(storedState.preferredLang);
          }
        }
      } catch (error) {
        console.warn(
          "Failed to restore offline app state:",
          error
        );
      } finally {
        hydrationInProgressRef.current = false;
      }
    };

    const syncQueuedRequests = async () => {
      if (offlineSyncInProgressRef.current) return;

      offlineSyncInProgressRef.current = true;

      try {
        await syncOfflineRequests();
      } catch (error) {
        console.warn(
          "Offline request sync failed:",
          error
        );
      } finally {
        offlineSyncInProgressRef.current = false;
      }
    };

    void hydrateOfflineState();
    void syncQueuedRequests();

    const handleOnline = () => {
      if (cancelled) return;
      setIsOffline(false);
      void syncQueuedRequests();
    };

    const handleOffline = () => {
      if (cancelled) return;
      setIsOffline(true);
    };

    window.addEventListener("online", handleOnline);
    window.addEventListener("offline", handleOffline);

    return () => {
      cancelled = true;

      window.removeEventListener(
        "online",
        handleOnline
      );

      window.removeEventListener(
        "offline",
        handleOffline
      );
    };
  }, []);

  useEffect(() => {
    if (
      !preferredLang ||
      lastPersistedLangRef.current === preferredLang
    ) {
      return;
    }

    lastPersistedLangRef.current = preferredLang;

    void persistAppState({
      preferredLang,
      persistedAt: Date.now(),
    });
  }, [preferredLang]);

  const location = useLocation();

  useNotifications();

  useBrowserCacheBudget({
    enabled: true,
    usageRatioLimit: liteMode ? 0.72 : 0.85,
  });

  /* ---------------- THEME SYSTEM (Moved to ThemeProvider) ---------------- */

/* ---------------- LANGUAGE AUTO-TRANS ---------------- */

useEffect(() => {
  let cancelled = false;

  if (!preferredLang) return;

  // Some older route components still carry the Google Translate marker on
  // visible UI copy. Keep explicit brand markers protected (translate="no"),
  // but allow all other component text to participate in the selected language.
  allowUserInterfaceTranslation();

  void synchronizeTranslation(preferredLang);

  const uiTranslationObserver = new MutationObserver((mutations) => {
    allowUserInterfaceTranslation();
    if (mutations.some(({ addedNodes }) => addedNodes.length > 0)) {
      refreshGoogleTranslation(preferredLang);
    }
  });
  uiTranslationObserver.observe(document.body, { childList: true, subtree: true });

  const handleWidgetLoad = () => {
    if (!cancelled) void synchronizeTranslation(preferredLang);
  };

  document.addEventListener("googleTranslateWidgetLoaded", handleWidgetLoad);

  return () => {
    cancelled = true;
    uiTranslationObserver.disconnect();
    document.removeEventListener("googleTranslateWidgetLoaded", handleWidgetLoad);
    cleanupGoogleTranslate();
  };
}, [preferredLang]);


  useEffect(() => {
    const hideGoogleTranslateBanner = () => {
      const bannerFrame = document.querySelector(
        ".goog-te-banner-frame"
      );

      if (bannerFrame) {
        bannerFrame.style.display = "none";
      }

      document.body.style.top = "0px";

      const translateElement = document.querySelector(
        ".goog-te-balloon-frame"
      );

      if (translateElement) {
        translateElement.style.display = "none";
      }
    };

    hideGoogleTranslateBanner();

    const interval = setInterval(
      hideGoogleTranslateBanner,
      1000
    );

    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (!isFirebaseConfigured()) {
      const timeout = setTimeout(
        () => setLoading(false),
        3000
      );

      setLoading(false);

      return () => clearTimeout(timeout);
    }

    const userDocUnsubscribeRef = {
      current: null,
    };

    const unsubscribeAuth = onAuthStateChanged(
      auth,
      (currentUser) => {
        setUser(currentUser);

        const hydrateUserSnapshot = async () => {
          if (
            !currentUser?.uid ||
            restoredSnapshotRef.current
          ) {
            return false;
          }

          restoredSnapshotRef.current = true;

          try {
            const snapshot =
              await loadUserProfileSnapshot(
                currentUser.uid
              );

            if (
              snapshot &&
              typeof snapshot === "object"
            ) {
              const normalizedSnapshot =
                normalizeUserProfile(snapshot);

              setUserData(normalizedSnapshot);

              setProfileCompleted(
                normalizedSnapshot.profileCompleted === true
              );

              return true;
            }
          } catch (error) {
            console.warn(
              "Failed to restore offline user profile snapshot:",
              error
            );
          }

          return false;
        };

        if (currentUser) {
          userDocUnsubscribeRef.current = onSnapshot(
            doc(db, "users", currentUser.uid),
            (userDoc) => {
              if (userDoc.exists()) {
                const data = normalizeUserProfile(
                  userDoc.data()
                );

                setUserData(data);

                setProfileCompleted(
                  data.profileCompleted === true
                );

                restoredSnapshotRef.current = false;
              } else if (currentUser.isAnonymous) {
                setUserData({
                  displayName: "Guest Farmer",
                  isAnonymous: true,
                });

                setProfileCompleted(true);
              } else {
                setUserData(null);
                setProfileCompleted(false);

                void hydrateUserSnapshot().finally(() =>
                  setLoading(false)
                );

                return;
              }

              setLoading(false);
            },
            (error) => {
              console.error(
                "Firestore sync error:",
                error
              );

              setUserData(null);
              setProfileCompleted(false);

              void hydrateUserSnapshot().finally(() =>
                setLoading(false)
              );
            }
          );
        } else {
          restoredSnapshotRef.current = false;
          setUserData(null);
          setProfileCompleted(true);
          setLoading(false);
        }
      }
    );

    return () => {
      unsubscribeAuth();

      if (userDocUnsubscribeRef.current) {
        userDocUnsubscribeRef.current();
      }
    };
  }, []);

  useEffect(() => {
    if (!user || !isFirebaseConfigured()) return;

    const ensurePublicKey = async () => {
      try {
        let { publicJwk } =
          await cryptoService.ensureKeys(user.uid);

        if (!publicJwk) {
          const publicKeySnap = await getDoc(
            doc(db, "public_keys", user.uid)
          );

          if (publicKeySnap.exists()) {
            publicJwk = publicKeySnap.data().jwk;

            await cryptoService.savePublicKey(
              user.uid,
              publicJwk
            );
          }
        }

        if (!publicJwk) {
          throw new Error(
            "ECDH public key unavailable after initialization"
          );
        }

        const pubKeyRef = doc(
          db,
          "public_keys",
          user.uid
        );

        await setDoc(
          pubKeyRef,
          { jwk: publicJwk },
          { merge: true }
        );
      } catch (error) {
        console.error(
          "Failed to generate/publish ECDH keys globally:",
          error
        );
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

  // Scroll to Top logic - removed duplicate

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (
        scorecardRef.current &&
        !scorecardRef.current.contains(
          event.target
        )
      ) {
        setShowScorecard(false);
      }
    };

    document.addEventListener(
      "mousedown",
      handleClickOutside
    );

    return () =>
      document.removeEventListener(
        "mousedown",
        handleClickOutside
      );
  }, []);

  const handleThemeToggle = toggleTheme;
  const handleThemeSelect = (nextTheme) => {
    setTheme(nextTheme);
    setShowMoreMenu(false);
  };
  const handleLogout = async () => {
    try {
      await signOut(auth);
      await Promise.allSettled([
        clearOfflineData(),
        clearOfflineRequests(),
        new Promise((resolve, reject) => {
          const req = indexedDB.deleteDatabase("fasal_e2ee");
          req.onsuccess = resolve;
          req.onerror = () => reject(req.error);
          req.onblocked = resolve;
        }),
      ]);
      window.location.href = "/";
    } catch (error) {
      console.error("Sign out error:", error);
    }
  };
  const scrollToTop = () => window.scrollTo({ top: 0, behavior: "smooth" });

  const [backendStatus, setBackendStatus] = useState("checking");

useEffect(() => {
  fetch(`${import.meta.env.VITE_API_BASE_URL}/health`)
    .then(res => {
      if (res.ok) return res.json();
      throw new Error("Backend not healthy");
    })
    .then(() => setBackendStatus("online"))
    .catch(() => setBackendStatus("offline"));
}, []);

  return (
    <div className={`app ${theme !== "light" ? "theme-dark" : ""} ${theme === "night" ? "theme-night" : ""} ${liteMode ? "lite-mode" : ""}`}>
      {user?.isAnonymous && <GuestBanner />}

      {loading && <Loader fullPage={true} message={<span className="notranslate" translate="no">Initializing Fasal Saathi...</span>} />}

      {isOffline && (
        <div className="offline-banner" role="alert">
          {t("alerts.offline")}
        </div>
      )}

      {/* Scroll Progress Bar */}
      <div className="scroll-progress-bar" style={{ width: `${scrollProgress}%` }} aria-hidden="true" />

      <nav className={`navbar ${isOpen ? "menu-open" : ""}`} role="navigation" aria-label="Main Navigation">
        <div className="nav-left">
          <Link to="/" className="brand" translate="no">Fasal Saathi</Link>
        </div>

        <ul className={`nav-center ${isOpen ? "active" : ""}`}>
          <li><NavLink to="/" onClick={() => setIsOpen(false)}>{t("nav.home")}</NavLink></li>
          <li><NavLink to="/about" onClick={() => setIsOpen(false)}>{t("nav.about")}</NavLink></li>
          <li><NavLink to="/how-it-works" onClick={() => setIsOpen(false)}>{t("nav.howItWorks")}</NavLink></li>
          <li><NavLink to="/crop-guide" onClick={() => setIsOpen(false)}> {t("nav.cropGuide")}</NavLink></li>
          <li><NavLink to="/resources" onClick={() => setIsOpen(false)}>{t("nav.resources")}</NavLink></li>
        </ul>

        <div className="nav-right">
          <button onClick={handleThemeToggle} className="theme-toggle" aria-label="Cycle Theme" title={`Current theme: ${theme}`}>
            {theme === "light" ? "🌙" : theme === "dark" ? "☀️" : "🌙"}
          </button>

          <SyncBadge />

          <button
            onClick={(e) => { e.stopPropagation(); setShowMoreMenu(!showMoreMenu); }}
            className={`more-menu-toggle ${showMoreMenu ? 'active' : ''}`}
            aria-label="More Options"
          >
            <span className="notranslate">{t("menu.more")}</span>
            <FaChevronDown className="chevron" />
          </button>

          {showMoreMenu && (
            <div className="more-dropdown" onClick={(e) => e.stopPropagation()} role="menu">
              <div className="dropdown-links">
                <div className="language-selector-section">
                  <label className="language-label">{t("nav.language") || "Language"}:</label>
                  <LanguageDropdown
                    options={LANGUAGE_OPTIONS}
                    value={preferredLang}
                    onChange={(lang) => {
                      setPreferredLang(lang);
                      try {
                        localStorage.setItem("agri:preferredLanguage", lang);
                        sessionStorage.setItem("agri:preferredLanguage", lang);
                      } catch {
                        console.warn("Unable to persist language preference");
                      }
                      void persistAppState({ preferredLang: lang });
                      void i18n.changeLanguage(lang);
                      window.location.reload();
                    }}
                  />
                </div>
                <div className="theme-selector-section">
                  <span className="theme-selector-label">{t("nav.theme") || "Theme"}:</span>
                  <div className="theme-option-grid" role="group" aria-label="Theme selection">
                      {[
                        { value: "light", label: t("theme.light"), icon: "☀️" },
                        { value: "dark", label: t("theme.dark"), icon: "🌙" },
                        { value: "night", label: t("theme.night"), icon: "🌇" },
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
                <Link to="/voice-assistant" onClick={() => setShowMoreMenu(false)} role="menuitem"><FaMicrophone /> {t("menu.voiceAssistant")}</Link>
                <Link to="/myth-checker" onClick={() => setShowMoreMenu(false)} role="menuitem"><FaMedal /> {t("menu.mythChecker")}</Link>
                <Link to="/crop-comparison" onClick={() => setShowMoreMenu(false)} role="menuitem"><FaLeaf /> {t("menu.cropComparison")}</Link>
                <div className="performance-toggle-section">
                  <button
                    className={`lite-mode-toggle ${liteMode ? 'active' : ''}`}
                    onClick={() => setLiteMode(!liteMode)}
                    role="menuitem"
                  >
                    <div className="toggle-info">
                      <FaBolt className="zap-icon" />
                      <span>{t("menu.liteMode")} {liteMode ? t("menu.liteModeOn") : t("menu.liteModeOff")}</span>
                    </div>
                    <div className="toggle-switch">
                      <div className="switch-handle" />
                    </div>
                  </button>
                </div>
                <Link to="/dashboard" onClick={() => setShowMoreMenu(false)} role="menuitem"><FaTachometerAlt /> {t("nav.dashboard")}</Link>
                {userData?.role === "admin" && (
                  <Link to="/admin/feedback" onClick={() => setShowMoreMenu(false)} role="menuitem"><FaShieldAlt /> {t("menu.feedbackAdmin")}</Link>
                )}
                <Link to="/profile-settings" onClick={() => setShowMoreMenu(false)} role="menuitem"><FaCog /> {t("menu.profileSettings")}</Link>
                <Link to="/community" onClick={() => setShowMoreMenu(false)} role="menuitem"><FaComments /> {t("menu.community")}</Link>
                <Link to="/leaderboard" onClick={() => setShowMoreMenu(false)} role="menuitem"><FaTrophy />{t("menu.leaderboard")}</Link>
                <Link to="/referrals" onClick={() => setShowMoreMenu(false)} role="menuitem"><FaUserPlus /> {t("menu.referrals")}</Link>
                <Link to="/risk-index" onClick={() => setShowMoreMenu(false)} role="menuitem"><FaShieldAlt /> {t("menu.riskIndex")}</Link>
                <Link to="/farm-finance" onClick={() => setShowMoreMenu(false)} role="menuitem"><FaFileInvoiceDollar /> {t("menu.farmFinance")}</Link>
                <Link to="/glossary" onClick={() => setShowMoreMenu(false)} role="menuitem"><FaBook /> {t("menu.glossary")}</Link>
                <Link to="/feature-drift" onClick={() => setShowMoreMenu(false)} role="menuitem"><FaInfoCircle /> {t("menu.featureDrift")}</Link>
                <Link to="/contact" onClick={() => setShowMoreMenu(false)} role="menuitem"><FaInfoCircle /> {t("menu.contact")}</Link>
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
                        { label: t("scorecard.primaryCrop"), value: userData.cropType || t("scorecard.na") },
                        { label: t("scorecard.language"), value: LANGUAGE_OPTIONS.find(l => l.value === (userData.language || preferredLang))?.label || preferredLang },
                        { label: t("scorecard.location"), value: userData.address || t("scorecard.fetching") }
                      ].map((item, i) => (
                        <div key={i} className="score-item">
                          <label>{item.label}</label>
                          <span>{item.value}</span>
                        </div>
                      ))}
                    </div>
                    <div className="scorecard-footer">
                       <button onClick={handleLogout} className="btn-logout-alt">{t("scorecard.signOut")}</button>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <Link to="/login" className="btn-get-started">{t("nav.getStarted")}</Link>
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
            <h2>{t("alerts.verifyEmail")}</h2>
            <p>{t("alerts.verifyEmailText")} <b>{user.email}</b>.<br /> {t("alerts.verifyEmailButton")}</p>
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
        <React.Suspense fallback={<Loader fullPage={true} message={<span className="notranslate" translate="no">Loading route...</span>} />}>
          <Routes>
            <Route path="/" element={<Home user={user} />} />
            <Route path="/advisor" element={<Advisor userData={userData} />} />
            <Route path="/how-it-works" element={<How />} />
            <Route path="/dashboard" element={<Dashboard userData={userData} wsStatus={priceAlertStatus} />} />
            <Route path="/crop-guide" element={<CropGuide />} />
            <Route path="/schemes" element={<Schemes />} />
            <Route path="/resources" element={<Resources />} />
            <Route path="/login" element={<Auth />} />
            <Route path="/auth" element={<Navigate to="/login" replace />} />
            <Route path="/profile-setup" element={<ProfileSetup user={user} profileCompleted={profileCompleted} />} />
            <Route path="/calendar" element={<Calendar userData={userData} />} />
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
            <Route path="/seasonal-pest-calendar" element={<PestCalendar />} />
            <Route path="/pest-detection" element={<PestDetection />} />
            <Route path="/equipment-management" element={<EquipmentManagement />} />
            <Route path="/helpline" element={<Helpline />} />
            <Route path="/glossary" element={<Glossary />} />
            <Route path="/risk-index" element={<RiskIndex />} />
            <Route path="/crop-rotation" element={<CropRotation />} />
            <Route path="/seed-verifier" element={<SeedVerifier />} />
            <Route path="/farm-finance" element={<FarmFinance />} />
            <Route path="/feature-drift" element={<FeatureDriftMonitor />} />
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
            <Route
  path="/spray-scheduler"
  element={
    <SprayScheduler
      schedules={[
        { crop: "Wheat", pest: "Rust", product: "Fungicide A", date: "2026-06-10", status: "upcoming" },
        { crop: "Rice", pest: "Blast", product: "Fungicide B", date: "2026-06-07", status: "today" },
        { crop: "Maize", pest: "Stem Borer", product: "Insecticide C", date: "2026-06-05", status: "overdue" },
      ]}
    />
  }
/>

            <Route path="/prediction-explainer" element={<PredictionExplainer />} />
            <Route path="/retraining-monitor" element={<RetrainingPipelineMonitor />} />
            <Route path="/insurance-claim" element={<CropInsuranceClaim />} />
            <Route
              path="/myth-checker"
              element={
                <div className="app-content">
                  <FarmingMythChecker />
                </div>
              }
            />
            <Route path="/crop-comparison" element={<CropComparison />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </React.Suspense>
      </main>

      {/* Floating Buttons */}
      <VoiceMicWidget />
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

              {backendStatus === "offline" && (
        <div className="backend-banner" role="alert">
          🚨 {t("alerts.backendUnavailable")}
        </div>
      )}

      <ToastContainer position="bottom-right" />
      <Footer />
    </div>
    
  );
}

export default App;

