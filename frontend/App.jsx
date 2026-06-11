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

// ============================================
// Google Translate Synchronization Utilities
// ============================================

const GOOGLE_TRANSLATE_TIMEOUT = 15000;
const GOOGLE_TRANSLATE_SYNC_DELAY = 1200;

let googleTranslateObserver = null;
let googleTranslateRetryTimeout = null;
let lastAppliedLanguage = null;
let translateInitializationInProgress = false;

/**
 * Apply translation only when necessary
 */
const applyGoogleTranslate = (langCode) => {
  try {
    const select = document.querySelector(".goog-te-combo");

    if (!select) {
      return false;
    }

    // Prevent redundant re-application
    if (select.value === langCode && lastAppliedLanguage === langCode) {
      return true;
    }

    select.value = langCode;

    select.dispatchEvent(
      new Event("change", { bubbles: true })
    );

    lastAppliedLanguage = langCode;

    return true;
  } catch (error) {
    console.error(
      "Google Translate apply error:",
      error
    );

    return false;
  }
};

/**
 * Wait for Google Translate widget with MutationObserver
 */
const waitForGoogleTranslateWidget = (
  timeoutMs = GOOGLE_TRANSLATE_TIMEOUT
) => {
  return new Promise((resolve, reject) => {
    const existingWidget = document.querySelector(
      ".goog-te-combo"
    );

    if (existingWidget) {
      resolve(existingWidget);
      return;
    }

    const timeoutId = setTimeout(() => {
      cleanup();
      reject(
        new Error(
          "Google Translate widget initialization timeout"
        )
      );
    }, timeoutMs);

    const cleanup = () => {
      clearTimeout(timeoutId);

      if (googleTranslateObserver) {
        googleTranslateObserver.disconnect();
        googleTranslateObserver = null;
      }
    };

    googleTranslateObserver = new MutationObserver(() => {
      const widget = document.querySelector(
        ".goog-te-combo"
      );

      if (widget) {
        cleanup();
        resolve(widget);
      }
    });

    googleTranslateObserver.observe(document.body, {
      childList: true,
      subtree: true,
    });
  });
};

/**
 * Robust translation synchronization
 */
const applyGoogleTranslateRobust = async (
  langCode,
  options = {}
) => {
  const {
    retry = true,
    onReady,
    onError,
  } = options;

  // Prevent overlapping initialization calls
  if (translateInitializationInProgress) {
    return;
  }

  translateInitializationInProgress = true;

  try {
    await waitForGoogleTranslateWidget();

    const applied = applyGoogleTranslate(langCode);

    if (!applied) {
      throw new Error(
        "Failed to apply translation state"
      );
    }

    onReady?.();
  } catch (error) {
    console.warn(
      "Google Translate synchronization failed:",
      error.message
    );

    // Retry once after delayed script injection
    if (retry) {
      clearTimeout(googleTranslateRetryTimeout);

      googleTranslateRetryTimeout = setTimeout(() => {
        void applyGoogleTranslateRobust(langCode, {
          retry: false,
        });
      }, GOOGLE_TRANSLATE_SYNC_DELAY);
    }

    onError?.(error);
  } finally {
    translateInitializationInProgress = false;
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
  const scrollFrameRef = useRef(null);

  const lastScrollStateRef = useRef({
    showScrollTop: false,
    scrollProgress: 0,
  });
  const hydrationInProgressRef = useRef(false);
  const offlineSyncInProgressRef = useRef(false);
  const lastPersistedLangRef = useRef(null);
  const restoredSnapshotRef = useRef(false);
  const getStoredLanguagePreference = () => {
    try {
      return sessionStorage.getItem("agri:preferredLanguage");
    } catch {
      return null;
    }
  };

  const { i18n } = useTranslation();
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
  }, [detectAndSetLiteMode]);

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
      setIsOffline(false);
      void syncQueuedRequests();
    };

    const handleOffline = () => {
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

  const synchronizeTranslation = async () => {
    if (!preferredLang || cancelled) return;

    // Skip redundant sync
    if (lastAppliedLanguage === preferredLang) {
      return;
    }

    // Fast path
    if (applyGoogleTranslate(preferredLang)) {
      return;
    }

    // Robust fallback path
    await applyGoogleTranslateRobust(
      preferredLang,
      {
        onReady: () => {
          console.log(
            "Google Translate synchronized successfully"
          );
        },

        onError: () => {
          console.warn(
            "Translation fallback active"
          );
        },
      }
    );
  };

  void synchronizeTranslation();

  const handleWidgetLoad = () => {
    if (cancelled) return;

    if (!applyGoogleTranslate(preferredLang)) {
      void applyGoogleTranslateRobust(
        preferredLang,
        { retry: false }
      );
    }
  };

  document.addEventListener(
    "googleTranslateWidgetLoaded",
    handleWidgetLoad
  );

  return () => {
    cancelled = true;

    document.removeEventListener(
      "googleTranslateWidgetLoaded",
      handleWidgetLoad
    );

    if (googleTranslateObserver) {
      googleTranslateObserver.disconnect();
      googleTranslateObserver = null;
    }

    if (googleTranslateRetryTimeout) {
      clearTimeout(
        googleTranslateRetryTimeout
      );

      googleTranslateRetryTimeout = null;
    }
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

  useEffect(() => {
    const handleScroll = () => {
      if (scrollFrameRef.current) return;

      scrollFrameRef.current =
        requestAnimationFrame(() => {
          const shouldShowScrollTop =
            window.scrollY > 300;

          const totalHeight =
            document.documentElement.scrollHeight -
            window.innerHeight;

          const progress =
            totalHeight > 0
              ? (window.scrollY / totalHeight) * 100
              : 0;

          if (
            lastScrollStateRef.current
              .showScrollTop !==
            shouldShowScrollTop
          ) {
            lastScrollStateRef.current.showScrollTop =
              shouldShowScrollTop;

            setShowScrollTop(shouldShowScrollTop);
          }

          if (
            Math.abs(
              lastScrollStateRef.current
                .scrollProgress - progress
            ) > 1
          ) {
            lastScrollStateRef.current.scrollProgress =
              progress;

            setScrollProgress(progress);
          }

          scrollFrameRef.current = null;
        });
    };

    window.addEventListener("scroll", handleScroll, {
      passive: true,
    });

    return () => {
      window.removeEventListener("scroll", handleScroll);

      if (scrollFrameRef.current) {
        cancelAnimationFrame(scrollFrameRef.current);
      }
    };
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
          <li><NavLink to="/" onClick={() => setIsOpen(false)}>Home</NavLink></li>
          <li><NavLink to="/about" onClick={() => setIsOpen(false)}>About</NavLink></li>
          <li><NavLink to="/how-it-works" onClick={() => setIsOpen(false)}>How It Works</NavLink></li>
          <li><NavLink to="/crop-guide" onClick={() => setIsOpen(false)}> Crop Guide</NavLink></li>
          <li><NavLink to="/resources" onClick={() => setIsOpen(false)}>Resources</NavLink></li>
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
                      try {
                        sessionStorage.setItem("agri:preferredLanguage", lang);
                      } catch (error) {
                        console.warn("Unable to persist language preference");
                      }
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
                <Link to="/myth-checker" onClick={() => setShowMoreMenu(false)} role="menuitem"><FaMedal /> Myth Checker</Link>
                <Link to="/crop-comparison" onClick={() => setShowMoreMenu(false)} role="menuitem"><FaLeaf /> Crop Comparison</Link>
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
                <Link to="/feature-drift" onClick={() => setShowMoreMenu(false)} role="menuitem"><FaInfoCircle /> Feature Drift Monitor</Link>
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
            <Route path="/dashboard" element={<Dashboard userData={userData} wsStatus={priceAlertStatus} />} />
            <Route path="/crop-guide" element={<CropGuide />} />
            <Route path="/schemes" element={<Schemes />} />
            <Route path="/resources" element={<Resources />} />
            <Route path="/login" element={<Auth />} />
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
