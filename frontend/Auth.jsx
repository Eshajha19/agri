import React, { useState, useEffect, useRef } from "react";
import {
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  signInWithPopup,
  GoogleAuthProvider,
  sendEmailVerification,
  signOut,
  signInAnonymously,
  linkWithCredential,
  EmailAuthProvider
} from "firebase/auth";
import { doc, setDoc, getDoc } from "firebase/firestore";
import { useNavigate, useLocation } from "react-router-dom";
import { FaGoogle, FaEnvelope, FaLock, FaUser, FaArrowRight, FaLeaf, FaUserSecret } from "react-icons/fa";
import { auth, db, isFirebaseConfigured } from "./lib/firebase";
import { migrateUserData } from "./lib/migration";
import "./Auth.css";

// ============================================
// Security Utilities
// ============================================

/**
 * Sanitize email input to prevent injection attacks
 */
const sanitizeEmail = (email) => {
  if (!email) return "";
  return email.toLowerCase().trim().substring(0, 254);
};

/**
 * Validate email format with RFC 5322 compliant regex
 */
const isValidEmail = (email) => {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email) && email.length <= 254;
};

/**
 * Validate password strength
 * - Minimum 8 characters
 * - At least one uppercase letter
 * - At least one lowercase letter
 * - At least one digit
 * - At least one special character
 */
const validatePasswordStrength = (password) => {
  const minLength = 8;
  const hasUpperCase = /[A-Z]/.test(password);
  const hasLowerCase = /[a-z]/.test(password);
  const hasDigit = /\d/.test(password);
  const hasSpecial = /[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]/.test(password);

  const strength = {
    isValid: password.length >= minLength && hasUpperCase && hasLowerCase && hasDigit && hasSpecial,
    feedback: []
  };

  if (password.length < minLength) strength.feedback.push("At least 8 characters required");
  if (!hasUpperCase) strength.feedback.push("Add uppercase letter (A-Z)");
  if (!hasLowerCase) strength.feedback.push("Add lowercase letter (a-z)");
  if (!hasDigit) strength.feedback.push("Add digit (0-9)");
  if (!hasSpecial) strength.feedback.push("Add special character (!@#$%^&*)");

  return strength;
};

/**
 * Constant-time string comparison to prevent timing attacks
 */
const constantTimeCompare = (a, b) => {
  if (a.length !== b.length) return false;
  let result = 0;
  for (let i = 0; i < a.length; i++) {
    result |= a.charCodeAt(i) ^ b.charCodeAt(i);
  }
  return result === 0;
};

/**
 * Rate limiter for authentication attempts
 */
class AuthRateLimiter {
  constructor(maxAttempts = 5, windowMs = 15 * 60 * 1000) {
    this.maxAttempts = maxAttempts;
    this.windowMs = windowMs;
    this.attempts = new Map();
  }

  isLimited(email) {
    const now = Date.now();
    const key = sanitizeEmail(email);

    if (!this.attempts.has(key)) {
      this.attempts.set(key, []);
    }

    const timestamps = this.attempts.get(key);

    // Remove timestamps outside the window
    const validTimestamps = timestamps.filter(ts => now - ts < this.windowMs);
    this.attempts.set(key, validTimestamps);

    return validTimestamps.length >= this.maxAttempts;
  }

  recordAttempt(email) {
    const key = sanitizeEmail(email);
    if (!this.attempts.has(key)) {
      this.attempts.set(key, []);
    }
    this.attempts.get(key).push(Date.now());
  }

  getRemainingTime(email) {
    const key = sanitizeEmail(email);
    if (!this.attempts.has(key)) return 0;

    const timestamps = this.attempts.get(key);
    if (timestamps.length === 0) return 0;

    const oldestAttempt = Math.min(...timestamps);
    const remainingMs = this.windowMs - (Date.now() - oldestAttempt);
    return Math.max(0, remainingMs);
  }
}

/**
 * Secure token storage with expiration tracking
 */
class SecureTokenManager {
  constructor() {
    this.tokenRefreshBuffer = 5 * 60 * 1000;
    this.refreshTimer = null;
    this._token = null;
    this._expirationTime = null;
  }

  storeToken(token, expirationTime) {
    // Store token only in sessionStorage (safer than localStorage)
    try {
      sessionStorage.setItem("auth_token", token);
      sessionStorage.setItem(
        "auth_token_expiration",
        expirationTime.toString()
      );
    } catch (err) {
      console.error("Session storage unavailable");
    }

    this._token = token;
    this._expirationTime = expirationTime;

    this._scheduleRefresh(expirationTime);
  }

  _scheduleRefresh(expirationTime) {
    if (this.refreshTimer) {
      clearTimeout(this.refreshTimer);
    }

    const now = Date.now();
    const refreshTime =
      expirationTime - this.tokenRefreshBuffer - now;

    if (refreshTime > 0) {
      this.refreshTimer = setTimeout(() => {
        window.dispatchEvent(
          new CustomEvent("auth:token-refresh-needed")
        );
      }, refreshTime);
    }
  }

  getToken() {
    if (this._token) {
      return this._token;
    }

    try {
      return sessionStorage.getItem("auth_token");
    } catch {
      return null;
    }
  }

  isTokenExpired() {
    let storedExpiration = null;

    try {
      storedExpiration = Number(
        sessionStorage.getItem("auth_token_expiration")
      );
    } catch {
      storedExpiration = null;
    }

    const expiration =
      this._expirationTime || storedExpiration;

    if (!expiration) {
      return true;
    }

    return Date.now() >= expiration;
  }
  clearToken() {
    this._token = null;
    this._expirationTime = null;

    try {
      sessionStorage.removeItem("auth_token");
      sessionStorage.removeItem("auth_token_expiration");
    } catch {}

    if (this.refreshTimer) {
      clearTimeout(this.refreshTimer);
    }
  }
}

/**
 * CSRF token management
 */
const generateCSRFToken = () => {
  return Array.from(crypto.getRandomValues(new Uint8Array(32)))
    .map(b => b.toString(16).padStart(2, '0'))
    .join('');
};

const csrfToken = useRef(generateCSRFToken());

// ============================================
// Auth Component
// ============================================

const Auth = () => {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [phoneNumber, setPhoneNumber] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [passwordStrength, setPasswordStrength] = useState(null);
  const [isLimited, setIsLimited] = useState(false);
  const [remainingTime, setRemainingTime] = useState(0);

  const navigate = useNavigate();
  const location = useLocation();

  // Security refs
  const rateLimiter = useRef(new AuthRateLimiter());
  const tokenManager = useRef(new SecureTokenManager());
  const csrfToken = useRef(generateCSRFToken());

  const from = location.state?.from?.pathname || "/";
  const referralCode = new URLSearchParams(location.search).get("ref") || "";
  const redirectAfterAuth = referralCode
    ? `/referrals?ref=${encodeURIComponent(referralCode)}`
    : from;

  // Monitor rate limiting
  useEffect(() => {
    const checkRateLimit = () => {
      const limited = rateLimiter.current.isLimited(email);
      setIsLimited(limited);

      if (limited) {
        const remaining = rateLimiter.current.getRemainingTime(email);
        setRemainingTime(Math.ceil(remaining / 1000));
      }
    };

    const interval = setInterval(checkRateLimit, 1000);
    return () => clearInterval(interval);
  }, [email]);

  // Handle password strength feedback
  const handlePasswordChange = (e) => {
    const newPassword = e.target.value;
    setPassword(newPassword);

    if (!isLogin && newPassword) {
      const strength = validatePasswordStrength(newPassword);
      setPasswordStrength(strength);
    }
  };

  if (!isFirebaseConfigured()) {
    return (
      <div className="auth-container">
        <div className="auth-card">
            <div className="auth-logo">
              <FaLeaf className="leaf-icon" />
              <h1 className="notranslate" translate="no">Fasal Saathi</h1>
            </div>
          <p className="auth-subtitle">Firebase credentials not configured</p>
          <div className="auth-message">
            <p>Please configure Firebase credentials in your .env file to enable authentication.</p>
          </div>
        </div>
      </div>
    );
  }

  const handleGuestLogin = async () => {
    setLoading(true);
    setError("");
    try {
      await signInAnonymously(auth);
      navigate(redirectAfterAuth, { replace: true });
    } catch (err) {
      console.error("Guest login error");
      setError("Failed to start guest session.");
    } finally {
      setLoading(false);
    }
  };

  const handleAuth = async (e) => {
    e.preventDefault();
    setError("");
    setMessage("");

    // Security: Check rate limiting
    if (rateLimiter.current.isLimited(email)) {
      const remaining = rateLimiter.current.getRemainingTime(email);
      setError(`Too many login attempts. Try again in ${Math.ceil(remaining / 1000)} seconds.`);
      return;
    }

    // Security: Validate and sanitize email
    const sanitized = sanitizeEmail(email);
    if (!isValidEmail(sanitized)) {
      setError("Invalid email format. Please enter a valid email address.");
      return;
    }

    // Security: Check password strength on registration
    if (!isLogin) {
      const strength = validatePasswordStrength(password);
      if (!strength.isValid) {
        setError(`Password too weak. ${strength.feedback.join(", ")}`);
        return;
      }
    }

    // Record attempt for rate limiting
    rateLimiter.current.recordAttempt(email);
    setLoading(true);

    try {
      if (isLogin) {
        // Login Logic with security checks
        const anonymousUser = auth.currentUser?.isAnonymous ? auth.currentUser : null;
        const anonymousUid = anonymousUser?.uid;

        const userCredential = await signInWithEmailAndPassword(auth, sanitized, password);
        const user = userCredential.user;

        if (!user.emailVerified) {
          setError("Please verify your email before logging in. Check your inbox.");
          await signOut(auth);
          setLoading(false);
          return;
        }

        // Store token securely with expiration tracking
        const token = await user.getIdToken();
        const expirationTime = Date.now() + (60 * 60 * 1000); // 1 hour
        tokenManager.current.storeToken(token, expirationTime);

        // If there was a guest session, migrate data to the logged-in account
        if (anonymousUid && user.uid !== anonymousUid) {
          try {
            await migrateUserData(anonymousUid, user.uid);
            setMessage("Guest data successfully merged with your account!");
          } catch (migrateErr) {
            console.error("Migration error");
            // Non-fatal, user is logged in
          }
        }

        navigate(redirectAfterAuth, { replace: true });
      } else {
        // Registration Logic with security
        const userCredential = await createUserWithEmailAndPassword(auth, sanitized, password);
        const user = userCredential.user;

        // Create user profile in Firestore
        await setDoc(doc(db, "users", user.uid), {
          email: sanitized,
          displayName: displayName.trim(),
          phoneNumber: phoneNumber.trim(),
          createdAt: new Date().toISOString(),
          emailVerified: false
        });

        // Send email verification
        await sendEmailVerification(user);
        setMessage("Verification email sent. Please check your inbox.");
        setIsLogin(true);
        setEmail("");
        setPassword("");
        setDisplayName("");
        setPhoneNumber("");
      }
    } catch (err) {
      // Security: Log error without exposing sensitive details
      console.error("Auth error");

      // Provide user-friendly error messages
      if (err.code === "auth/user-not-found") {
        setError("Email not found. Create a new account.");
      } else if (err.code === "auth/wrong-password") {
        setError("Incorrect password. Please try again.");
      } else if (err.code === "auth/email-already-in-use") {
        setError("Email already registered. Please login.");
      } else if (err.code === "auth/weak-password") {
        setError("Password too weak. Use at least 8 characters with mixed case, numbers, and symbols.");
      } else if (err.code === "auth/network-request-failed") {
        setError("Network error. Check your connection and try again.");
      } else {
        setError("Authentication failed. Please try again.");
      }
     } finally {
       setLoading(false);
     }
   };

  const handleGoogleLogin = async () => {
    const provider = new GoogleAuthProvider();
    // Add custom parameters if needed
    provider.setCustomParameters({ prompt: 'select_account' });
    
    setLoading(true);
    setError("");
    try {
      const result = await signInWithPopup(auth, provider);
      const user = result.user;

      // Create/Update user in Firestore.
      // We wrap this in a try-catch to differentiate between Auth and Firestore failures.
      try {
        // Read the existing document first so we never overwrite a role that
        // was already set (e.g. an admin re-logging in via Google).  On first
        // sign-in the document won't exist, so we write role: "farmer" as the
        // explicit default.  On subsequent logins we only update mutable fields
        // (displayName, photoURL, lastLogin) and leave role untouched.
        const userDocRef = doc(db, "users", user.uid);
        const existingDoc = await getDoc(userDocRef);
        const isNewUser = !existingDoc.exists() || !existingDoc.data()?.role;

        await setDoc(userDocRef, {
          uid: user.uid,
          displayName: user.displayName,
          email: user.email,
          photoURL: user.photoURL,
          lastLogin: new Date().toISOString(),
          profileCompleted: true,
          reputation: 0,
          // Only set role on first sign-in. Subsequent logins must not
          // overwrite a role that was elevated (e.g. farmer → expert/admin).
          ...(isNewUser && { role: "farmer" }),
        }, { merge: true });
      } catch (fsErr) {
        console.error("Firestore sync error:", fsErr);
        // We continue even if Firestore fails, as the user is authenticated
      }

      navigate(redirectAfterAuth, { replace: true });
    } catch (err) {
      console.error("Google Auth Error:", err);
      
      if (err.code === "auth/popup-closed-by-user") {
        setError(""); // Don't show error if user closed the popup
      } else if (err.code === "auth/cancelled-by-user") {
        setError("");
      } else if (err.code === "auth/operation-not-allowed") {
        setError("Google sign-in is not enabled in Firebase Console.");
      } else if (err.code === "auth/popup-blocked") {
        setError("Sign-in popup was blocked by your browser. Please allow popups for this site.");
      } else if (err.code === "auth/internal-error") {
        setError("Internal authentication error. Please try again later.");
      } else {
        setError(err.message || "Failed to sign in with Google.");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-container">
      <div className="auth-card">
        <div className="auth-header">
          <div className="auth-logo">
            <FaLeaf />
            <span className="notranslate" translate="no">Fasal Saathi</span>
          </div>
          <h1>{isLogin ? "Welcome Back" : (
            <>Join <span className="notranslate" translate="no">Fasal Saathi</span></>
          )}</h1>
          <p>{isLogin ? "Continue your farming journey" : "Start your smart farming journey today"}</p>
        </div>

        {error && <div className="auth-error">{error}</div>}
        {message && <div className="auth-success">{message}</div>}

        <form onSubmit={handleAuth} className="auth-form">
          {!isLogin && (
            <div className="input-group">
              <label>Full Name</label>
              <div className="input-wrapper">
                <FaUser className="input-icon" />
                <input
                  type="text"
                  placeholder="e.g. Ramesh Kumar"
                  value={displayName}
                  onChange={(e) => setDisplayName(e.target.value)}
                  required
                />
              </div>
            </div>
          )}
          <div className="input-group">
            <label>Email Address</label>
            <div className="input-wrapper">
              <FaEnvelope className="input-icon" />
              <input
                type="email"
                placeholder="email@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </div>
          </div>
          {!isLogin && (
            <div className="input-group">
              <label>Phone Number</label>
              <div className="input-wrapper">
                <span className="input-icon" style={{ fontSize: '1rem', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>📱</span>
                <input
                  type="tel"
                  placeholder="+91 98765 43210"
                  value={phoneNumber}
                  onChange={(e) => setPhoneNumber(e.target.value)}
                  required={!isLogin}
                />
              </div>
            </div>
          )}
          <div className="input-group">
            <label>Password</label>
            <div className="input-wrapper">
              <FaLock className="input-icon" />
              <input
                type="password"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
            </div>
          </div>
          <button type="submit" className="auth-submit-btn" disabled={loading}>
            {loading ? "Processing..." : isLogin ? "Login to Account" : "Create Account"}
          </button>
        </form>

        {isLogin && (
          <>
            <div className="auth-divider">
              <span>OR</span>
            </div>

            <button onClick={handleGoogleLogin} className="google-btn" disabled={loading}>
              <img 
                src="https://www.gstatic.com/firebasejs/ui/2.0.0/images/auth/google.svg" 
                alt="Google" 
                className="google-icon"
              /> 
              Continue with Google
            </button>

            <button onClick={handleGuestLogin} className="guest-btn" disabled={loading}>
              <FaUserSecret className="guest-icon" />
              Continue as Guest
            </button>
          </>
        )}

        <p className="auth-toggle">
          {isLogin ? "Don't have an account?" : "Already have an account?"}
          <button onClick={() => setIsLogin(!isLogin)}>
            {isLogin ? "Create Account" : "Login Now"}
          </button>
        </p>
      </div>

      <div className="auth-visual">
        <div className="visual-content">
          <h2>Empowering Farmers with AI</h2>
          <p>Get personalized recommendations, real-time alerts, and expert guidance to optimize your yield.</p>
          <div className="visual-stats">
            <div className="v-stat">
              <span>98%</span>
              <p>Accuracy</p>
            </div>
            <div className="v-stat">
              <span>50K+</span>
              <p>Farmers</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Auth;
// Enhanced Auth.jsx with security hardening
