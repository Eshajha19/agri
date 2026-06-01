// Enhanced Auth.jsx with authentication throttle lifecycle hardening

import React, { useState, useEffect, useRef, useMemo } from "react";
import {
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  signInWithPopup,
  GoogleAuthProvider,
  sendEmailVerification,
  signOut,
  signInAnonymously,
} from "firebase/auth";

import { doc, setDoc, getDoc } from "firebase/firestore";

import { useNavigate, useLocation } from "react-router-dom";

import {
  FaGoogle,
  FaEnvelope,
  FaLock,
  FaUser,
  FaLeaf,
  FaUserSecret,
} from "react-icons/fa";

import { auth, db, isFirebaseConfigured } from "./lib/firebase";
import { migrateUserData } from "./lib/migration";

import "./Auth.css";

/* ============================================
   Security Utilities
============================================ */

const sanitizeEmail = (email) => {
  if (typeof email !== "string") return "";

  return email
    .toLowerCase()
    .trim()
    .substring(0, 254);
};

const isValidEmail = (email) => {
  const emailRegex =
    /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

  return (
    emailRegex.test(email) &&
    email.length <= 254
  );
};

const validatePasswordStrength = (password) => {
  const minLength = 8;

  const hasUpperCase = /[A-Z]/.test(password);
  const hasLowerCase = /[a-z]/.test(password);
  const hasDigit = /\d/.test(password);
  const hasSpecial =
    /[!@#$%^&*()_+\-=[\]{};':"\\|,.<>/?]/.test(
      password
    );

  const strength = {
    isValid:
      password.length >= minLength &&
      hasUpperCase &&
      hasLowerCase &&
      hasDigit &&
      hasSpecial,
    feedback: [],
  };

  if (password.length < minLength) {
    strength.feedback.push(
      "At least 8 characters required"
    );
  }

  if (!hasUpperCase) {
    strength.feedback.push(
      "Add uppercase letter (A-Z)"
    );
  }

  if (!hasLowerCase) {
    strength.feedback.push(
      "Add lowercase letter (a-z)"
    );
  }

  if (!hasDigit) {
    strength.feedback.push(
      "Add digit (0-9)"
    );
  }

  if (!hasSpecial) {
    strength.feedback.push(
      "Add special character (!@#$%^&*)"
    );
  }

  return strength;
};

class AuthRateLimiter {
  constructor(
    maxAttempts = 5,
    windowMs = 15 * 60 * 1000
  ) {
    this.maxAttempts = maxAttempts;
    this.windowMs = windowMs;
    this.attempts = new Map();
    this.cleanupInterval = null;

    this.startCleanup();
  }

  normalizeKey(email) {
    const sanitized = sanitizeEmail(email);

    if (!sanitized || !isValidEmail(sanitized)) {
      return null;
    }

    return sanitized;
  }

  cleanupExpiredEntries() {
    const now = Date.now();

    for (const [key, timestamps] of this.attempts.entries()) {
      const validTimestamps = timestamps.filter(
        (ts) => now - ts < this.windowMs
      );

      if (validTimestamps.length === 0) {
        this.attempts.delete(key);
      } else {
        this.attempts.set(key, validTimestamps);
      }
    }
  }

  startCleanup() {
    this.cleanupInterval = setInterval(() => {
      this.cleanupExpiredEntries();
    }, 60 * 1000);
  }

  stopCleanup() {
    if (this.cleanupInterval) {
      clearInterval(this.cleanupInterval);
      this.cleanupInterval = null;
    }
  }

  isLimited(email) {
    const key = this.normalizeKey(email);

    if (!key) return false;

    this.cleanupExpiredEntries();

    const attempts =
      this.attempts.get(key) || [];

    return attempts.length >= this.maxAttempts;
  }

  recordAttempt(email) {
    const key = this.normalizeKey(email);

    if (!key) return;

    this.cleanupExpiredEntries();

    const attempts =
      this.attempts.get(key) || [];

    attempts.push(Date.now());

    this.attempts.set(key, attempts);
  }

  getRemainingTime(email) {
    const key = this.normalizeKey(email);

    if (!key) return 0;

    const timestamps =
      this.attempts.get(key) || [];

    if (timestamps.length === 0) {
      return 0;
    }

    const oldestAttempt =
      Math.min(...timestamps);

    const remainingMs =
      this.windowMs -
      (Date.now() - oldestAttempt);

    return Math.max(
      0,
      Math.ceil(remainingMs / 1000)
    );
  }
}

/* ============================================
   Auth Component
============================================ */

const Auth = () => {
  const [isLogin, setIsLogin] =
    useState(true);

  const [email, setEmail] =
    useState("");

  const [password, setPassword] =
    useState("");

  const [displayName, setDisplayName] =
    useState("");

  const [phoneNumber, setPhoneNumber] =
    useState("");

  const [error, setError] =
    useState("");

  const [loading, setLoading] =
    useState(false);

  const [message, setMessage] =
    useState("");

  const [passwordStrength, setPasswordStrength] =
    useState(null);

  const [isLimited, setIsLimited] =
    useState(false);

  const [remainingTime, setRemainingTime] =
    useState(0);

  const navigate = useNavigate();

  const location = useLocation();

  const rateLimiter = useRef(
    new AuthRateLimiter()
  );

  const countdownRef = useRef(null);

  const from =
    location.state?.from?.pathname || "/";

  const referralCode =
    new URLSearchParams(
      location.search
    ).get("ref") || "";

  const redirectAfterAuth = referralCode
    ? `/referrals?ref=${encodeURIComponent(
        referralCode
      )}`
    : from;

  const sanitizedEmail = useMemo(
    () => sanitizeEmail(email),
    [email]
  );

  useEffect(() => {
    const updateRateLimitState = () => {
      if (!sanitizedEmail) {
        setIsLimited(false);
        setRemainingTime(0);
        return;
      }

      const limited =
        rateLimiter.current.isLimited(
          sanitizedEmail
        );

      setIsLimited((prev) =>
        prev !== limited ? limited : prev
      );

      if (limited) {
        const remaining =
          rateLimiter.current.getRemainingTime(
            sanitizedEmail
          );

        setRemainingTime((prev) =>
          prev !== remaining
            ? remaining
            : prev
        );
      } else {
        setRemainingTime(0);
      }
    };

    updateRateLimitState();

    countdownRef.current = setInterval(
      updateRateLimitState,
      1000
    );

    return () => {
      if (countdownRef.current) {
        clearInterval(countdownRef.current);
      }
    };
  }, [sanitizedEmail]);

  useEffect(() => {
    return () => {
      rateLimiter.current.stopCleanup();

      if (countdownRef.current) {
        clearInterval(countdownRef.current);
      }
    };
  }, []);

  const handlePasswordChange = (e) => {
    const newPassword = e.target.value;

    setPassword(newPassword);

    if (!isLogin && newPassword) {
      const strength =
        validatePasswordStrength(
          newPassword
        );

      setPasswordStrength(strength);
    }
  };

  if (!isFirebaseConfigured()) {
    return (
      <div className="auth-container">
        <div className="auth-card">
          <div className="auth-logo">
            <FaLeaf className="leaf-icon" />
            <h1
              className="notranslate"
              translate="no"
            >
              Fasal Saathi
            </h1>
          </div>

          <p className="auth-subtitle">
            Firebase credentials not configured
          </p>

          <div className="auth-message">
            <p>
              Please configure Firebase
              credentials in your .env file.
            </p>
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

      navigate(redirectAfterAuth, {
        replace: true,
      });
    } catch (err) {
      console.error("Guest login error");

      setError(
        "Failed to start guest session."
      );
    } finally {
      setLoading(false);
    }
  };

  const handleAuth = async (e) => {
    e.preventDefault();

    setError("");
    setMessage("");

    if (
      rateLimiter.current.isLimited(
        sanitizedEmail
      )
    ) {
      const remaining =
        rateLimiter.current.getRemainingTime(
          sanitizedEmail
        );

      setError(
        `Too many login attempts. Try again in ${remaining} seconds.`
      );

      return;
    }

    if (
      !isValidEmail(sanitizedEmail)
    ) {
      setError(
        "Invalid email format."
      );

      return;
    }

    if (!isLogin) {
      const strength =
        validatePasswordStrength(
          password
        );

      if (!strength.isValid) {
        setError(
          `Password too weak. ${strength.feedback.join(
            ", "
          )}`
        );

        return;
      }
    }

    rateLimiter.current.recordAttempt(
      sanitizedEmail
    );

    setLoading(true);

    try {
      if (isLogin) {
        const anonymousUser =
          auth.currentUser?.isAnonymous
            ? auth.currentUser
            : null;

        const anonymousUid =
          anonymousUser?.uid;

        const userCredential =
          await signInWithEmailAndPassword(
            auth,
            sanitizedEmail,
            password
          );

        const user =
          userCredential.user;

        if (!user.emailVerified) {
          setError(
            "Please verify your email before logging in."
          );

          await signOut(auth);

          setLoading(false);

          return;
        }

        if (
          anonymousUid &&
          user.uid !== anonymousUid
        ) {
          try {
            await migrateUserData(
              anonymousUid,
              user.uid
            );

            setMessage(
              "Guest data successfully merged."
            );
          } catch (migrateErr) {
            console.error(
              "Migration error"
            );
          }
        }

        navigate(
          redirectAfterAuth,
          {
            replace: true,
          }
        );
      } else {
        const userCredential =
          await createUserWithEmailAndPassword(
            auth,
            sanitizedEmail,
            password
          );

        const user =
          userCredential.user;

        await setDoc(
          doc(db, "users", user.uid),
          {
            email: sanitizedEmail,
            displayName:
              displayName.trim(),
            phoneNumber:
              phoneNumber.trim(),
            createdAt:
              new Date().toISOString(),
            emailVerified: false,
          }
        );

        await sendEmailVerification(
          user
        );

        setMessage(
          "Verification email sent."
        );

        setIsLogin(true);

        setEmail("");
        setPassword("");
        setDisplayName("");
        setPhoneNumber("");
      }
    } catch (err) {
      console.error("Auth error");

      if (
        err.code ===
        "auth/user-not-found"
      ) {
        setError(
          "Email not found."
        );
      } else if (
        err.code ===
        "auth/wrong-password"
      ) {
        setError(
          "Incorrect password."
        );
      } else if (
        err.code ===
        "auth/email-already-in-use"
      ) {
        setError(
          "Email already registered."
        );
      } else if (
        err.code ===
        "auth/network-request-failed"
      ) {
        setError(
          "Network error. Try again."
        );
      } else {
        setError(
          "Authentication failed."
        );
      }
    } finally {
      setLoading(false);
    }
  };

  const handleGoogleLogin = async () => {
    const provider =
      new GoogleAuthProvider();

    provider.setCustomParameters({
      prompt: "select_account",
    });

    setLoading(true);
    setError("");

    try {
      const result =
        await signInWithPopup(
          auth,
          provider
        );

      const user = result.user;

      try {
        const userDocRef = doc(
          db,
          "users",
          user.uid
        );

        const existingDoc =
          await getDoc(userDocRef);

        const isNewUser =
          !existingDoc.exists() ||
          !existingDoc.data()?.role;

        await setDoc(
          userDocRef,
          {
            uid: user.uid,
            displayName:
              user.displayName,
            email: user.email,
            photoURL:
              user.photoURL,
            lastLogin:
              new Date().toISOString(),
            profileCompleted: true,
            reputation: 0,
            ...(isNewUser && {
              role: "farmer",
            }),
          },
          { merge: true }
        );
      } catch (fsErr) {
        console.error(
          "Firestore sync error:",
          fsErr
        );
      }

      navigate(
        redirectAfterAuth,
        {
          replace: true,
        }
      );
    } catch (err) {
      console.error(
        "Google Auth Error:",
        err
      );

      if (
        err.code ===
          "auth/popup-closed-by-user" ||
        err.code ===
          "auth/cancelled-by-user"
      ) {
        setError("");
      } else if (
        err.code ===
        "auth/popup-blocked"
      ) {
        setError(
          "Popup blocked by browser."
        );
      } else {
        setError(
          "Failed to sign in with Google."
        );
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
            <span
              className="notranslate"
              translate="no"
            >
              Fasal Saathi
            </span>
          </div>

          <h1>
            {isLogin
              ? "Welcome Back"
              : "Create Account"}
          </h1>

          <p>
            Secure authentication with
            lifecycle hardening
          </p>
        </div>

        {error && (
          <div className="auth-error">
            {error}
          </div>
        )}

        {message && (
          <div className="auth-success">
            {message}
          </div>
        )}

        <form
          onSubmit={handleAuth}
          className="auth-form"
        >
          {!isLogin && (
            <div className="input-group">
              <label>
                Full Name
              </label>

              <div className="input-wrapper">
                <FaUser className="input-icon" />

                <input
                  type="text"
                  value={displayName}
                  onChange={(e) =>
                    setDisplayName(
                      e.target.value
                    )
                  }
                  required
                />
              </div>
            </div>
          )}

          <div className="input-group">
            <label>
              Email Address
            </label>

            <div className="input-wrapper">
              <FaEnvelope className="input-icon" />

              <input
                type="email"
                value={email}
                onChange={(e) =>
                  setEmail(
                    e.target.value
                  )
                }
                required
              />
            </div>
          </div>

          <div className="input-group">
            <label>Password</label>

            <div className="input-wrapper">
              <FaLock className="input-icon" />

              <input
                type="password"
                value={password}
                onChange={
                  handlePasswordChange
                }
                required
              />
            </div>
          </div>

          {!isLogin &&
            passwordStrength && (
              <div className="password-feedback">
                {passwordStrength.feedback.map(
                  (item) => (
                    <div key={item}>
                      • {item}
                    </div>
                  )
                )}
              </div>
            )}

          {isLimited && (
            <div className="auth-error">
              Too many attempts.
              Retry in{" "}
              {remainingTime} seconds.
            </div>
          )}

          <button
            type="submit"
            className="auth-submit-btn"
            disabled={
              loading || isLimited
            }
          >
            {loading
              ? "Processing..."
              : isLogin
              ? "Login"
              : "Create Account"}
          </button>
        </form>

        {isLogin && (
          <>
            <div className="auth-divider">
              <span>OR</span>
            </div>

            <button
              onClick={
                handleGoogleLogin
              }
              className="google-btn"
              disabled={loading}
            >
              <FaGoogle />
              Continue with Google
            </button>

            <button
              onClick={
                handleGuestLogin
              }
              className="guest-btn"
              disabled={loading}
            >
              <FaUserSecret className="guest-icon" />
              Continue as Guest
            </button>
          </>
        )}

        <p className="auth-toggle">
          {isLogin
            ? "Don't have an account?"
            : "Already have an account?"}

          <button
            onClick={() =>
              setIsLogin(!isLogin)
            }
          >
            {isLogin
              ? "Create Account"
              : "Login Now"}
          </button>
        </p>
      </div>
    </div>
  );
};

export default Auth;