import React, { useState } from "react";
import {
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  signInWithPopup,
  GoogleAuthProvider,
  sendEmailVerification,
  signOut
} from "firebase/auth";
import { doc, setDoc } from "firebase/firestore";
import { useNavigate, useLocation } from "react-router-dom";
import { FaGoogle, FaEnvelope, FaLock, FaUser, FaArrowRight, FaLeaf } from "react-icons/fa";
import { auth, db, isFirebaseConfigured } from "./lib/firebase";
import "./Auth.css";

const Auth = () => {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [phoneNumber, setPhoneNumber] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");
  const navigate = useNavigate();
  const location = useLocation();

  const from = location.state?.from?.pathname || "/";

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

  const handleAuth = async (e) => {
    e.preventDefault();
    setError("");
    setMessage("");
    setLoading(true);

    try {
      if (isLogin) {
        // Login Logic
        const userCredential = await signInWithEmailAndPassword(auth, email, password);
        const user = userCredential.user;

        if (!user.emailVerified) {
          setError("Please verify your email before logging in. Check your inbox.");
          await signOut(auth);
          setLoading(false);
          return;
        }

        navigate(from, { replace: true });
      } else {
        // Sign Up Logic
        const userCredential = await createUserWithEmailAndPassword(auth, email, password);
        const user = userCredential.user;

        // Send verification email
        await sendEmailVerification(user);

        // Store user info in Firestore
        await setDoc(doc(db, "users", user.uid), {
          uid: user.uid,
          displayName: displayName,
          email: email,
          phoneNumber: phoneNumber,
          createdAt: new Date().toISOString(),
          verified: false
        });

        setMessage("Account created! Please check your email for verification link.");
        setIsLogin(true); // Switch to login after signup
      }
    } catch (err) {
      console.error(err);
      if (err.code === "auth/email-already-in-use") {
        setError("Email already in use. Try logging in.");
      } else if (err.code === "auth/invalid-credential") {
        setError("Invalid email or password.");
      } else if (err.code === "auth/weak-password") {
        setError("Password should be at least 6 characters.");
      } else {
        setError(err.message);
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

      // Create/Update user in Firestore
      // We wrap this in a try-catch to differentiate between Auth and Firestore failures
      try {
        await setDoc(doc(db, "users", user.uid), {
          uid: user.uid,
          displayName: user.displayName,
          email: user.email,
          photoURL: user.photoURL,
          lastLogin: new Date().toISOString(),
          profileCompleted: true // Google users often don't need the full setup
        }, { merge: true });
      } catch (fsErr) {
        console.error("Firestore sync error:", fsErr);
        // We continue even if Firestore fails, as the user is authenticated
      }

      navigate(from, { replace: true });
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
