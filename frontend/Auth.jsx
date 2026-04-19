import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { FaGoogle, FaEnvelope, FaLock, FaUser, FaArrowRight, FaLeaf } from "react-icons/fa";
import { auth, db, doc, getDoc, setDoc, isFirebaseConfigured } from "./lib/firebase";
import "./Auth.css";

const Auth = () => {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");
  const navigate = useNavigate();

  if (!isFirebaseConfigured()) {
    return (
      <div className="auth-container">
        <div className="auth-card">
          <div className="auth-logo">
            <FaLeaf className="leaf-icon" />
            <h1>Fasal Saathi</h1>
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
      const { createUserWithEmailAndPassword, signInWithEmailAndPassword } = await import("firebase/auth");
      const { doc, setDoc } = await import("firebase/firestore");

      if (isLogin) {
        const userCredential = await signInWithEmailAndPassword(auth, email, password);
        navigate("/");
      } else {
        const userCredential = await createUserWithEmailAndPassword(auth, email, password);
        await setDoc(doc(db, "users", userCredential.user.uid), {
          email: userCredential.user.email,
          displayName: displayName,
          profileCompleted: false,
          createdAt: new Date().toISOString(),
        });
        setMessage("Account created! Please complete your profile.");
        setTimeout(() => navigate("/profile-setup"), 1500);
      }
    } catch (err) {
      setError(err.message || "Authentication failed");
    } finally {
      setLoading(false);
    }
  };

  const handleGoogleAuth = async () => {
    setError("");
    setLoading(true);

    try {
      const { signInWithPopup, GoogleAuthProvider } = await import("firebase/auth");
      const { doc, setDoc, getDoc } = await import("firebase/firestore");

      const provider = new GoogleAuthProvider();
      const result = await signInWithPopup(auth, provider);
      const user = result.user;

      const userDoc = await getDoc(doc(db, "users", user.uid));
      if (!userDoc.exists()) {
        await setDoc(doc(db, "users", user.uid), {
          email: user.email,
          displayName: user.displayName,
          profileCompleted: false,
          createdAt: new Date().toISOString(),
        });
        navigate("/profile-setup");
      } else {
        navigate("/");
      }
    } catch (err) {
      setError(err.message || "Google sign-in failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-container">
      <div className="auth-card">
        <div className="auth-logo">
          <FaLeaf className="leaf-icon" />
          <h1>Fasal Saathi</h1>
        </div>
        <p className="auth-subtitle">{isLogin ? "Welcome back, Farmer!" : "Join Fasal Saathi"}</p>

        {error && <div className="auth-error">{error}</div>}
        {message && <div className="auth-message">{message}</div>}

        <form onSubmit={handleAuth} className="auth-form">
          {!isLogin && (
            <div className="form-group">
              <FaUser className="input-icon" />
              <input
                type="text"
                placeholder="Full Name"
                value={displayName}
                onChange={(e) => setDisplayName(e.target.value)}
                required
              />
            </div>
          )}

          <div className="form-group">
            <FaEnvelope className="input-icon" />
            <input
              type="email"
              placeholder="Email Address"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </div>

          <div className="form-group">
            <FaLock className="input-icon" />
            <input
              type="password"
              placeholder="Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              minLength={6}
            />
          </div>

          <button type="submit" className="auth-button" disabled={loading}>
            {loading ? "Please wait..." : isLogin ? "Login" : "Sign Up"}
            <FaArrowRight className="button-icon" />
          </button>
        </form>

        <div className="auth-divider">
          <span>or</span>
        </div>

        <button onClick={handleGoogleAuth} className="google-button" disabled={loading}>
          <FaGoogle className="google-icon" />
          Continue with Google
        </button>

        <p className="auth-toggle">
          {isLogin ? "Don't have an account? " : "Already have an account? "}
          <button onClick={() => setIsLogin(!isLogin)} type="button">
            {isLogin ? "Sign Up" : "Login"}
          </button>
        </p>
      </div>
    </div>
  );
};

export default Auth;