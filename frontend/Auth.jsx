import React, { useState, useEffect } from "react";
import {
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  signInWithPopup,
  signInWithRedirect,
  getRedirectResult,
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
  const referralCode = new URLSearchParams(location.search).get("ref") || "";
  const redirectAfterAuth = referralCode
    ? `/referrals?ref=${encodeURIComponent(referralCode)}`
    : from;

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
      console.error("Guest login error:", err);
      setError("Failed to start guest session.");
    } finally {
      setLoading(false);
    }
  };

  const handleAuth = async (e) => {
    e.preventDefault();
    setError("");
    setMessage("");
    setLoading(true);

    try {
      if (isLogin) {
        // Login Logic
        const anonymousUser = auth.currentUser?.isAnonymous ? auth.currentUser : null;
        const anonymousUid = anonymousUser?.uid;

        const userCredential = await signInWithEmailAndPassword(auth, email, password);
        const user = userCredential.user;

        if (!user.emailVerified) {
          setError("Please verify your email before logging in. Check your inbox.");
          await signOut(auth);
          setLoading(false);
          return;
        }

        // If there was a guest session, migrate data to the logged-in account
        if (anonymousUid && user.uid !== anonymousUid) {
          try {
            await migrateUserData(anonymousUid, user.uid);
            setMessage("Guest data successfully merged with your account!");
          } catch (migrateErr) {
            console.error("Migration error:", migrateErr);
            // Non-fatal, user is logged in
          }
        }

        navigate(redirectAfterAuth, { replace: true });
      } else {
        // Sign Up Logic
        const anonymousUser = auth.currentUser?.isAnonymous ? auth.currentUser : null;
        
        let user;
        if (anonymousUser) {
          // Link anonymous account to email/password
          const credential = EmailAuthProvider.credential(email, password);
          try {
            const linkedCredential = await linkWithCredential(anonymousUser, credential);
            user = linkedCredential.user;
          } catch (linkErr) {
            if (linkErr.code === "auth/email-already-in-use") {
              setError("Email already in use. Please login instead to merge your guest data.");
              setLoading(false);
              return;
            }
            throw linkErr;
          }
        } else {
          const userCredential = await createUserWithEmailAndPassword(auth, email, password);
          user = userCredential.user;
        }

        // Send verification email
        await sendEmailVerification(user);

        // Store/Update user info in Firestore
        // role: "farmer" is written explicitly so the backend's verify_role
        // function never has to fall back to a default — the field is always
        // present from the moment the account is created.
        await setDoc(doc(db, "users", user.uid), {
          uid: user.uid,
          displayName: displayName,
          email: email,
          phoneNumber: phoneNumber,
          createdAt: new Date().toISOString(),
          verified: false,
          role: "farmer",
          reputation: 0,
          profileCompleted: false
        }, { merge: true });

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

  const syncUserToFirestore = async (user) => {
    try {
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
        ...(isNewUser && { role: "farmer" }),
      }, { merge: true });
    } catch (fsErr) {
      console.error("Firestore sync error:", fsErr);
    }
    navigate(redirectAfterAuth, { replace: true });
  };

  // Handle redirect result on mount (user returning from Google OAuth redirect)
  useEffect(() => {
    getRedirectResult(auth).then((result) => {
      if (!result) return;
      setLoading(true);
      syncUserToFirestore(result.user);
    }).catch((err) => {
      console.warn("Redirect auth result error:", err);
    });
  }, []);

  const handleGoogleLogin = async () => {
    const provider = new GoogleAuthProvider();
    provider.setCustomParameters({ prompt: 'select_account' });

    setLoading(true);
    setError("");
    try {
      const result = await signInWithPopup(auth, provider);
      await syncUserToFirestore(result.user);
    } catch (err) {
      console.error("Google Auth Popup Error:", err);

      // Popup-specific failures → fall back to redirect
      const shouldRedirect = [
        "auth/popup-blocked",
        "auth/popup-closed-by-user",
        "auth/cancelled-by-user",
      ].includes(err.code);

      if (shouldRedirect) {
        try {
          await signInWithRedirect(auth, provider);
          return;
        } catch (redirectErr) {
          console.error("Redirect auth fallback also failed:", redirectErr);
        }
      }

      if (err.code === "auth/popup-closed-by-user" || err.code === "auth/cancelled-by-user") {
        setError("");
      } else if (err.code === "auth/operation-not-allowed") {
        setError("Google sign-in is not enabled in Firebase Console.");
      } else if (err.code === "auth/popup-blocked") {
        setError("Popup blocked and redirect fallback failed. Please allow popups or try a different browser.");
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
