import React, { useState, useEffect, useRef } from "react";
import { createUserWithEmailAndPassword } from "firebase/auth";
import { auth } from "./firebase"; // adjust path
import "./App.css";
import Advisor from "./Advisor";
import How from "./How";
import Home from "./Home";
import { FaHome, FaComments, FaInfoCircle, FaLeaf, FaBars, FaTimes } from "react-icons/fa";
import { useLocation } from "react-router-dom";
import { signInWithEmailAndPassword } from "firebase/auth";
import { onAuthStateChanged } from "firebase/auth";
import { signOut } from "firebase/auth";
import { useNavigate } from "react-router-dom";
import { Routes, Route, Link } from "react-router-dom";
import { GoogleAuthProvider, signInWithPopup } from "firebase/auth";
import { FaGoogle } from "react-icons/fa";
import { updateProfile } from "firebase/auth";

// 🔹 ScrollToTop component to fix navigation positioning
function ScrollToTop() {
  const { pathname } = useLocation();
  useEffect(() => { window.scrollTo(0, 0); }, [pathname]);
  return null;
}

function App() {
  const [error, setError] = useState("");
  const [loginLang, setLoginLang] = useState("");
  const [showAlert, setShowAlert] = useState(true);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [isOpen, setIsOpen] = useState(false);
  const [sunlight, setSunlight] = useState(false);
  const [signupName, setSignupName] = useState("");
const [signupEmail, setSignupEmail] = useState("");
const [signupPassword, setSignupPassword] = useState("");
const [nameError, setNameError] = useState("");
const [emailError, setEmailError] = useState("");
const [passwordError, setPasswordError] = useState("");
const navigate = useNavigate();
  const validateEmail = (email) => {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
};  
const validatePassword = (password) => {
  const regex =
    /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[!@#$%^&*(),.?":{}|<>]).{6,}$/;

  return regex.test(password);
};

  const [name, setName] = useState(localStorage.getItem("farmerName") || "");
  useEffect(() => {
  const unsubscribe = onAuthStateChanged(auth, (user) => {
    if (user) {
      setName(user.displayName || user.email);
      localStorage.setItem("farmerName", user.displayName || user.email);
    } else {
      setName("");
    }
  });
  return () => unsubscribe();
}, []);
  const [inputName, setInputName] = useState("");
  const [preferredLang, setPreferredLang] = useState(
    localStorage.getItem("preferredLanguage") || ""
  );
  const handleGoogleLogin = async () => {
  const provider = new GoogleAuthProvider();

  try {
    const result = await signInWithPopup(auth, provider);

    const user = result.user;

    localStorage.setItem("farmerName", user.displayName);
    setName(user.displayName);

    navigate("/");

  } catch (error) {
    console.error(error);
    setError("Google sign-in failed");
  }
};


  const handleLogin = async (e) => {
  e.preventDefault();
  if (!email || !password) {
  alert("Enter email & password");
  return;
}

  try {
    const userCredential = await signInWithEmailAndPassword(
      auth,
      email,
      password
    );
    localStorage.setItem("farmerName", inputName);
    setName(inputName);

    alert("Login successful!");

    navigate("/");

  } catch (error) {
    alert("Invalid email or password");
  }
};
const handleLogout = async () => {
  

  localStorage.removeItem("farmerName");
  localStorage.removeItem("preferredLanguage");

  setName("");
  setPreferredLang("");

  alert("Logged out");
  navigate("/");
};
const handleSignup = async (e) => {
  e.preventDefault();

  // reset errors
  setNameError("");
  setEmailError("");
  setPasswordError("");

  let valid = true;

  // Name validation
  if (!signupName.trim()) {
    setNameError("Name is required");
    valid = false;
  }

  // Email validation
  if (!signupEmail.trim()) {
    setEmailError("Email is required");
    valid = false;
  } else if (!validateEmail(signupEmail)) {
    setEmailError("Invalid email format");
    valid = false;
  }

  // Password validation
  if (!signupPassword.trim()) {
    setPasswordError("Password is required");
    valid = false;
  } else if (!validatePassword(signupPassword)) {
    setPasswordError(
      "Min 6 chars, 1 upper, 1 lower, 1 number, 1 special char"
    );
    valid = false;
  }

  if (!valid) {
    window.scrollTo({ top: 0, behavior: "smooth" });
    return;
  }
  

  // Firebase signup
  try {
    const userCredential = await createUserWithEmailAndPassword(
  auth,
  signupEmail,
  signupPassword
);

// 🔥 IMPORTANT LINE (name save here)
await updateProfile(userCredential.user, {
  displayName: signupName,
});

localStorage.setItem("farmerName", signupName);
setName(signupName);

navigate("/");

  } catch (error) {
    if (error.code === "auth/email-already-in-use") {
      setEmailError("Email already registered");
    }
  }
};

  return (
    <>
      <ScrollToTop />
      <div className={sunlight ? "app sunlight" : "app"}>

        {/* Navbar */}
        {/* NAVBAR */}
        <nav className="navbar">
          <div className="nav-left">
            <FaLeaf className="icon" />
            <Link to="/" className="brand">
              Fasal Saathi
            </Link>
          </div>

          <ul className={`nav-center ${isOpen ? "active" : ""}`}>
            <li>
              <Link to="/" onClick={() => setIsOpen(false)}>
                <FaHome className="icon" /> Home
              </Link>
            </li>
            <li>
              <Link to="/advisor" onClick={() => setIsOpen(false)}>
                <FaComments className="icon" /> Chat
              </Link>
            </li>
            <li>
              <Link to="/how-it-works" onClick={() => setIsOpen(false)}>
                <FaInfoCircle className="icon" /> How It Works
              </Link>
            </li>
          </ul>

          <div className="nav-right">
            <button
              onClick={() => setSunlight(!sunlight)}
              className="sunlight-toggle"
            >
              {sunlight ? "👁️ Normal View" : "☀️ Sunlight Mode"}
            </button>

            {/* Language Dropdown */}
            {/* LANGUAGE SELECT */}
            <select
              className="lang-select"
              value={preferredLang}
              onChange={(e) => {
                const lang = e.target.value;
                setPreferredLang(lang);
                localStorage.setItem("preferredLanguage", lang);
              }}
            >
              <option value="">Select Language</option>
              <option value="en">🌍 English</option>
              <option value="hi">🇮🇳 हिंदी</option>
              <option value="mr">🇮🇳 मराठी</option>
              <option value="bn">🇮🇳 বাংলা</option>
              <option value="ta">🇮🇳 தமிழ்</option>
              <option value="te">🇮🇳 తెలుగు</option>
              <option value="gu">🇮🇳 ગુજરાતી</option>
              <option value="pa">🇮🇳 ਪੰਜਾਬੀ</option>
              <option value="kn">🇮🇳 ಕನ್ನಡ</option>
              <option value="ml">🇮🇳 മലയാളം</option>
              <option value="or">🇮🇳 ଓଡ଼ିଆ</option>
            </select>
            {/* <div className="nav-auth">
  <button className="btn-outline">Login</button>
  <button className="btn-primary">Sign Up</button>
</div> */}

            {/* USER */}
            <div className="nav-user">
  {name ? (
    <>
      👋 {name}
      <button className="logout-btn" onClick={handleLogout}>
        Logout
      </button>
    </>
  ) : (
    <div className="nav-auth">
      <Link to="/login" onClick={() => setIsOpen(false)}>
        <button className="btn-outline">Login</button>
      </Link>

      <Link to="/signup" onClick={() => setIsOpen(false)}>
        <button className="btn-outline">Sign Up</button>
      </Link>
    </div>
  )}
</div>
          </div>

          <button className="hamburger" onClick={() => setIsOpen(!isOpen)}>
            {isOpen ? <FaTimes /> : <FaBars />}
          </button>
        </nav>

        {/* ALERT */}
        {showAlert && (
          <div className="alert-bar">
            🌧️ Weather Alert: Heavy rainfall expected in parts of Maharashtra this evening.
            <button className="close-btn" onClick={() => setShowAlert(false)}>
              <FaTimes />
            </button>
          </div>
        )}

        {/* ROUTES */}
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/advisor" element={<Advisor />} />
          <Route path="/how-it-works" element={<How />} />
          
          <Route
          
            path="/login"
            element={
              <div className="login-page">
                <div className="login-card">
                  <h2>👨‍🌾 Farmer Login</h2>

                <form onSubmit={handleLogin}>
  <input
  type="text"
  placeholder="Name"
  value={inputName}
  onChange={(e) => setInputName(e.target.value)}
/>

  <input
    type="email"
    placeholder="Enter your email"
    value={email}
    onChange={(e) => setEmail(e.target.value)}
  />

  <input
    type="password"
    placeholder="Enter password"
    value={password}
    onChange={(e) => setPassword(e.target.value)}
  />

  <select
    value={loginLang}
    onChange={(e) => setLoginLang(e.target.value)}
  >
    <option value="">Select Language</option>
    <option value="en">English</option>
    <option value="hi">Hindi</option>
    <option value="mr">Marathi</option>
  </select>

  <button type="submit">Login</button>
  <p>
  Don't have an account? <Link to="/signup">Signup</Link>
</p>
</form>
                </div>
              </div>

            }
          />
          <Route
  path="/signup"
  element={
    <div className="login-page">
      <div className="login-card">
        <h2>📝 Register</h2>
{error && <p className="error">{error}</p>}
        <form onSubmit={handleSignup}>
  <input
  type="text"
  placeholder="Name"
  value={signupName}
  onChange={(e) => setSignupName(e.target.value)}
  className={nameError ? "input error-input" : "input"}
/>
{nameError && <p className="error-text">{nameError}</p>}

<input
  type="email"
  placeholder="Email"
  value={signupEmail}
  onChange={(e) => setSignupEmail(e.target.value)}
  className={emailError ? "input error-input" : "input"}
/>
{emailError && <p className="error-text">{emailError}</p>}

<input
  type="password"
  placeholder="Password"
  value={signupPassword}
  onChange={(e) => setSignupPassword(e.target.value)}
  className={passwordError ? "input error-input" : "input"}
/>
{passwordError && <p className="error-text">{passwordError}</p>}

  <button type="submit">Signup</button>
</form>
{/* <p style={{ marginTop: "10px" }}>
  Already have an account?{" "}
  <Link to="/login">Login</Link>
</p> */}
<div className="social-login">
  <div className="divider">
    <span>OR</span>
  </div>

  <span className="social-icons">
    <button onClick={handleGoogleLogin} className="icon-btn google">
      <FaGoogle size={24} />
    </button>

  </span>

  <p style={{ marginTop: "10px" }}>
    Already have an account? <Link to="/login">Login</Link>
  </p>
</div>

      </div>
    </div>
  }
/>
        </Routes>
      </div>
    </>
    
  );
}

export default App;
