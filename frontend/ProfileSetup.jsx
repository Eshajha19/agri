import React, { useState, useEffect } from "react";
import { auth, db, isFirebaseConfigured } from "./lib/firebase";
import { doc, setDoc } from "firebase/firestore";
import { useNavigate } from "react-router-dom";
import { FaUser, FaGlobe, FaMapMarkerAlt, FaSeedling, FaArrowRight } from "react-icons/fa";
import "./ProfileSetup.css";

const LANGUAGE_OPTIONS = [
  { value: "en", label: "🌍 English" },
  { value: "hi", label: "🇮🇳 हिंदी" },
  { value: "mr", label: "🇮🇳 मराठी" },
  { value: "bn", label: "🇮🇳 বাংলা" },
  { value: "ta", label: "🇮🇳 தமிழ்" },
  { value: "te", label: "🇮🇳 తెలుగు" },
  { value: "gu", label: "🇮🇳 ગુજરાતી" },
  { value: "pa", label: "🇮🇳 ਪੰਜਾਬੀ" },
  { value: "kn", label: "🇮🇳 ಕನ್ನಡ" },
  { value: "ml", label: "🇮🇳 മലയാളം" },
  { value: "or", label: "🇮🇳 ଓଡ଼ିଆ" },
  { value: "as", label: "🇮🇳 অসমীয়া" },
];

const ProfileSetup = ({ user, profileCompleted }) => {
  const [name, setName] = useState("");
  const [language, setLanguage] = useState("en");
  const [cropType, setCropType] = useState("");
  const [location, setLocation] = useState(null);
  const [address, setAddress] = useState("");
  const [phoneNumber, setPhoneNumber] = useState("");
  const [whatsappAlerts, setWhatsappAlerts] = useState(false);
  const [loading, setLoading] = useState(false);
  const [locLoading, setLocLoading] = useState(false);
  const [error, setError] = useState("");
  const navigate = useNavigate();

  useEffect(() => {
    // If auth state is resolved and user is not logged in, go to login
    if (user === null && !loading) {
      navigate("/login");
      return;
    }
    
    // If profile is already completed, go home
    if (user && profileCompleted) {
      navigate("/");
    }
  }, [user, profileCompleted, navigate]);

  useEffect(() => {
    if (!isFirebaseConfigured()) {
      navigate("/login");
      return;
    }
    
    // If profile is already completed, go home
    if (user && profileCompleted) {
      navigate("/");
    } else if (!user && !localStorage.getItem("isLoggingIn")) {
      navigate("/login");
    }
  }, [user, profileCompleted, navigate]);

  useEffect(() => {
    requestLocation();
  }, []);

  const requestLocation = () => {
    if ("geolocation" in navigator) {
      setLocLoading(true);
      setError("");
      navigator.geolocation.getCurrentPosition(
        async (position) => {
          const lat = position.coords.latitude;
          const lng = position.coords.longitude;
          setLocation({ lat, lng });

          try {
            const response = await fetch(
              `https://api.bigdatacloud.net/data/reverse-geocode-client?latitude=${lat}&longitude=${lng}&localityLanguage=en`
            );
            const data = await response.json();
            
            if (data) {
              const locality = data.locality || data.city || "";
              const principalSubdivision = data.principalSubdivision || "";
              const country = data.countryName || "";
              
              let formattedAddress = locality;
              if (principalSubdivision) {
                formattedAddress += (formattedAddress ? ", " : "") + principalSubdivision;
              }
              if (!formattedAddress && country) {
                formattedAddress = country;
              }
              
              setAddress(formattedAddress || "Location Found");
            } else {
              setAddress("Location Detected");
            }
          } catch (err) {
            console.error("Geocoding error:", err);
            setAddress("Location Detected");
          }
          setLocLoading(false);
        },
        (err) => {
          console.error(err);
          setError("Location access denied. Please enable location for better service.");
          setLocLoading(false);
        }
      );
    } else {
      setError("Geolocation is not supported by your browser.");
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");

    if (!name || !cropType) {
      setError("Please fill in all details.");
      return;
    }

    setLoading(true);
    try {
      const currentUser = auth.currentUser;
      if (currentUser) {
        // Save profile to Firestore
        await setDoc(doc(db, "users", currentUser.uid), {
          displayName: name,
          language: language,
          cropType: cropType,
          location: location,
          address: address,
          phoneNumber: phoneNumber,
          whatsappAlerts: whatsappAlerts,
          profileCompleted: true,
          reputation: 0,
          badges: [],
          updatedAt: new Date().toISOString()
        }, { merge: true });

        // If WhatsApp alerts are enabled, subscribe via backend
        if (whatsappAlerts && phoneNumber) {
          try {
            await fetch("http://localhost:8000/api/whatsapp/subscribe", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                phone_number: phoneNumber,
                user_id: currentUser.uid,
                name: name
              })
            });
          } catch (whatsappErr) {
            console.error("WhatsApp subscription error:", whatsappErr);
          }
        }
        
        navigate("/");
      } else {
        navigate("/login");
      }
    } catch (err) {
      console.error("Save profile error:", err);
      setError("Failed to save profile. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="setup-container">
      <div className="setup-card">
        <div className="setup-header">
          <FaSeedling className="setup-logo-icon" />
          <h1>Complete Your Profile</h1>
            <p>Help us personalize your <span className="notranslate" translate="no">Fasal Saathi</span> experience</p>
        </div>

        {error && <div className="setup-error">{error}</div>}

        <form onSubmit={handleSubmit} className="setup-form">
          <div className="setup-group">
            <label><FaUser /> Farmer Name</label>
            <div className="setup-input-wrapper">
              <FaUser className="setup-icon" />
              <input
                type="text"
                placeholder="Enter your full name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                required
              />
            </div>
          </div>

          <div className="setup-group">
            <label><FaGlobe /> Preferred Language</label>
            <div className="setup-input-wrapper">
              <FaGlobe className="setup-icon" />
              <select
                value={language}
                onChange={(e) => setLanguage(e.target.value)}
              >
                {LANGUAGE_OPTIONS.map((lang) => (
                  <option key={lang.value} value={lang.value}>
                    {lang.label}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="setup-group">
            <label><FaSeedling /> Primary Crop Type</label>
            <div className="setup-input-wrapper">
              <FaSeedling className="setup-icon" />
              <select
                value={cropType}
                onChange={(e) => setCropType(e.target.value)}
                required
              >
                <option value="">Select your primary crop</option>
                <option value="rice">🌾 Rice</option>
                <option value="wheat">🌾 Wheat</option>
                <option value="cotton">🌿 Cotton</option>
                <option value="sugarcane">🎋 Sugarcane</option>
                <option value="maize">🌽 Maize</option>
                <option value="soybean">🫘 Soybean</option>
                <option value="potato">🥔 Potato</option>
                <option value="onion">🧅 Onion</option>
                <option value="tomato">🍅 Tomato</option>
                <option value="vegetables">🥬 Vegetables</option>
                <option value="fruits">🍎 Fruits</option>
                <option value="other">🌱 Other</option>
              </select>
            </div>
          </div>
          
          <div className="setup-group">
            <label><FaMapMarkerAlt /> Phone Number (for WhatsApp Alerts)</label>
            <div className="setup-input-wrapper">
              <span className="setup-icon" style={{ fontSize: '1.2rem' }}>📱</span>
              <input
                type="tel"
                placeholder="e.g. +91 98765 43210"
                value={phoneNumber}
                onChange={(e) => setPhoneNumber(e.target.value)}
              />
            </div>
          </div>

          <div className="setup-group checkbox-group">
            <label className="checkbox-container">
              <input
                type="checkbox"
                checked={whatsappAlerts}
                onChange={(e) => setWhatsappAlerts(e.target.checked)}
              />
              <span className="checkmark"></span>
              Receive real-time weather & pest alerts on WhatsApp
            </label>
          </div>

          <div className="setup-group">
            <label><FaMapMarkerAlt /> Farm Location</label>
            <div className={`loc-box ${address ? 'success' : locLoading ? 'pending' : ''}`}>
              {locLoading ? (
                <>
                  <span className="loc-status">📍 Getting your location...</span>
                  <div className="small-spinner"></div>
                </>
              ) : address ? (
                <>
                  <span className="loc-status">✅ {address}</span>
                  <button type="button" onClick={requestLocation} className="loc-btn">
                    Update
                  </button>
                </>
              ) : (
                <>
                  <span className="loc-status">Click to get your location</span>
                  <button type="button" onClick={requestLocation} className="loc-btn">
                    Get Location
                  </button>
                </>
              )}
            </div>
          </div>

          <button type="submit" className="setup-submit" disabled={loading}>
            {loading ? "Saving..." : "Start Journey"}
            {!loading && <FaArrowRight />}
          </button>
        </form>
      </div>
    </div>
  );
};

export default ProfileSetup;
