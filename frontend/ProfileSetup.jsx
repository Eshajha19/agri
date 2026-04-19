import React, { useState, useEffect } from "react";
import { auth, db, doc, getDoc, updateDoc, isFirebaseConfigured } from "./lib/firebase";
import { useNavigate } from "react-router-dom";
import { FaUser, FaGlobe, FaMapMarkerAlt, FaSeedling, FaArrowRight } from "react-icons/fa";
import "./ProfileSetup.css";

const LANGUAGE_OPTIONS = [
  { value: "en", label: "🌍 English" },
  { value: "hi", label: "🇮🇳 हिंदी" },
  { value: "mr", label: "🇮🇳 मराठी" },
  { value: "bn", label: "🇮🇳 বাংলা" },
  { value: "ta", label: "🇮🇳 தமிழ்" },
  { value: "te", label: "🇮🇳 তেলুগు" },
  { value: "gu", label: "🇮🇳 ગુજરાતી" },
  { value: "pa", label: "🇮🇳 ਪੰਜਾਬੀ" },
  { value: "kn", label: "🇮🇳 ಕನ್ನಡ" },
  { value: "ml", label: "🇮🇳 മലയാളം" },
  { value: "or", label: "🇮🇳 ଓଡ଼ିଆ" },
  { value: "as", label: "🇮🇳 অসমীয়া" },
];

const ProfileSetup = () => {
  const [name, setName] = useState("");
  const [language, setLanguage] = useState("en");
  const [cropType, setCropType] = useState("");
  const [location, setLocation] = useState(null);
  const [address, setAddress] = useState("");
  const [loading, setLoading] = useState(false);
  const [locLoading, setLocLoading] = useState(false);
  const [error, setError] = useState("");
  const navigate = useNavigate();

  useEffect(() => {
    if (!isFirebaseConfigured()) {
      navigate("/auth");
      return;
    }
    requestLocation();
    
    const checkExistingData = async () => {
      if (auth?.currentUser) {
        try {
          const { doc, getDoc } = await import("firebase/firestore");
          const userDoc = await getDoc(doc(db, "users", auth.currentUser.uid));
          if (userDoc.exists() && userDoc.data().profileCompleted) {
            navigate("/");
          }
        } catch (err) {
          console.error(err);
        }
      }
    };
    checkExistingData();
  }, []);

  const requestLocation = () => {
    if ("geolocation" in navigator) {
      setLocLoading(true);
      navigator.geolocation.getCurrentPosition(
        async (position) => {
          const { latitude, longitude } = position.coords;
          setLocation({ lat: latitude, lng: longitude });

          try {
            const response = await fetch(
              `https://api.bigdatacloud.net/data/reverse-geocode-client?latitude=${latitude}&longitude=${longitude}&localityLanguage=en`
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
    if (!name || !cropType) {
      setError("Please fill in all details.");
      return;
    }
    setLoading(true);
    try {
      const user = auth?.currentUser;
      if (user) {
        const { doc, setDoc } = await import("firebase/firestore");
        await setDoc(doc(db, "users", user.uid), {
          displayName: name,
          language: language,
          cropType: cropType,
          location: location,
          address: address,
          profileCompleted: true,
        }, { merge: true });
        navigate("/");
      }
    } catch (err) {
      setError(err.message || "Failed to save profile");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="setup-container">
      <div className="setup-card">
        <div className="setup-header">
          <div className="setup-logo">🌱</div>
          <h1>Welcome to Fasal Saathi</h1>
          <p>Help us serve you better</p>
        </div>

        {error && <div className="setup-error">{error}</div>}

        <form onSubmit={handleSubmit} className="setup-form">
          <div className="setup-group">
            <label>
              <FaUser /> Your Name
            </label>
            <div className="setup-input">
              <FaUser className="setup-icon" />
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Enter your full name"
                className="setup-input"
                required
              />
            </div>
          </div>

          <div className="setup-group">
            <label>
              <FaGlobe /> Preferred Language
            </label>
            <div className="setup-input">
              <FaGlobe className="setup-icon" />
              <select
                value={language}
                onChange={(e) => setLanguage(e.target.value)}
                className="setup-input"
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
            <label>
              <FaSeedling /> Primary Crop You Grow
            </label>
            <div className="setup-input">
              <FaSeedling className="setup-icon" />
              <select
                value={cropType}
                onChange={(e) => setCropType(e.target.value)}
                className="setup-input"
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
            <label>
              <FaMapMarkerAlt /> Your Location
            </label>
            <div className={`loc-box ${address ? 'success' : locLoading ? 'pending' : ''}`}>
              {locLoading ? (
                <>
                  <span>📍 Getting your location...</span>
                  <span className="loading-spinner"></span>
                </>
              ) : address ? (
                <>
                  <span>✅ {address}</span>
                  <button type="button" onClick={requestLocation} className="loc-btn">
                    Update
                  </button>
                </>
              ) : (
                <>
                  <span>Click to get your location</span>
                  <button type="button" onClick={requestLocation} className="loc-btn">
                    Get Location
                  </button>
                </>
              )}
            </div>
            <input
              type="hidden"
              value={address}
              onChange={(e) => setAddress(e.target.value)}
            />
            <input
              type="hidden"
              value={location ? JSON.stringify(location) : ""}
            />
          </div>

          <button type="submit" className="setup-submit" disabled={loading}>
            {loading ? "Saving..." : "Complete Setup"}
            {!loading && <FaArrowRight />}
          </button>
        </form>
      </div>
    </div>
  );
};

export default ProfileSetup;