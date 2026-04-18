import React, { useState, useEffect } from "react";
import { auth, db, isFirebaseConfigured } from "./lib/firebase";
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
        const { doc, updateDoc } = await import("firebase/firestore");
        await updateDoc(doc(db, "users", user.uid), {
          displayName: name,
          language: language,
          cropType: cropType,
          location: location,
          address: address,
          profileCompleted: true,
        });
        navigate("/");
      }
    } catch (err) {
      setError(err.message || "Failed to save profile");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="profile-setup-container">
      <div className="profile-setup-card">
        <h1>Complete Your Profile</h1>
        <p className="subtitle">Help us serve you better</p>

        {error && <div className="error-message">{error}</div>}

        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label>
              <FaUser /> Name
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Enter your name"
              required
            />
          </div>

          <div className="form-group">
            <label>
              <FaGlobe /> Language
            </label>
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

          <div className="form-group">
            <label>
              <FaSeedling /> Primary Crop
            </label>
            <select
              value={cropType}
              onChange={(e) => setCropType(e.target.value)}
              required
            >
              <option value="">Select your primary crop</option>
              <option value="rice">Rice</option>
              <option value="wheat">Wheat</option>
              <option value="cotton">Cotton</option>
              <option value="sugarcane">Sugarcane</option>
              <option value="maize">Maize</option>
              <option value="soybean">Soybean</option>
              <option value="potato">Potato</option>
              <option value="onion">Onion</option>
              <option value="tomato">Tomato</option>
              <option value="vegetables">Vegetables</option>
              <option value="fruits">Fruits</option>
              <option value="other">Other</option>
            </select>
          </div>

          <div className="form-group">
            <label>
              <FaMapMarkerAlt /> Location
            </label>
            <div className="location-input">
              <input
                type="text"
                value={address}
                onChange={(e) => setAddress(e.target.value)}
                placeholder={locLoading ? "Getting location..." : "Enter your location"}
                required
              />
              <button
                type="button"
                className="location-btn"
                onClick={requestLocation}
                disabled={locLoading}
              >
                {locLoading ? "..." : "📍"}
              </button>
            </div>
          </div>

          <button type="submit" className="submit-btn" disabled={loading}>
            {loading ? "Saving..." : "Complete Profile"}
            <FaArrowRight />
          </button>
        </form>
      </div>
    </div>
  );
};

export default ProfileSetup;