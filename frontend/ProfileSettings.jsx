import React, { useState, useEffect } from "react";
import { auth, db, isFirebaseConfigured } from "./lib/firebase";
import { doc, getDoc, setDoc } from "firebase/firestore";
import { useNavigate } from "react-router-dom";
import {
  FaArrowRight,
  FaExclamationTriangle,
  FaGlobe,
  FaLock,
  FaMapMarkerAlt,
  FaSeedling,
  FaTimes,
  FaTrashAlt,
  FaUser,
} from "react-icons/fa";
import { EmailAuthProvider, reauthenticateWithCredential, signOut } from "firebase/auth";
import "./ProfileSetup.css";

const DELETE_CONFIRMATION = "DELETE";
const ACCOUNT_STORAGE_KEYS = [
  "agri:preferredLanguage",
  "agri:voiceAssistantSession",
  "agri:voiceAssistantHistory",
  "diseaseHistory",
  "pestHistory",
  "irrigationData",
  "farm_reports",
  "agriLmsProgress",
  "qrFarmBatches",
  "fasalSaathiIncome",
  "fasalSaathiDiary",
  "userId",
];
const ACCOUNT_STORAGE_PREFIXES = [
  "agri:sustainabilityHistory:",
  "p2p_chat_",
  "remote_ecdh_public_",
];

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

const CROP_OPTIONS = [
  { value: "rice", label: "🌾 Rice" },
  { value: "wheat", label: "🌾 Wheat" },
  { value: "cotton", label: "🌿 Cotton" },
  { value: "sugarcane", label: "🎋 Sugarcane" },
  { value: "maize", label: "🌽 Maize" },
  { value: "soybean", label: "🫘 Soybean" },
  { value: "potato", label: "🥔 Potato" },
  { value: "onion", label: "🧅 Onion" },
  { value: "tomato", label: "🍅 Tomato" },
  { value: "vegetables", label: "🥬 Vegetables" },
  { value: "fruits", label: "🍎 Fruits" },
  { value: "other", label: "🌱 Other" },
];

const SOIL_OPTIONS = [
  "Loamy",
  "Sandy",
  "Clay",
  "Silty",
  "Peaty",
  "Chalky",
  "Alluvial",
  "Lateritic",
];

const IRRIGATION_OPTIONS = [
  "Drip",
  "Sprinkler",
  "Flood",
  "Manual",
  "Rainfed",
  "Mixed",
];

const WEATHER_PREF_OPTIONS = [
  { value: "weather", label: "Weather alerts" },
  { value: "pest", label: "Pest alerts" },
  { value: "irrigation", label: "Irrigation advice" },
  { value: "fertilizer", label: "Fertilizer tips" },
  { value: "recommendation", label: "Recommendation summaries" },
];

const ProfileSettings = ({ user, userData }) => {
  const [name, setName] = useState("");
  const [language, setLanguage] = useState("en");
  const [role, setRole] = useState("farmer");
  const [cropType, setCropType] = useState("");
  const [preferredCrops, setPreferredCrops] = useState([]);
  const [soilType, setSoilType] = useState("");
  const [farmSize, setFarmSize] = useState("");
  const [irrigationMethod, setIrrigationMethod] = useState("");
  const [weatherPreferences, setWeatherPreferences] = useState([]);
  const [location, setLocation] = useState(null);
  const [address, setAddress] = useState("");
  const [phoneNumber, setPhoneNumber] = useState("");
  const [whatsappAlerts, setWhatsappAlerts] = useState(false);
  const [loading, setLoading] = useState(false);
  const [locLoading, setLocLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [deleteModalOpen, setDeleteModalOpen] = useState(false);
  const [deletePassword, setDeletePassword] = useState("");
  const [deleteConfirmation, setDeleteConfirmation] = useState("");
  const [deleteLoading, setDeleteLoading] = useState(false);
  const [deleteError, setDeleteError] = useState("");
  const [deleteSuccess, setDeleteSuccess] = useState("");
  const navigate = useNavigate();

  const requiresPasswordReauth =
    user?.providerData?.some((provider) => provider.providerId === "password") && Boolean(user?.email);

  useEffect(() => {
    if (!user) {
      navigate("/login");
      return;
    }

    if (userData) {
      setName(userData.displayName || "");
      setLanguage(userData.language || "en");
      setRole(userData.role || "farmer");
      setCropType(userData.cropType || "");
      setPreferredCrops(userData.preferredCrops || []);
      setSoilType(userData.soilType || "");
      setFarmSize(userData.farmSize || "");
      setIrrigationMethod(userData.irrigationMethod || "");
      setWeatherPreferences(userData.weatherPreferences || []);
      setLocation(userData.location || null);
      setAddress(userData.address || "");
      setPhoneNumber(userData.phoneNumber || "");
      setWhatsappAlerts(!!userData.whatsappAlerts);
    } else if (user && isFirebaseConfigured()) {
      const fetchUserProfile = async () => {
        try {
          const docRef = doc(db, "users", user.uid);
          const snapshot = await getDoc(docRef);
          if (snapshot.exists()) {
            const data = snapshot.data();
            setName(data.displayName || "");
            setLanguage(data.language || "en");
            setRole(data.role || "farmer");
            setCropType(data.cropType || "");
            setPreferredCrops(data.preferredCrops || []);
            setSoilType(data.soilType || "");
            setFarmSize(data.farmSize || "");
            setIrrigationMethod(data.irrigationMethod || "");
            setWeatherPreferences(data.weatherPreferences || []);
            setLocation(data.location || null);
            setAddress(data.address || "");
            setPhoneNumber(data.phoneNumber || "");
            setWhatsappAlerts(!!data.whatsappAlerts);
          }
        } catch (fetchErr) {
          console.error("Failed to load profile data:", fetchErr);
        }
      };
      fetchUserProfile();
    }
  }, [user, userData, navigate]);

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

  const handlePreferredCropsChange = (e) => {
    const values = Array.from(e.target.selectedOptions, (option) => option.value);
    setPreferredCrops(values);
  };

  const handleWeatherPreferenceToggle = (preference) => {
    setWeatherPreferences((current) =>
      current.includes(preference)
        ? current.filter((item) => item !== preference)
        : [...current, preference]
    );
  };

  const clearAccountBrowserData = () => {
    const storageKeys = new Set(ACCOUNT_STORAGE_KEYS);

    for (let index = localStorage.length - 1; index >= 0; index -= 1) {
      const key = localStorage.key(index);
      if (!key) {
        continue;
      }

      const matchesExactKey = storageKeys.has(key);
      const matchesPrefix = ACCOUNT_STORAGE_PREFIXES.some((prefix) => key.startsWith(prefix));

      if (matchesExactKey || matchesPrefix) {
        localStorage.removeItem(key);
      }
    }
  };

  const openDeleteModal = () => {
    setDeleteError("");
    setDeleteSuccess("");
    setDeletePassword("");
    setDeleteConfirmation("");
    setDeleteModalOpen(true);
  };

  const closeDeleteModal = () => {
    if (deleteLoading) {
      return;
    }

    setDeleteModalOpen(false);
    setDeleteError("");
    setDeletePassword("");
    setDeleteConfirmation("");
  };

  const handleDeleteAccount = async (e) => {
    e.preventDefault();
    setDeleteError("");
    setDeleteSuccess("");

    if (deleteConfirmation.trim().toUpperCase() !== DELETE_CONFIRMATION) {
      setDeleteError('Type DELETE exactly to confirm permanent account deletion.');
      return;
    }

    if (requiresPasswordReauth && !deletePassword.trim()) {
      setDeleteError("Re-enter your password to continue.");
      return;
    }

    const currentUser = auth.currentUser;
    if (!currentUser) {
      setDeleteError("You need to be signed in to delete your account.");
      return;
    }

    setDeleteLoading(true);

    try {
      if (requiresPasswordReauth) {
        const credential = EmailAuthProvider.credential(currentUser.email, deletePassword);
        await reauthenticateWithCredential(currentUser, credential);
      }

      const idToken = await currentUser.getIdToken(true);
      const response = await fetch("/api/account/delete", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${idToken}`,
        },
        body: JSON.stringify({
          confirmation_text: DELETE_CONFIRMATION,
          reason: "user_requested",
        }),
      });

      const result = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(result.detail || result.message || "Account deletion failed.");
      }

      clearAccountBrowserData();

      try {
        await signOut(auth);
      } catch (signOutErr) {
        console.warn("Firebase sign-out failed after account deletion:", signOutErr);
      }

      setDeleteSuccess("Your account has been permanently deleted.");
      setDeleteModalOpen(false);
      setTimeout(() => navigate("/login"), 1200);
    } catch (deleteErr) {
      setDeleteError(deleteErr.message || "Unable to delete the account right now.");
    } finally {
      setDeleteLoading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setSuccess("");

    if (!name.trim() || !cropType || preferredCrops.length === 0 || !soilType || !farmSize.trim() || !irrigationMethod) {
      setError("Please complete all required farming profile fields.");
      return;
    }

    setLoading(true);
    try {
      const currentUser = auth.currentUser;
      if (!currentUser) {
        navigate("/login");
        return;
      }

      await setDoc(
        doc(db, "users", currentUser.uid),
        {
          displayName: name.trim(),
          language,
          role,
          cropType,
          preferredCrops,
          soilType,
          farmSize: farmSize.trim(),
          farmArea: farmSize.trim(),
          irrigationMethod,
          irrigationType: irrigationMethod,
          weatherPreferences,
          location,
          address,
          phoneNumber: phoneNumber.trim(),
          whatsappAlerts,
          profileCompleted: true,
          updatedAt: new Date().toISOString(),
        },
        { merge: true }
      );

      setSuccess("Profile updated successfully.");
      setTimeout(() => navigate("/"), 1200);
    } catch (err) {
      console.error("Save profile error:", err);
      setError("Failed to update profile. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="setup-container">
      <div className="setup-card">
        <div className="setup-header">
          <FaSeedling className="setup-logo-icon" />
          <h1>Edit Farming Profile</h1>
          <p>Update your location, crop choices, soil details and recommendations preferences.</p>
        </div>

        {error && <div className="setup-error">{error}</div>}
        {success && <div className="setup-success">{success}</div>}

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

          <div className="setup-group">
            <label><FaSeedling /> Primary Crop Type</label>
            <div className="setup-input-wrapper">
              <FaSeedling className="setup-icon" />
              <select value={cropType} onChange={(e) => setCropType(e.target.value)} required>
                <option value="">Select your primary crop</option>
                {CROP_OPTIONS.map((crop) => (
                  <option key={crop.value} value={crop.value}>{crop.label}</option>
                ))}
              </select>
            </div>
          </div>

          <div className="setup-group">
            <label><FaSeedling /> Preferred Crops</label>
            <div className="setup-input-wrapper">
              <span className="setup-icon">🌱</span>
              <select
                multiple
                value={preferredCrops}
                onChange={handlePreferredCropsChange}
                size={4}
                required
              >
                {CROP_OPTIONS.map((crop) => (
                  <option key={crop.value} value={crop.value}>{crop.label}</option>
                ))}
              </select>
            </div>
          </div>

          <div className="setup-group">
            <label><FaGlobe /> Preferred Language</label>
            <div className="setup-input-wrapper">
              <FaGlobe className="setup-icon" />
              <select value={language} onChange={(e) => setLanguage(e.target.value)}>
                {LANGUAGE_OPTIONS.map((lang) => (
                  <option key={lang.value} value={lang.value}>{lang.label}</option>
                ))}
              </select>
            </div>
          </div>

          <div className="setup-group">
            <label>Soil Type</label>
            <div className="setup-input-wrapper">
              <span className="setup-icon">🧪</span>
              <select value={soilType} onChange={(e) => setSoilType(e.target.value)} required>
                <option value="">Choose your soil type</option>
                {SOIL_OPTIONS.map((soil) => (
                  <option key={soil} value={soil}>{soil}</option>
                ))}
              </select>
            </div>
          </div>

          <div className="setup-group">
            <label>Farm Size</label>
            <div className="setup-input-wrapper">
              <span className="setup-icon">📏</span>
              <input
                type="text"
                placeholder="e.g. 2.5 acres / 1.0 hectare"
                value={farmSize}
                onChange={(e) => setFarmSize(e.target.value)}
                required
              />
            </div>
          </div>

          <div className="setup-group">
            <label>Irrigation Method</label>
            <div className="setup-input-wrapper">
              <span className="setup-icon">💧</span>
              <select value={irrigationMethod} onChange={(e) => setIrrigationMethod(e.target.value)} required>
                <option value="">Choose irrigation method</option>
                {IRRIGATION_OPTIONS.map((method) => (
                  <option key={method} value={method}>{method}</option>
                ))}
              </select>
            </div>
          </div>

          <div className="setup-group">
            <label>Weather / Recommendation Preferences</label>
            <div className="checkbox-group" style={{ flexDirection: 'column', gap: '0.75rem' }}>
              {WEATHER_PREF_OPTIONS.map((option) => (
                <label key={option.value} className="checkbox-container" style={{ paddingLeft: '35px' }}>
                  <input
                    type="checkbox"
                    checked={weatherPreferences.includes(option.value)}
                    onChange={() => handleWeatherPreferenceToggle(option.value)}
                  />
                  <span className="checkmark"></span>
                  {option.label}
                </label>
              ))}
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

          <button type="submit" className="setup-submit" disabled={loading}>
            {loading ? "Saving..." : "Save Profile"}
            {!loading && <FaArrowRight />}
          </button>
        </form>

        <section className="privacy-security-section" aria-labelledby="privacy-security-title">
          <div className="privacy-security-copy">
            <div className="privacy-security-icon">
              <FaLock />
            </div>
            <div>
              <h2 id="privacy-security-title">Privacy & Security</h2>
              <p>
                Permanently delete your account and remove your profile, farm data, consultations, WhatsApp subscription,
                and other stored personal records from Fasal Saathi.
              </p>
            </div>
          </div>

          <button type="button" className="delete-account-btn" onClick={openDeleteModal}>
            <FaTrashAlt />
            Delete Account
          </button>
        </section>

        {deleteSuccess && <div className="setup-success setup-success--danger">{deleteSuccess}</div>}
      </div>

      {deleteModalOpen && (
        <div className="delete-account-overlay" role="presentation" onClick={closeDeleteModal}>
          <div
            className="delete-account-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="delete-account-title"
            aria-describedby="delete-account-description"
            onClick={(event) => event.stopPropagation()}
          >
            <button type="button" className="delete-modal-close" onClick={closeDeleteModal} aria-label="Close dialog">
              <FaTimes />
            </button>

            <div className="delete-modal-warning">
              <FaExclamationTriangle />
            </div>

            <h2 id="delete-account-title">Delete account permanently?</h2>
            <p id="delete-account-description">
              This will remove your profile, farm records, consultations, uploaded files, and device-linked data.
              This action cannot be undone.
            </p>

            <form className="delete-account-form" onSubmit={handleDeleteAccount}>
              <label className="delete-account-field">
                <span>Type DELETE to confirm</span>
                <input
                  type="text"
                  value={deleteConfirmation}
                  onChange={(event) => setDeleteConfirmation(event.target.value)}
                  placeholder="DELETE"
                  autoComplete="off"
                  autoCapitalize="characters"
                  spellCheck="false"
                />
              </label>

              {requiresPasswordReauth && (
                <label className="delete-account-field">
                  <span>Re-enter your password</span>
                  <input
                    type="password"
                    value={deletePassword}
                    onChange={(event) => setDeletePassword(event.target.value)}
                    placeholder="Password"
                    autoComplete="current-password"
                  />
                </label>
              )}

              {deleteError && <div className="delete-account-error">{deleteError}</div>}

              <div className="delete-account-actions">
                <button type="button" className="delete-account-cancel" onClick={closeDeleteModal} disabled={deleteLoading}>
                  Cancel
                </button>
                <button type="submit" className="delete-account-confirm" disabled={deleteLoading}>
                  {deleteLoading ? "Deleting..." : "Delete My Account"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};

export default ProfileSettings;
