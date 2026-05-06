import React, { useState } from "react";
import { collection, addDoc } from "firebase/firestore";
import { db, auth, isFirebaseConfigured } from "./lib/firebase";
import useFeedbackStats from "./hooks/useFeedbackStats";

import {
  Star,
  Send,
  Sprout,
  MapPin,
  User,
  MessageSquare,
  CheckCircle2,
  ShieldCheck,
} from "lucide-react";

import "./Feedback.css";

const CROP_OPTIONS = [
  "Rice", "Wheat", "Cotton", "Sugarcane", "Maize",
  "Soybean", "Potato", "Onion", "Tomato", "Vegetables",
  "Fruits", "Other",
];

const CATEGORY_OPTIONS = [
  { value: "general", label: "💬 General Feedback" },
  { value: "feature", label: "✨ Feature Request" },
  { value: "bug", label: "🐛 Report a Bug" },
  { value: "ui", label: "🎨 UI/UX Improvement" },
  { value: "accuracy", label: "🎯 AI Accuracy" },
  { value: "other", label: "📌 Other" },
];

// ✅ FRONTEND IMAGE TESTIMONIALS
const TESTIMONIALS = [
  {
    text: "Soil alerts helped reduce irrigation cost by 20%",
    name: "Ramesh Patil",
    crop: "Rice Farmer",
    img: "/images/farmer1.png",
  },
  {
    text: "Weather alerts saved my crop from heavy rain",
    name: "Suresh Yadav",
    crop: "Wheat Farmer",
    img: "/images/farmer2.png",
  },
];

export default function Feedback() {
  const { avgRating, totalFeedbacks, loading } = useFeedbackStats();

  const [form, setForm] = useState({
    name: "",
    cropType: "",
    location: "",
    category: "general",
    message: "",
    rating: 0,
  });

  const [hoverRating, setHoverRating] = useState(0);
  const [submitted, setSubmitted] = useState(false);
  const [loadingSubmit, setLoadingSubmit] = useState(false);
  const [error, setError] = useState("");

  const handleChange = (field, value) => {
    setForm((prev) => ({ ...prev, [field]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");

    if (!form.message.trim()) return setError("Enter feedback.");
    if (form.rating === 0) return setError("Select rating.");

    if (!isFirebaseConfigured()) {
      return setError("Firebase not configured.");
    }

    setLoadingSubmit(true);

    try {
      const user = auth?.currentUser;

      await addDoc(collection(db, "feedback"), {
        userId: user?.uid || "anonymous",
        name: form.name || "Anonymous",
        cropType: form.cropType,
        location: form.location,
        category: form.category,
        message: form.message,
        rating: form.rating,
        createdAt: new Date().toISOString(),
      });

      setSubmitted(true);
    } catch (err) {
      setError("Failed: " + err.message);
    } finally {
      setLoadingSubmit(false);
    }
  };

  const handleReset = () => {
    setForm({
      name: "",
      cropType: "",
      location: "",
      category: "general",
      message: "",
      rating: 0,
    });
    setSubmitted(false);
  };

  const weeklyEstimate = Math.min(totalFeedbacks, 25);

  // SUCCESS SCREEN
  if (submitted) {
    return (
      <div className="feedback-page">
        <div className="feedback-success-card">
          <CheckCircle2 size={60} className="success-icon" />

          <h2>Thank You 🙏</h2>

          <p>
            Your feedback is received. Our team reviews it within
            <strong> 48 hours</strong>.
          </p>

          <div className="submitted-rating">
            {[1,2,3,4,5].map(s => (
              <Star key={s} size={26} fill={s <= form.rating ? "#f59e0b" : "none"} />
            ))}
          </div>

          <button onClick={handleReset} className="fb-btn-primary">
            Submit Again
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="feedback-page">
      <div className="feedback-wrapper">

        {/* LEFT PANEL */}
        <div className="feedback-info-panel">

          <img src="/images/farm-hero.jpg" alt="Farm" className="hero-image" />

          <h1>Help Us Grow Better 🌾</h1>

          <p>Your feedback improves Fasal Saathi for farmers.</p>

          {/* ✅ DYNAMIC STATS */}
          <div className="info-stats">
            <div className="info-stat-item">
              ⭐ {loading ? "..." : `${avgRating}/5`}
            </div>

            <div className="info-stat-item">
              💬 {loading ? "..." : totalFeedbacks} Feedbacks
            </div>

            <div className="info-stat-item">
              🚀 Live Improvements
            </div>
          </div>

          {/* ✅ LIVE INDICATOR */}
          <div className="live-indicator">
            🔴 {loading ? "..." : weeklyEstimate}+ farmers shared feedback
          </div>

          {/* ✅ TESTIMONIALS */}
          {TESTIMONIALS.map((t, i) => (
            <div key={i} className="info-testimonial">
              <p>"{t.text}"</p>

              <div className="testimonial-user">
                <img src={t.img} alt={t.name} />
                <div>
                  <strong>{t.name}</strong>
                  <span>{t.crop}</span>
                </div>
              </div>
            </div>
          ))}

          {/* ✅ TRUST BADGES */}
          <div className="trust-badges">
            <div><ShieldCheck size={14}/> Your data is secure</div>
            <div>📊 Used only for improving service</div>
            <div>🚫 No spam</div>
          </div>

          <div className="credibility-note">
            Built with farmer feedback across India 🌾
          </div>

        </div>

        {/* RIGHT PANEL */}
        <div className="feedback-form-panel">

          <h2>Share Feedback</h2>

          {error && <div className="fb-error">{error}</div>}

          <form onSubmit={handleSubmit}>

            {/* RATING */}
            <div className="stars-row">
              {[1,2,3,4,5].map(star => (
                <button
                  key={star}
                  type="button"
                  onMouseEnter={() => setHoverRating(star)}
                  onMouseLeave={() => setHoverRating(0)}
                  onClick={() => handleChange("rating", star)}
                >
                  <Star
                    size={34}
                    fill={(hoverRating || form.rating) >= star ? "#f59e0b" : "none"}
                  />
                </button>
              ))}
            </div>

            {/* CATEGORY */}
            <div className="category-grid">
              {CATEGORY_OPTIONS.map(cat => (
                <button
                  key={cat.value}
                  type="button"
                  className={form.category === cat.value ? "active" : ""}
                  onClick={() => handleChange("category", cat.value)}
                >
                  {cat.label}
                </button>
              ))}
            </div>

            {/* MESSAGE */}
            <textarea
              placeholder="Your feedback..."
              value={form.message}
              onChange={(e) => handleChange("message", e.target.value)}
            />

            {/* OPTIONAL */}
            <input
              type="text"
              placeholder="Your Name"
              value={form.name}
              onChange={(e) => handleChange("name", e.target.value)}
            />

            <input
              type="text"
              placeholder="Location"
              value={form.location}
              onChange={(e) => handleChange("location", e.target.value)}
            />

            <select
              value={form.cropType}
              onChange={(e) => handleChange("cropType", e.target.value)}
            >
              <option value="">Select Crop</option>
              {CROP_OPTIONS.map(c => (
                <option key={c}>{c}</option>
              ))}
            </select>

            <p className="privacy-note">
              🔒 Your data is safe. No sharing.
            </p>

            <button className="fb-submit-btn" disabled={loadingSubmit}>
              {loadingSubmit ? "Submitting..." : "Submit"}
            </button>

          </form>
        </div>
      </div>
    </div>
  );
}