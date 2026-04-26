import React, { useState } from "react";
import { collection, addDoc } from "firebase/firestore";
import { db, auth, isFirebaseConfigured } from "./lib/firebase";
import {
  Star,
  Send,
  Sprout,
  MapPin,
  User,
  MessageSquare,
  CheckCircle2,
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

export default function Feedback() {
  const [form, setForm] = useState({
    name: "",
    cropType: "",
    location: "",
    category: "general",
    message: "",
    rating: 0,
  });
  const [hoverRating, setHoverRating] = useState(0);
  const [loading, setLoading] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [error, setError] = useState("");

  const handleChange = (field, value) => {
    setForm((prev) => ({ ...prev, [field]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");

    if (!form.message.trim()) {
      setError("Please enter your feedback message.");
      return;
    }
    if (form.rating === 0) {
      setError("Please select a rating.");
      return;
    }

    if (!isFirebaseConfigured()) {
      setError("Firebase is not configured. Please check your .env file.");
      return;
    }

    setLoading(true);
    try {
      const user = auth?.currentUser;
      await addDoc(collection(db, "feedback"), {
        userId: user?.uid || "anonymous",
        userEmail: user?.email || "anonymous",
        name: form.name || (user?.displayName ?? "Anonymous"),
        cropType: form.cropType,
        location: form.location,
        category: form.category,
        message: form.message,
        rating: form.rating,
        createdAt: new Date().toISOString(),
      });
      setSubmitted(true);
    } catch (err) {
      console.error("Feedback submit error:", err);
      setError("Failed to submit feedback. Please try again: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setForm({ name: "", cropType: "", location: "", category: "general", message: "", rating: 0 });
    setSubmitted(false);
    setError("");
  };

  if (submitted) {
    return (
      <div className="feedback-page">
        <div className="feedback-success-card">
          <div className="success-icon-ring">
            <CheckCircle2 size={64} className="success-icon" />
          </div>
          <h2>Thank You! 🙏</h2>
          <p>Your feedback has been submitted successfully. We'll use it to make Fasal Saathi even better for farmers like you.</p>
          <div className="submitted-rating">
            {[1, 2, 3, 4, 5].map((s) => (
              <Star
                key={s}
                size={28}
                className={s <= form.rating ? "star-filled" : "star-empty"}
                fill={s <= form.rating ? "#f59e0b" : "none"}
              />
            ))}
          </div>
          <button className="fb-btn-primary" onClick={handleReset}>
            Submit Another Feedback
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="feedback-page">
      <div className="feedback-wrapper">
        {/* Left Panel */}
        <div className="feedback-info-panel">
          <div className="info-badge">🌾 Farmer Feedback</div>
          <h1>Help Us Grow Better</h1>
          <p>
            Your opinion directly shapes the future of Fasal Saathi. Share your
            experience, suggest features, or report issues — every word matters.
          </p>
          <div className="info-stats">
            {[
              { icon: "⭐", label: "Average Rating", value: "4.8/5" },
              { icon: "💬", label: "Feedbacks Received", value: "2,400+" },
              { icon: "🚀", label: "Features Added from Feedback", value: "18+" },
            ].map((stat, i) => (
              <div key={i} className="info-stat-item">
                <span className="stat-emoji">{stat.icon}</span>
                <div>
                  <div className="stat-value">{stat.value}</div>
                  <div className="stat-label">{stat.label}</div>
                </div>
              </div>
            ))}
          </div>
          <div className="info-testimonial">
            <p>"This app doubled my yield last season. My feedback on soil analysis was implemented!"</p>
            <span>— Ramesh Kumar, Maharashtra</span>
          </div>
        </div>

        {/* Right Panel - Form */}
        <div className="feedback-form-panel">
          <h2>Share Your Feedback</h2>

          {error && <div className="fb-error">{error}</div>}

          <form onSubmit={handleSubmit} className="fb-form">
            {/* Star Rating */}
            <div className="fb-rating-section">
              <label>Overall Rating *</label>
              <div className="stars-row">
                {[1, 2, 3, 4, 5].map((star) => (
                  <button
                    key={star}
                    type="button"
                    className="star-btn"
                    onMouseEnter={() => setHoverRating(star)}
                    onMouseLeave={() => setHoverRating(0)}
                    onClick={() => handleChange("rating", star)}
                    aria-label={`Rate ${star} star`}
                  >
                    <Star
                      size={36}
                      className="star-icon"
                      fill={(hoverRating || form.rating) >= star ? "#f59e0b" : "none"}
                      stroke={(hoverRating || form.rating) >= star ? "#f59e0b" : "#cbd5e1"}
                    />
                  </button>
                ))}
                {form.rating > 0 && (
                  <span className="rating-label">
                    {["", "Poor", "Fair", "Good", "Great", "Excellent!"][form.rating]}
                  </span>
                )}
              </div>
            </div>

            {/* Category */}
            <div className="fb-group">
              <label>Feedback Category</label>
              <div className="category-grid">
                {CATEGORY_OPTIONS.map((cat) => (
                  <button
                    key={cat.value}
                    type="button"
                    className={`cat-chip ${form.category === cat.value ? "active" : ""}`}
                    onClick={() => handleChange("category", cat.value)}
                  >
                    {cat.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Message */}
            <div className="fb-group">
              <label><MessageSquare size={15} /> Feedback / Suggestions *</label>
              <textarea
                rows="4"
                placeholder="Share your experience, suggestions, or report an issue..."
                value={form.message}
                onChange={(e) => handleChange("message", e.target.value)}
                required
              />
            </div>

            {/* Optional Fields */}
            <div className="fb-row">
              <div className="fb-group">
                <label><User size={15} /> Your Name (Optional)</label>
                <input
                  type="text"
                  placeholder="e.g. Ramesh Kumar"
                  value={form.name}
                  onChange={(e) => handleChange("name", e.target.value)}
                />
              </div>
              <div className="fb-group">
                <label><MapPin size={15} /> Location (Optional)</label>
                <input
                  type="text"
                  placeholder="e.g. Nashik, Maharashtra"
                  value={form.location}
                  onChange={(e) => handleChange("location", e.target.value)}
                />
              </div>
            </div>

            <div className="fb-group">
              <label><Sprout size={15} /> Primary Crop (Optional)</label>
              <select
                value={form.cropType}
                onChange={(e) => handleChange("cropType", e.target.value)}
              >
                <option value="">Select your main crop</option>
                {CROP_OPTIONS.map((c) => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </select>
            </div>

            <button type="submit" className="fb-submit-btn" disabled={loading}>
              {loading ? (
                <><span className="fb-spinner"></span> Submitting...</>
              ) : (
                <><Send size={18} /> Submit Feedback</>
              )}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
