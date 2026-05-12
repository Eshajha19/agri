import React, { useState, useEffect } from "react";
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
  "Rice",
  "Wheat",
  "Cotton",
  "Sugarcane",
  "Maize",
  "Soybean",
  "Potato",
  "Onion",
  "Tomato",
  "Vegetables",
  "Fruits",
  "Other",
];

const CATEGORY_OPTIONS = [
  { value: "general", label: "💬 General Feedback" },
  { value: "feature", label: "✨ Feature Request" },
  { value: "bug", label: "🐛 Report a Bug" },
  { value: "ui", label: "🎨 UI/UX Improvement" },
  { value: "accuracy", label: "🎯 AI Accuracy" },
  { value: "other", label: "📌 Other" },
];

const TESTIMONIALS = [
  {
    text: "This app doubled my yield last season. My feedback on soil analysis was implemented!",
    author: "Ramesh Kumar",
    location: "Maharashtra",
  },
  {
    text: "Accurate weather alerts helped me save my crops during unexpected rain.",
    author: "Sunita Devi",
    location: "Bihar",
  },
  {
    text: "The pest detection feature is a game changer. Very easy to use!",
    author: "Arjun Patel",
    location: "Gujarat",
  },
  {
    text: "I suggested adding crop-specific tips and they actually added it!",
    author: "Mahesh Yadav",
    location: "Uttar Pradesh",
  },
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
  const [testimonialIndex, setTestimonialIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setTestimonialIndex(
        (prev) => (prev + 1) % TESTIMONIALS.length
      );
    }, 4000);

    return () => clearInterval(interval);
  }, []);

  const handleChange = (field, value) => {
    setForm((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");

    if (!form.message.trim()) {
      setError("Please enter your feedback.");
      return;
    }

    if (form.rating === 0) {
      setError("Please provide a rating.");
      return;
    }

    if (!isFirebaseConfigured()) {
      setError(
        "Firebase is not configured properly."
      );
      return;
    }

    setLoading(true);

    try {
      const user = auth?.currentUser;

      await addDoc(collection(db, "feedback"), {
        userId: user?.uid || "anonymous",
        userEmail: user?.email || "anonymous",
        name:
          form.name ||
          user?.displayName ||
          "Anonymous",
        cropType: form.cropType,
        location: form.location,
        category: form.category,
        message: form.message,
        rating: form.rating,
        createdAt: new Date().toISOString(),
      });

      setSubmitted(true);
    } catch (err) {
      console.error(err);
      setError(
        "Failed to submit feedback. Please try again."
      );
    } finally {
      setLoading(false);
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
    setError("");
  };

  if (submitted) {
    return (
      <div className="feedback-page">
        <div className="success-card">
          <div className="success-icon-ring">
            <CheckCircle2
              size={70}
              className="success-icon"
            />
          </div>

          <h2>Thank You! 🌾</h2>

          <p>
            Your feedback helps us improve{" "}
            <span
              className="notranslate"
              translate="no"
            >
              Fasal Saathi
            </span>{" "}
            for farmers across India.
          </p>

          <div className="submitted-rating">
            {[1, 2, 3, 4, 5].map((star) => (
              <Star
                key={star}
                size={28}
                fill={
                  star <= form.rating
                    ? "#f59e0b"
                    : "none"
                }
                stroke={
                  star <= form.rating
                    ? "#f59e0b"
                    : "#cbd5e1"
                }
              />
            ))}
          </div>

          <button
            className="fb-submit-btn"
            onClick={handleReset}
          >
            Submit Another Feedback
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
          <div className="info-badge">
            🌾 Farmer Feedback
          </div>

          <h1>Help Us Grow Better</h1>

          <p>
            Your opinion directly shapes the future
            of{" "}
            <span
              className="notranslate"
              translate="no"
            >
              Fasal Saathi
            </span>
            . Share your experience, suggest new
            features, or report issues — every word
            matters.
          </p>

          {/* Farmer Images */}

          <div className="farmer-showcase">
            <div className="farmer-images">
              <img
                src="/farmer1.png"
                alt="Farmer"
                className="farmer-img"
              />

              <img
                src="/farmer2.png"
                alt="Farmer"
                className="farmer-img"
              />
            </div>
          </div>

          {/* Stats */}

          <div className="info-stats">
            {[
              {
                icon: "⭐",
                value: "4.8/5",
                label: "Average Rating",
              },
              {
                icon: "💬",
                value: "2,400+",
                label: "Feedback Received",
              },
              {
                icon: "🚀",
                value: "18+",
                label: "Features Added",
              },
            ].map((item, index) => (
              <div
                key={index}
                className="info-stat-item"
              >
                <span className="stat-emoji">
                  {item.icon}
                </span>

                <div>
                  <div className="stat-value">
                    {item.value}
                  </div>

                  <div className="stat-label">
                    {item.label}
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Testimonial */}

          <div className="info-testimonial">
            <div className="quote-icon">❝</div>

            <p className="testimonial-text">
              {
                TESTIMONIALS[testimonialIndex]
                  .text
              }
            </p>

            <div className="testimonial-footer">
              <span className="testimonial-author">
                {
                  TESTIMONIALS[testimonialIndex]
                    .author
                }
              </span>

              <span className="testimonial-location">
                {
                  TESTIMONIALS[testimonialIndex]
                    .location
                }
              </span>
            </div>

            <div className="testimonial-dots">
              {TESTIMONIALS.map((_, i) => (
                <button
                  key={i}
                  className={`dot ${
                    i === testimonialIndex
                      ? "active"
                      : ""
                  }`}
                  onClick={() =>
                    setTestimonialIndex(i)
                  }
                />
              ))}
            </div>
          </div>
        </div>

        {/* RIGHT PANEL */}

        <div className="feedback-form-panel">
          <h2>Share Your Feedback</h2>

          {error && (
            <div className="fb-error">
              {error}
            </div>
          )}

          <form
            className="fb-form"
            onSubmit={handleSubmit}
          >
            {/* Rating */}

            <div className="fb-rating-section">
              <label>Overall Rating *</label>

              <div className="stars-row">
                {[1, 2, 3, 4, 5].map((star) => (
                  <button
                    key={star}
                    type="button"
                    className="star-btn"
                    onMouseEnter={() =>
                      setHoverRating(star)
                    }
                    onMouseLeave={() =>
                      setHoverRating(0)
                    }
                    onClick={() =>
                      handleChange(
                        "rating",
                        star
                      )
                    }
                  >
                    <Star
                      size={36}
                      fill={
                        (hoverRating ||
                          form.rating) >=
                        star
                          ? "#f59e0b"
                          : "none"
                      }
                      stroke={
                        (hoverRating ||
                          form.rating) >=
                        star
                          ? "#f59e0b"
                          : "#cbd5e1"
                      }
                    />
                  </button>
                ))}

                {form.rating > 0 && (
                  <span className="rating-label">
                    {
                      [
                        "",
                        "Poor",
                        "Fair",
                        "Good",
                        "Great",
                        "Excellent!",
                      ][form.rating]
                    }
                  </span>
                )}
              </div>
            </div>

            <div className="form-divider"></div>

            {/* Categories */}

            <div className="fb-group">
              <label>
                Feedback Category
              </label>

              <div className="category-grid">
                {CATEGORY_OPTIONS.map(
                  (cat) => (
                    <button
                      key={cat.value}
                      type="button"
                      className={`cat-chip ${
                        form.category ===
                        cat.value
                          ? "active"
                          : ""
                      }`}
                      onClick={() =>
                        handleChange(
                          "category",
                          cat.value
                        )
                      }
                    >
                      {cat.label}
                    </button>
                  )
                )}
              </div>
            </div>

            {/* Message */}

            <div className="fb-group">
              <label>
                <MessageSquare size={16} />
                Feedback / Suggestions *
              </label>

              <textarea
                rows="5"
                placeholder="Share your experience, feature request, or report an issue..."
                value={form.message}
                onChange={(e) =>
                  handleChange(
                    "message",
                    e.target.value
                  )
                }
              />
            </div>

            {/* Name + Location */}

            <div className="fb-row">
              <div className="fb-group">
                <label>
                  <User size={16} />
                  Your Name
                </label>

                <input
                  type="text"
                  placeholder="e.g. Ramesh Kumar"
                  value={form.name}
                  onChange={(e) =>
                    handleChange(
                      "name",
                      e.target.value
                    )
                  }
                />
              </div>

              <div className="fb-group">
                <label>
                  <MapPin size={16} />
                  Location
                </label>

                <input
                  type="text"
                  placeholder="e.g. Nashik, Maharashtra"
                  value={form.location}
                  onChange={(e) =>
                    handleChange(
                      "location",
                      e.target.value
                    )
                  }
                />
              </div>
            </div>

            {/* Crop */}

            <div className="fb-group">
              <label>
                <Sprout size={16} />
                Primary Crop
              </label>

              <select
                value={form.cropType}
                onChange={(e) =>
                  handleChange(
                    "cropType",
                    e.target.value
                  )
                }
              >
                <option value="">
                  Select your main crop
                </option>

                {CROP_OPTIONS.map((crop) => (
                  <option
                    key={crop}
                    value={crop}
                  >
                    {crop}
                  </option>
                ))}
              </select>
            </div>

            <div className="form-divider"></div>

            {/* Submit */}

            <button
              type="submit"
              className="fb-submit-btn"
              disabled={loading}
            >
              {loading ? (
                <>
                  <span className="fb-spinner"></span>
                  Submitting...
                </>
              ) : (
                <>
                  <Send size={18} />
                  Submit Feedback
                </>
              )}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}