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
  Quote,
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
  { value: "general", label: "💬 General" },
  { value: "feature", label: "✨ Feature" },
  { value: "bug", label: "🐛 Bug" },
  { value: "ui", label: "🎨 UI/UX" },
  { value: "accuracy", label: "🎯 Accuracy" },
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
      setError("Please select a rating.");
      return;
    }

    if (!isFirebaseConfigured()) {
      setError(
        "Firebase is not configured. Please check your .env file."
      );
      return;
    }

    setLoading(true);

    try {
      const user = auth?.currentUser;
      
      // Prepare data for API submission
      const feedbackData = {
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
      };

      // Submit to secure backend API instead of direct Firestore write
      const response = await fetch("/api/feedback", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(feedbackData),
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.detail || result.error || "Failed to submit feedback");
      }

      if (!result.success) {
        throw new Error(result.message || "Feedback submission failed");
      }

      console.log("Feedback submitted successfully. ID:", result.feedback_id);
      setSubmitted(true);
    } catch (err) {
      console.error("Feedback submit error:", err);
      setError(
        "Failed to submit feedback. Please try again."
      );
      
      // User-friendly error messages
      let errorMessage = "Failed to submit feedback. Please try again.";
      
      if (err.message.includes("Message is required")) {
        errorMessage = "Please enter a valid feedback message.";
      } else if (err.message.includes("Invalid data format")) {
        errorMessage = "Your feedback contains invalid characters. Please remove any special symbols and try again.";
      } else if (err.message.includes("rating")) {
        errorMessage = "Please select a valid rating between 1 and 5 stars.";
      } else {
        errorMessage = err.message || "Failed to submit feedback. Please try again.";
      }
      
      setError(errorMessage);
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
          <div className="success-ring">
            <CheckCircle2 size={70} />
          </div>

          <h2>Feedback Submitted 🎉</h2>

          <p>
            Thank you for helping improve{" "}
            <span className="brand-name">
              Fasal Saathi
            </span>
            .
          </p>

          <div className="submitted-stars">
            {[1, 2, 3, 4, 5].map((s) => (
              <Star
                key={s}
                size={28}
                fill={
                  s <= form.rating
                    ? "#f59e0b"
                    : "none"
                }
                stroke={
                  s <= form.rating
                    ? "#f59e0b"
                    : "#d1d5db"
                }
              />
            ))}
          </div>

          <button
            className="submit-btn"
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
      <div className="feedback-container">

        {/* LEFT PANEL */}
        <div className="left-panel">

          <div className="badge">
            🌾 Farmer Feedback
          </div>

          <h1>
            Help Us Build a Better Experience
          </h1>

          <p className="subtitle">
            Your feedback helps improve{" "}
            <span className="brand-name">
              Fasal Saathi
            </span>{" "}
            for farmers across India.
          </p>

          {/* Farmer Images */}
          <div className="farmer-showcase">
            <img
              src="/farmer1.png"
              alt="Farmer"
              className="farmer-img img1"
            />

            <img
              src="/farmer2.png"
              alt="Farmer"
              className="farmer-img img2"
            />
          </div>

          {/* Stats */}
          <div className="stats-grid">

            <div className="stat-card">
              <span>⭐</span>
              <div>
                <h3>4.8/5</h3>
                <p>Average Rating</p>
              </div>
            </div>

            <div className="stat-card">
              <span>💬</span>
              <div>
                <h3>2,400+</h3>
                <p>Feedbacks</p>
              </div>
            </div>

            <div className="stat-card">
              <span>🚀</span>
              <div>
                <h3>18+</h3>
                <p>Features Added</p>
              </div>
            </div>

          </div>

          {/* Testimonial */}
          <div className="testimonial-card">

            <Quote className="quote-icon" size={28} />

            <p className="testimonial-text">
              {
                TESTIMONIALS[testimonialIndex]
                  .text
              }
            </p>

            <div className="testimonial-user">
              <strong>
                {
                  TESTIMONIALS[testimonialIndex]
                    .author
                }
              </strong>
              <span>
                {
                  TESTIMONIALS[testimonialIndex]
                    .location
                }
              </span>
            </div>

            <div className="dots">
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
        <div className="right-panel">

          <div className="form-header">
            <h2>Share Your Feedback</h2>
            <p>
              We read every feedback carefully.
            </p>
          </div>

          {error && (
            <div className="error-box">
              {error}
            </div>
          )}

          <form
            className="feedback-form"
            onSubmit={handleSubmit}
          >

            {/* Rating */}
            <div className="form-group">
              <label>
                Overall Rating *
              </label>

              <div className="rating-row">

                {[1, 2, 3, 4, 5].map((star) => (
                  <button
                    type="button"
                    key={star}
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
                      size={34}
                      fill={
                        (
                          hoverRating ||
                          form.rating
                        ) >= star
                          ? "#f59e0b"
                          : "none"
                      }
                      stroke={
                        (
                          hoverRating ||
                          form.rating
                        ) >= star
                          ? "#f59e0b"
                          : "#cbd5e1"
                      }
                    />
                  </button>
                ))}

                {form.rating > 0 && (
                  <span className="rating-text">
                    {
                      [
                        "",
                        "Poor",
                        "Fair",
                        "Good",
                        "Great",
                        "Excellent",
                      ][form.rating]
                    }
                  </span>
                )}
              </div>
            </div>

            {/* Categories */}
            <div className="form-group">
              <label>
                Feedback Category
              </label>

              <div className="category-grid">
                {CATEGORY_OPTIONS.map((cat) => (
                  <button
                    type="button"
                    key={cat.value}
                    className={`category-btn ${
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
                ))}
              </div>
            </div>

            {/* Message */}
            <div className="form-group">
              <label>
                <MessageSquare size={16} />
                Feedback *
              </label>

              <textarea
                rows="5"
                placeholder="Tell us your experience..."
                value={form.message}
                onChange={(e) =>
                  handleChange(
                    "message",
                    e.target.value
                  )
                }
              />
            </div>

            {/* Row */}
            <div className="input-row">

              <div className="form-group">
                <label>
                  <User size={16} />
                  Name
                </label>

                <input
                  type="text"
                  placeholder="Your name"
                  value={form.name}
                  onChange={(e) =>
                    handleChange(
                      "name",
                      e.target.value
                    )
                  }
                />
              </div>

              <div className="form-group">
                <label>
                  <MapPin size={16} />
                  Location
                </label>

                <input
                  type="text"
                  placeholder="City, State"
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
            <div className="form-group">
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
                  Select Crop
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

            {/* Submit */}
            <button
              type="submit"
              className="submit-btn"
              disabled={loading}
            >
              {loading ? (
                <>
                  <span className="spinner"></span>
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