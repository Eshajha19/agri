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
  MessageCircle,
  Sparkles,
  Bug,
  Palette,
  Target,
  Pin,
  Wheat,
  Rocket,
} from "lucide-react";
import "./Feedback.css";

const CROP_OPTIONS = [
  "Rice", "Wheat", "Cotton", "Sugarcane", "Maize",
  "Soybean", "Potato", "Onion", "Tomato", "Vegetables",
  "Fruits", "Other",
];

const CATEGORY_OPTIONS = [
  { value: "general", label: "General Feedback", icon: <MessageCircle size={14} /> },
  { value: "feature", label: "Feature Request", icon: <Sparkles size={14} /> },
  { value: "bug", label: "Report a Bug", icon: <Bug size={14} /> },
  { value: "ui", label: "UI/UX Improvement", icon: <Palette size={14} /> },
  { value: "accuracy", label: "AI Accuracy", icon: <Target size={14} /> },
  { value: "other", label: "Other", icon: <Pin size={14} /> },
];

// ✅ Testimonials Data
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

  // ✅ Testimonial state
  const [testimonialIndex, setTestimonialIndex] = useState(0);

  // ✅ Auto-rotate testimonials
  useEffect(() => {
    const interval = setInterval(() => {
      setTestimonialIndex((prev) => (prev + 1) % TESTIMONIALS.length);
    }, 4000);

    return () => clearInterval(interval);
  }, []);

  const handleChange = (field, value) => {
    setForm((prev) => ({ ...prev, [field]: value }));
  };

   const handleSubmit = async (e) => {
     e.preventDefault();
     setError("");

     // Basic client-side validation
     if (!form.message.trim()) {
       setError("Please enter your feedback message.");
       return;
     }

     if (form.message.trim().length < 3) {
       setError("Feedback message must be at least 3 characters long.");
       return;
     }

     if (form.message.length > 2000) {
       setError("Feedback message is too long (maximum 2000 characters).");
       return;
     }

     if (form.rating === 0) {
       setError("Please select a rating.");
       return;
     }

     if (form.rating < 1 || form.rating > 5) {
       setError("Please select a valid rating between 1 and 5 stars.");
       return;
     }

     // Validate name length
     if (form.name && form.name.length > 100) {
       setError("Name is too long (maximum 100 characters).");
       return;
     }

     // Validate location length
     if (form.location && form.location.length > 200) {
       setError("Location is too long (maximum 200 characters).");
       return;
     }

     // Validate category
     const validCategories = ['general', 'feature', 'bug', 'ui', 'accuracy', 'other'];
     if (!validCategories.includes(form.category)) {
       setError("Invalid feedback category selected.");
       return;
     }

     // Validate crop type
     const validCrops = [
       'Rice', 'Wheat', 'Cotton', 'Sugarcane', 'Maize',
       'Soybean', 'Potato', 'Onion', 'Tomato', 'Vegetables',
       'Fruits', 'Other'
     ];
     if (form.cropType && !validCrops.includes(form.cropType)) {
       setError("Invalid crop type selected.");
       return;
     }

     // Security validation - check for dangerous patterns
     const dangerousPatterns = [
       /\$[a-zA-Z_][a-zA-Z0-9_]*\s*:/, // MongoDB operators
       /\{.*\}\s*:\s*\{/, // Nested object injection
       /\.\.\//, // Path traversal
       /<script.*?>.*?<\/script>/i, // XSS
       /on\w+\s*=/i, // Event handlers
       /javascript:/i, // JavaScript protocol
       /data:/i, // Data URLs
     ];

     const allFields = [form.message, form.name, form.location].filter(Boolean);
     for (const field of allFields) {
       for (const pattern of dangerousPatterns) {
         if (pattern.test(field)) {
           setError('Your input contains potentially unsafe characters. Please remove any special symbols and try again.');
           return;
         }
       }
     }

     setLoading(true);
     try {
       const user = auth?.currentUser;
       
       // Prepare data for Firestore submission
       const feedbackData = {
         userId: user?.uid || "anonymous",
         userEmail: user?.email || "anonymous",
         name: form.name || (user?.displayName ?? "Anonymous"),
         cropType: form.cropType || null,
         location: form.location || null,
         category: form.category,
         message: form.message.trim(),
         rating: form.rating,
         createdAt: new Date().toISOString(),
       };

       // Remove null values to keep Firestore clean
       Object.keys(feedbackData).forEach(key => 
         feedbackData[key] === null && delete feedbackData[key]
       );

       // Submit directly to Firestore
       const docRef = await addDoc(collection(db, "feedback"), feedbackData);
       
       console.log("Feedback submitted successfully. ID:", docRef.id);
       setSubmitted(true);
     } catch (err) {
       console.error("Feedback submit error:", err);
       
       // User-friendly error messages
       let errorMessage = "Failed to submit feedback. Please try again.";
       
       if (err.message) {
         const message = err.message.toLowerCase();
         if (message.includes("message is required") || message.includes("at least 3 characters")) {
           errorMessage = "Please enter a valid feedback message.";
         } else if (message.includes("too long") || message.includes("maximum")) {
           errorMessage = "Your feedback is too long. Please shorten it and try again.";
         } else if (message.includes("rating")) {
           errorMessage = "Please select a valid rating between 1 and 5 stars.";
         } else if (message.includes("unsafe") || message.contains("special symbols")) {
           errorMessage = "Your input contains potentially unsafe characters. Please remove any special symbols and try again.";
         } else if (message.includes("permission") || message.includes("denied")) {
           errorMessage = "You don't have permission to submit feedback. Please try again later.";
         } else {
           errorMessage = err.message || "Failed to submit feedback. Please try again.";
         }
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
        <div className="feedback-success-card">
          <div className="success-icon-ring">
            <CheckCircle2 size={64} className="success-icon" />
          </div>

          <h2>Thank You! 🙏</h2>

          <p>
            Your feedback has been submitted successfully. We'll use it to make{" "}
            <span className="notranslate" translate="no">
              Fasal Saathi
            </span>{" "}
            even better for farmers like you.
          </p>

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
          <div className="info-badge"><Wheat size={14} aria-hidden="true" /> Farmer Feedback</div>

          <h1>Help Us Grow Better</h1>

          <p>
            Your opinion directly shapes the future of{" "}
            <span className="notranslate" translate="no">
              Fasal Saathi
            </span>
            . Share your experience, suggest features, or report issues — every
            word matters.
          </p>

          {/* Farmer Showcase */}
          <div className="farmer-showcase">
            <div className="farmer-images">
              <img src="/farmer1.png" alt="Farmer 1" className="farmer-img img-1" />
              <img src="/farmer2.png" alt="Farmer 2" className="farmer-img img-2" />
            </div>
          </div>

          <div className="info-stats">
            {[
              { icon: <Star size={18} className="text-yellow-400" />, label: "Average Rating", value: "4.8/5" },
              { icon: <MessageSquare size={18} />, label: "Feedbacks Received", value: "2,400+" },
              { icon: <Rocket size={18} />, label: "Features Added from Feedback", value: "18+" },
            ].map((stat, i) => (
              <div key={i} className="info-stat-item">
                <span className="stat-emoji" aria-hidden="true">{stat.icon}</span>
                <div>
                  <div className="stat-value">{stat.value}</div>
                  <div className="stat-label">{stat.label}</div>
                </div>
              </div>
            ))}
          </div>

          {/* ✅ Improved Testimonials */}
          <div className="info-testimonial">
            <p className="testimonial-text">
              {TESTIMONIALS[testimonialIndex].text}
            </p>

            <span className="testimonial-author">
              — {TESTIMONIALS[testimonialIndex].author},{" "}
              {TESTIMONIALS[testimonialIndex].location}
            </span>

            <div className="testimonial-dots">
              {TESTIMONIALS.map((_, i) => (
                <span
                  key={i}
                  className={`dot ${i === testimonialIndex ? "active" : ""}`}
                  onClick={() => setTestimonialIndex(i)}
                />
              ))}
            </div>
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
                    <>
                      <span aria-hidden="true">{cat.icon}</span>
                      {cat.label}
                    </>
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