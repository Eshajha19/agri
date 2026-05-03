import { useState, useEffect } from "react";
import { toast } from "react-toastify";
import {
  FaEnvelope, FaPhone, FaCheckCircle, FaPaperPlane,
  FaHeadset, FaLeaf, FaUpload
} from "react-icons/fa";
import "./ContactUs.css";

const TOPICS = [
  { name: "General Query", icon: "💬" },
  { name: "Technical Support", icon: "🛠️" },
  { name: "Crop Advice", icon: "🌾" },
  { name: "Bug Report", icon: "🐞" },
  { name: "Partnership", icon: "🤝" },
  { name: "Feedback", icon: "⭐" },
];

export default function ContactUs() {
  const [form, setForm] = useState({
    name: "",
    email: "",
    phone: "",
    topic: "",
    message: "",
    file: null,
    website: "" // honeypot anti-spam
  });

  const [errors, setErrors] = useState({});
  const [focused, setFocused] = useState(null);
  const [loading, setLoading] = useState(false);
  const [submitted, setSubmitted] = useState(false);

  /* ---------------- VALIDATION ---------------- */
  const validate = () => {
    const e = {};

    if (!form.name.trim()) e.name = "Name is required";

    if (!form.email.trim()) e.email = "Email required";
    else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(form.email))
      e.email = "Invalid email";

    if (!form.topic) e.topic = "Select a topic";

    if (!form.message.trim()) e.message = "Message required";
    else if (form.message.length < 10)
      e.message = "Minimum 10 characters";

    if (form.phone && !/^\+?[\d\s-]{7,15}$/.test(form.phone))
      e.phone = "Invalid phone";

    return e;
  };

  /* ---------------- REAL-TIME VALIDATION ---------------- */
  useEffect(() => {
    setErrors(validate());
  }, [form]);

  /* ---------------- SMART TOPIC DETECTION ---------------- */
  useEffect(() => {
    const msg = form.message.toLowerCase();

    if (msg.includes("bug")) setForm(p => ({ ...p, topic: "Bug Report" }));
    else if (msg.includes("crop")) setForm(p => ({ ...p, topic: "Crop Advice" }));
    else if (msg.includes("support")) setForm(p => ({ ...p, topic: "Technical Support" }));
  }, [form.message]);

  /* ---------------- UPDATE FIELD ---------------- */
  const updateField = (key, value) => {
    setForm(prev => ({ ...prev, [key]: value }));
  };

  /* ---------------- COMPLETION SCORE ---------------- */
  const completion =
    (form.name ? 25 : 0) +
    (form.email ? 25 : 0) +
    (form.topic ? 25 : 0) +
    (form.message.length > 10 ? 25 : 0);

  /* ---------------- SUBMIT ---------------- */
  const handleSubmit = async (e) => {
    e.preventDefault();

    if (form.website) return; // spam protection

    const errs = validate();
    if (Object.keys(errs).length) return setErrors(errs);

    setLoading(true);

    try {
      const data = new FormData();
      Object.entries(form).forEach(([k, v]) => {
        if (v) data.append(k, v);
      });

      await fetch("/api/contact", {
        method: "POST",
        body: data,
      });

      await new Promise(res => setTimeout(res, 800)); // smooth UX

      setSubmitted(true);
      toast.success("Message sent successfully 🌱");
    } catch {
      toast.error("Failed to send message");
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setSubmitted(false);
    setForm({
      name: "",
      email: "",
      phone: "",
      topic: "",
      message: "",
      file: null,
      website: ""
    });
  };

  /* ---------------- UI ---------------- */
  return (
    <div className="contact-page">

      {/* HERO */}
      <div className="contact-hero">
        <FaLeaf className="hero-icon" />
        <h1>Contact Us</h1>
        <p>We’re here to help you grow 🌱</p>
      </div>

      <div className="contact-container">

        {/* SUCCESS */}
        {submitted ? (
          <div className="success-card">
            <FaCheckCircle className="success-icon" />
            <h2>Message Sent!</h2>
            <p>We’ll respond within 24 hours.</p>
            <button onClick={reset}>Send Another</button>
          </div>
        ) : (

          <div className="contact-card">

            {/* HEADER */}
            <div className="form-header">
              <FaHeadset />
              <div>
                <h2>Send Message</h2>
                <p>Quick & easy support</p>
              </div>
            </div>

            {/* TOPIC CHIPS */}
            <div className="topic-section">
              <label>Topic *</label>
              <div className="topic-chips">
                {TOPICS.map(t => (
                  <button
                    key={t.name}
                    type="button"
                    className={form.topic === t.name ? "active" : ""}
                    onClick={() => updateField("topic", t.name)}
                  >
                    {t.icon} {t.name}
                  </button>
                ))}
              </div>
              {errors.topic && <span className="error">{errors.topic}</span>}
            </div>

            {/* FORM */}
            <form onSubmit={handleSubmit} noValidate>

              {/* HONEYPOT */}
              <input
                type="text"
                value={form.website}
                onChange={(e) => updateField("website", e.target.value)}
                style={{ display: "none" }}
              />

              {/* NAME */}
              <div className={`floating ${focused === "name" ? "focus" : ""}`}>
                <input
                  type="text"
                  value={form.name}
                  onChange={(e) => updateField("name", e.target.value)}
                  onFocus={() => setFocused("name")}
                  onBlur={() => setFocused(null)}
                  required
                />
                <label>Name *</label>
                {errors.name && <span className="error">{errors.name}</span>}
              </div>

              {/* EMAIL */}
              <div className={`floating ${focused === "email" ? "focus" : ""}`}>
                <input
                  type="email"
                  value={form.email}
                  onChange={(e) => updateField("email", e.target.value)}
                  onFocus={() => setFocused("email")}
                  onBlur={() => setFocused(null)}
                  required
                />
                <label>Email *</label>
                {errors.email && <span className="error">{errors.email}</span>}
              </div>

              {/* PHONE */}
              <div className={`floating ${focused === "phone" ? "focus" : ""}`}>
                <input
                  type="tel"
                  value={form.phone}
                  onChange={(e) => updateField("phone", e.target.value)}
                  onFocus={() => setFocused("phone")}
                  onBlur={() => setFocused(null)}
                />
                <label>Phone (optional)</label>
                {errors.phone && <span className="error">{errors.phone}</span>}
              </div>

              {/* MESSAGE */}
              <div className={`floating ${focused === "message" ? "focus" : ""}`}>
                <textarea
                  rows="4"
                  value={form.message}
                  maxLength={500}
                  onChange={(e) => updateField("message", e.target.value)}
                  onFocus={() => setFocused("message")}
                  onBlur={() => setFocused(null)}
                />
                <label>Message *</label>
                <div className="char-count">{form.message.length}/500</div>
                {errors.message && <span className="error">{errors.message}</span>}
              </div>

              {/* FILE UPLOAD */}
              <div className="file-upload">
                <label><FaUpload /> Attach Screenshot</label>
                <input
                  type="file"
                  accept="image/*"
                  onChange={(e) => updateField("file", e.target.files[0])}
                />
              </div>

              {/* PROGRESS */}
              <div className="progress-bar">
                <div style={{ width: `${completion}%` }} />
              </div>
              <span className="progress-text">
                Form Strength: {completion}%
              </span>

              {/* SUBMIT */}
              <button className="submit-btn" disabled={loading}>
                {loading ? "Sending..." : (
                  <>
                    <FaPaperPlane /> Send Message
                  </>
                )}
              </button>

            </form>
          </div>
        )}
      </div>
    </div>
  );
}