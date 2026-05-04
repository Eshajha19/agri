import { useState, useEffect } from "react";
import { toast } from "react-toastify";
import {
  FaEnvelope, FaPhone, FaMapMarkerAlt,
  FaFacebook, FaTwitter, FaInstagram,
  FaYoutube, FaPaperPlane, FaLeaf,
  FaHeadset, FaCheckCircle, FaWhatsapp
} from "react-icons/fa";
import "./ContactUs.css";

const TOPICS = [
  "General Query",
  "Technical Support",
  "Crop Advice",
  "Bug Report",
  "Partnership",
  "Feedback"
];

const topicTemplates = {
  "Bug Report": "Describe the issue, steps to reproduce, and expected behavior...",
  "Crop Advice": "Mention your crop, soil type, and location..."
};

export default function ContactUs() {
  const [form, setForm] = useState({
    name: "",
    email: "",
    phone: "",
    topic: "",
    message: "",
    company: ""
  });

  const [errors, setErrors] = useState({});
  const [loading, setLoading] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [progress, setProgress] = useState(0);

  /* ---------------- AUTO SAVE ---------------- */
  useEffect(() => {
    const saved = localStorage.getItem("contactForm");
    if (saved) setForm(JSON.parse(saved));
  }, []);

  useEffect(() => {
    localStorage.setItem("contactForm", JSON.stringify(form));
  }, [form]);

  /* ---------------- VALIDATION ---------------- */
  const validate = () => {
    const e = {};
    if (!form.name.trim()) e.name = "Name is required.";
    if (!form.email.trim()) e.email = "Email is required.";
    else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(form.email))
      e.email = "Invalid email.";
    if (!form.topic) e.topic = "Select a topic.";
    if (!form.message.trim()) e.message = "Message required.";
    else if (form.message.length < 10)
      e.message = "Minimum 10 characters.";
    return e;
  };

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
    setErrors({ ...errors, [e.target.name]: "" });
  };

  /* ---------------- SUBMIT ---------------- */
  const handleSubmit = async (e) => {
    e.preventDefault();

    if (form.company) return; // spam

    const errs = validate();
    if (Object.keys(errs).length) {
      setErrors(errs);
      return;
    }

    setLoading(true);
    setProgress(0);

    const interval = setInterval(() => {
      setProgress((p) => (p < 90 ? p + 10 : p));
    }, 200);

    try {
      await new Promise((res) => setTimeout(res, 1500));

      clearInterval(interval);
      setProgress(100);
      setSubmitted(true);
      localStorage.removeItem("contactForm");

      toast.success("Message sent 🌱");
    } catch {
      toast.error("Failed to send.");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSubmitted(false);
    setForm({
      name: "",
      email: "",
      phone: "",
      topic: "",
      message: "",
      company: ""
    });
  };

  /* ---------------- UI ---------------- */
  return (
    <div className="contact-page">

      {/* HERO */}
      <div className="contact-hero">
        <div className="contact-hero-badge">
          <FaLeaf /> Get In Touch
        </div>
        <h1>We're Here to Help You Grow</h1>
        <p>Reach out anytime 🌱</p>
      </div>

      <div className="contact-container">
        <div className="contact-main-grid">

          {/* FORM */}
          <div className="contact-form-card">

            {submitted ? (
              <div className="cf-success">
                <FaCheckCircle className="cf-success-icon" />
                <h2>Message Delivered 🚀</h2>
                <p>We’ll reply within 24 hours.</p>
                <button onClick={handleReset}>
                  Send Another Message
                </button>
              </div>
            ) : (
              <>
                <div className="form-card-header">
                  <FaHeadset />
                  <h2>Send a Message</h2>
                </div>

                {/* TOPICS */}
                <div className="cf-topic-chips">
                  {TOPICS.map((t) => (
                    <button
                      key={t}
                      type="button"
                      className={`cf-chip ${form.topic === t ? "active" : ""}`}
                      onClick={() =>
                        setForm({
                          ...form,
                          topic: t,
                          message: topicTemplates[t] || form.message
                        })
                      }
                    >
                      {t}
                    </button>
                  ))}
                </div>
                {errors.topic && <span className="cf-error">{errors.topic}</span>}

                <form onSubmit={handleSubmit} noValidate>

                  {/* Honeypot */}
                  <input
                    type="text"
                    name="company"
                    style={{ display: "none" }}
                    onChange={handleChange}
                  />

                  {/* NAME */}
                  <div className="cf-group">
                    <label>Name</label>
                    <input
                      name="name"
                      value={form.name}
                      onChange={handleChange}
                      placeholder=" "
                    />
                    {errors.name && <span className="cf-error">{errors.name}</span>}
                  </div>

                  {/* EMAIL */}
                  <div className="cf-group">
                    <label>Email</label>
                    <input
                      name="email"
                      value={form.email}
                      onChange={handleChange}
                      placeholder=" "
                    />
                    {errors.email && <span className="cf-error">{errors.email}</span>}
                  </div>

                  {/* PHONE */}
                  <div className="cf-group">
                    <label>Phone (optional)</label>
                    <input
                      name="phone"
                      value={form.phone}
                      onChange={handleChange}
                      placeholder=" "
                    />
                  </div>

                  {/* MESSAGE */}
                  <div className="cf-group">
                    <label>Message</label>
                    <textarea
                      name="message"
                      value={form.message}
                      onChange={handleChange}
                      maxLength={500}
                      placeholder=" "
                    />
                    <div className="cf-char-count">
                      {form.message.length}/500
                    </div>
                    {errors.message && <span className="cf-error">{errors.message}</span>}
                  </div>

                  {/* PROGRESS */}
                  {loading && (
                    <div className="cf-progress">
                      <div style={{ width: `${progress}%` }} />
                    </div>
                  )}

                  <button
                    type="submit"
                    className="cf-submit-btn"
                    disabled={
                      loading ||
                      !form.name ||
                      !form.email ||
                      !form.topic ||
                      !form.message
                    }
                  >
                    {loading ? "Sending..." : <><FaPaperPlane /> Send Message</>}
                  </button>

                </form>
              </>
            )}
          </div>

          {/* SIDE PANEL */}
          <div className="contact-side-panel">
            <div className="contact-info-card"><FaEnvelope /> hello@fasalsaathi.demo</div>
            <div className="contact-info-card"><FaPhone /> +91 XXXXX XXXXX</div>
            <div className="contact-info-card"><FaMapMarkerAlt /> India</div>

            <div className="social-links">
              <FaFacebook />
              <FaTwitter />
              <FaInstagram />
              <FaYoutube />
              <FaWhatsapp />
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}