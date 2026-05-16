import { useState } from "react";
import { toast } from "react-toastify";
import {
  FaEnvelope,
  FaPhone,
  FaMapMarkerAlt,
  FaFacebook,
  FaTwitter,
  FaInstagram,
  FaYoutube,
  FaPaperPlane,
  FaLeaf,
  FaHeadset,
  FaClock,
  FaCheckCircle,
  FaWhatsapp,
} from "react-icons/fa";
import "./ContactUs.css";

const FAQ_ITEMS = [
  {
    q: "How do I get crop recommendations?",
    a: "Navigate to the Advisor section and enter your soil type, location, and season. Our AI will suggest the best crops for you.",
  },
  {
    q: (
      <span>
        Is{" "}
        <span className="notranslate" translate="no">
          Fasal Saathi
        </span>{" "}
        available in my language?
      </span>
    ),
    a: "Yes! We support 12 Indian languages including Hindi, Bengali, Tamil, Telugu, Marathi, and more. Use the language selector in the navbar.",
  },
  {
    q: "How accurate are the weather forecasts?",
    a: "We use real-time data from trusted meteorological sources to provide forecasts accurate up to 7 days for your location.",
  },
  {
    q: (
      <span>
        Can I use{" "}
        <span className="notranslate" translate="no">
          Fasal Saathi
        </span>{" "}
        offline?
      </span>
    ),
    a: "Yes, basic features work offline. Your data syncs automatically when you reconnect to the internet.",
  },
  {
    q: "How do I report a bug or issue?",
    a: "Use this contact form with the subject 'Bug Report', or visit our Community page to post in the support thread. Our team responds within 24 hours.",
  },
];

const TOPICS = [
  "General Query",
  "Technical Support",
  "Crop Advice",
  "Bug Report",
  "Partnership",
  "Feedback",
];

const SOCIALS = [
  { icon: FaFacebook, label: "Facebook", color: "#1877f2" },
  { icon: FaTwitter, label: "Twitter", color: "#1da1f2" },
  { icon: FaInstagram, label: "Instagram", color: "#e4405f" },
  { icon: FaYoutube, label: "YouTube", color: "#ff0000" },
];

const INFO_CARDS = [
  {
    icon: FaPhone,
    title: "Call Us",
    lines: ["+91 99999 99999", "Mon – Sat, 9 AM – 6 PM"],
  },
  {
    icon: FaEnvelope,
    title: "Email Us",
    lines: ["contact@fasalsaathi.com", "support@fasalsaathi.com"],
  },
  {
    icon: FaMapMarkerAlt,
    title: "Visit Us",
    lines: ["Greater Noida, India", "201310"],
  },
];

export default function ContactUs() {
  const [form, setForm] = useState({
    name: "",
    email: "",
    phone: "",
    topic: "",
    message: "",
  });

  const [errors, setErrors] = useState({});
  const [loading, setLoading] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [openFaq, setOpenFaq] = useState(null);
  const [focused, setFocused] = useState(null);

  const validate = () => {
    const e = {};

    if (!form.name.trim()) e.name = "Name is required.";

    if (!form.email.trim()) {
      e.email = "Email is required.";
    } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(form.email)) {
      e.email = "Enter a valid email.";
    }

    if (form.phone && !/^\+?[\d\s\-]{7,15}$/.test(form.phone)) {
      e.phone = "Enter a valid phone number.";
    }

    if (!form.topic) e.topic = "Please select a topic.";

    if (!form.message.trim()) {
      e.message = "Message is required.";
    } else if (form.message.trim().length < 10) {
      e.message = "Message must be at least 10 characters.";
    }

    return e;
  };

  const handleChange = (e) => {
    setForm({
      ...form,
      [e.target.name]: e.target.value,
    });

    if (errors[e.target.name]) {
      setErrors({
        ...errors,
        [e.target.name]: "",
      });
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    const errs = validate();

    if (Object.keys(errs).length) {
      setErrors(errs);
      return;
    }

    setLoading(true);

    try {
      const response = await fetch("/api/contact", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(form),
      });

      if (!response.ok) {
        throw new Error("Server error");
      }

      setSubmitted(true);

      toast.success(
        "Message sent! We'll get back to you within 24 hours. 🌱"
      );
    } catch {
      toast.error("Failed to send message. Please try again.");
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
    });
  };

  const toggleFaq = (index) => {
    setOpenFaq(openFaq === index ? null : index);
  };

  return (
    <div className="contact-page">
      <div className="contact-blob contact-blob-1" />
      <div className="contact-blob contact-blob-2" />

      {/* ─── HERO ─── */}
      <div className="contact-hero">
        <div className="contact-hero-badge">
          <FaLeaf />
          Get In Touch
        </div>

        <h1>
          We're Here to
          <br />
          <span className="hero-highlight">Help You Grow</span>
        </h1>

        <p>
          Have a question, suggestion, or need support? The{" "}
          <span className="notranslate" translate="no">
            Fasal Saathi
          </span>{" "}
          team is just a message away.
        </p>

        <div className="hero-quick-links">
          <a
            href="mailto:contact@fasalsaathi.com"
            className="hero-quick-btn"
          >
            <FaEnvelope />
            Email Us
          </a>

          <a
            href="https://wa.me/919999999999"
            className="hero-quick-btn whatsapp"
            target="_blank"
            rel="noopener noreferrer"
          >
            <FaWhatsapp />
            WhatsApp
          </a>
        </div>
      </div>

      <div className="contact-container">
        {/* ─── CONTACT INFO CARDS ─── */}
        <div className="contact-info-grid">
          {INFO_CARDS.map((card) => (
            <div className="contact-info-card" key={card.title}>
              <div className="info-icon-wrap">
                <card.icon />
              </div>
              <div className="info-text">
                <span className="info-title">{card.title}</span>
                {card.lines.map((line, i) => (
                  <span className="info-line" key={i}>
                    {line}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* ─── FORM + SIDEBAR ─── */}
        <div className="contact-layout">
          <div className="contact-form-card">
            {submitted ? (
              <div className="success-view">
                <div className="success-icon">
                  <FaCheckCircle />
                </div>
                <h2>Message Sent!</h2>
                <p>
                  Thank you for reaching out. Our team will get back to you
                  within 24 hours.
                </p>
                <button
                  className="cf-submit-btn"
                  type="button"
                  onClick={handleReset}
                >
                  <FaPaperPlane />
                  Send Another Message
                </button>
              </div>
            ) : (
              <form className="contact-form" onSubmit={handleSubmit} noValidate>
                <div className="cf-row">
                  <div className="cf-group">
                    <label htmlFor="c-name">Full Name</label>
                    <input
                      id="c-name"
                      type="text"
                      name="name"
                      placeholder="John Doe"
                      value={form.name}
                      onChange={handleChange}
                      onFocus={() => setFocused("name")}
                      onBlur={() => setFocused(null)}
                      className={focused === "name" || form.name ? "has-value" : ""}
                      aria-invalid={!!errors.name}
                    />
                    {errors.name && <span className="cf-error">{errors.name}</span>}
                  </div>

                  <div className="cf-group">
                    <label htmlFor="c-email">Email Address</label>
                    <input
                      id="c-email"
                      type="email"
                      name="email"
                      placeholder="john@example.com"
                      value={form.email}
                      onChange={handleChange}
                      onFocus={() => setFocused("email")}
                      onBlur={() => setFocused(null)}
                      className={focused === "email" || form.email ? "has-value" : ""}
                      aria-invalid={!!errors.email}
                    />
                    {errors.email && <span className="cf-error">{errors.email}</span>}
                  </div>
                </div>

                <div className="cf-row">
                  <div className="cf-group">
                    <label htmlFor="c-phone">Phone Number <span className="cf-optional">(optional)</span></label>
                    <input
                      id="c-phone"
                      type="tel"
                      name="phone"
                      placeholder="+91 98765 43210"
                      value={form.phone}
                      onChange={handleChange}
                      onFocus={() => setFocused("phone")}
                      onBlur={() => setFocused(null)}
                      className={focused === "phone" || form.phone ? "has-value" : ""}
                      aria-invalid={!!errors.phone}
                    />
                    {errors.phone && <span className="cf-error">{errors.phone}</span>}
                  </div>

                  <div className="cf-group">
                    <label htmlFor="c-topic">Topic</label>
                    <select
                      id="c-topic"
                      name="topic"
                      value={form.topic}
                      onChange={handleChange}
                      onFocus={() => setFocused("topic")}
                      onBlur={() => setFocused(null)}
                      className={`cf-select ${focused === "topic" || form.topic ? "has-value" : ""}`}
                      aria-invalid={!!errors.topic}
                    >
                      <option value="">Select a topic…</option>
                      {TOPICS.map((t) => (
                        <option key={t} value={t}>
                          {t}
                        </option>
                      ))}
                    </select>
                    {errors.topic && <span className="cf-error">{errors.topic}</span>}
                  </div>
                </div>

                <div className="cf-group">
                  <label htmlFor="c-message">Your Message</label>
                  <textarea
                    id="c-message"
                    name="message"
                    rows="5"
                    placeholder="Tell us how we can help…"
                    value={form.message}
                    onChange={handleChange}
                    onFocus={() => setFocused("message")}
                    onBlur={() => setFocused(null)}
                    className={focused === "message" || form.message ? "has-value" : ""}
                    aria-invalid={!!errors.message}
                  />
                  {errors.message && (
                    <span className="cf-error">{errors.message}</span>
                  )}
                </div>

                <button
                  className="cf-submit-btn"
                  type="submit"
                  disabled={loading}
                >
                  {loading ? (
                    <>
                      <span className="cf-spinner" />
                      Sending…
                    </>
                  ) : (
                    <>
                      <FaPaperPlane />
                      Send Message
                    </>
                  )}
                </button>
              </form>
            )}
          </div>

          {/* ─── SIDEBAR ─── */}
          <div className="contact-sidebar">
            {/* Response time card */}
            <div className="contact-response-card">
              <FaHeadset className="resp-icon" />
              <h3>Quick Response</h3>
              <p>
                We aim to respond within <strong>24 hours</strong> on business
                days. For urgent queries, reach out on WhatsApp.
              </p>
              <div className="resp-hours">
                <FaClock />
                <span>Mon – Sat · 9 AM – 6 PM IST</span>
              </div>
            </div>

            {/* Social card */}
            <div className="contact-social-card">
              <h3>Follow Us</h3>
              <p>Stay connected for tips and updates</p>
              <div className="social-links">
                {SOCIALS.map(({ icon: Icon, label, color }) => (
                  <a
                    key={label}
                    href="#"
                    className="social-link"
                    aria-label={label}
                    title={label}
                  >
                    <Icon />
                    <span>{label}</span>
                  </a>
                ))}
              </div>
            </div>

            {/* WhatsApp CTA */}
            <a
              href="https://wa.me/919999999999"
              className="whatsapp-cta-card"
              target="_blank"
              rel="noopener noreferrer"
            >
              <FaWhatsapp />
              <div>
                <strong>Chat on WhatsApp</strong>
                <span>Average reply &lt; 1 hour</span>
              </div>
            </a>
          </div>
        </div>
      </div>
    </div>
  );
}
