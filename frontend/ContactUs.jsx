import { useState } from "react";
import { toast } from "react-toastify";
import {
  FaEnvelope, FaPhone, FaMapMarkerAlt, FaFacebook,
  FaTwitter, FaInstagram, FaYoutube, FaPaperPlane,
  FaLeaf, FaHeadset, FaClock, FaCheckCircle, FaWhatsapp,
} from "react-icons/fa";
import "./ContactUs.css";

const FAQ_ITEMS = [
  { q: "How do I get crop recommendations?", a: "Navigate to the Advisor section and enter your soil type, location, and season. Our AI will suggest the best crops for you." },
  { q: <span>Is <span className="notranslate" translate="no">Fasal Saathi</span> available in my language?</span>, a: "Yes! We support 12 Indian languages including Hindi, Bengali, Tamil, Telugu, Marathi, and more. Use the language selector in the navbar." },
  { q: "How accurate are the weather forecasts?", a: "We use real-time data from trusted meteorological sources to provide forecasts accurate up to 7 days for your location." },
  { q: <span>Can I use <span className="notranslate" translate="no">Fasal Saathi</span> offline?</span>, a: "Yes, basic features work offline. Your data syncs automatically when you reconnect to the internet." },
  { q: "How do I report a bug or issue?", a: "Use this contact form with the subject 'Bug Report', or visit our Community page to post in the support thread. Our team responds within 24 hours." },
];

const TOPICS = ["General Query", "Technical Support", "Crop Advice", "Bug Report", "Partnership", "Feedback"];

export default function ContactUs() {
  const [form, setForm] = useState({ name: "", email: "", phone: "", topic: "", message: "" });
  const [errors, setErrors] = useState({});
  const [loading, setLoading] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [openFaq, setOpenFaq] = useState(null);
  const [focused, setFocused] = useState(null);

  const validate = () => {
    const e = {};
    if (!form.name.trim()) e.name = "Name is required.";
    if (!form.email.trim()) e.email = "Email is required.";
    else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(form.email)) e.email = "Enter a valid email.";
    if (form.phone && !/^\+?[\d\s\-]{7,15}$/.test(form.phone)) e.phone = "Enter a valid phone number.";
    if (!form.topic) e.topic = "Please select a topic.";
    if (!form.message.trim()) e.message = "Message is required.";
    else if (form.message.trim().length < 10) e.message = "Message must be at least 10 characters.";
    return e;
  };

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
    if (errors[e.target.name]) setErrors({ ...errors, [e.target.name]: "" });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const errs = validate();
    if (Object.keys(errs).length) { setErrors(errs); return; }
    setLoading(true);
    try {
      const response = await fetch("/api/contact", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      if (!response.ok) throw new Error("Server error");
      setSubmitted(true);
      toast.success("Message sent! We'll get back to you within 24 hours. 🌱");
    } catch {
      toast.error("Failed to send message. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSubmitted(false);
    setForm({ name: "", email: "", phone: "", topic: "", message: "" });
  };

  return (
    <div className="contact-page">
      {/* Floating blobs */}
      <div className="contact-blob contact-blob-1" />
      <div className="contact-blob contact-blob-2" />

      {/* Hero */}
      <div className="contact-hero">
        <div className="contact-hero-badge"><FaLeaf /> Get In Touch</div>
        <h1>We're Here to<br /><span className="hero-highlight">Help You Grow</span></h1>
         <p>Have a question, suggestion, or need support? The <span className="notranslate" translate="no">Fasal Saathi</span> team is just a message away.</p>
        {/* TODO: Replace placeholder contact links below with real values before shipping.
            Update CONTACT_EMAIL, CONTACT_PHONE, and SOCIAL_LINKS with actual
            project details once available. */}
        <div className="hero-quick-links">
          <a href="mailto:contact@fasalsaathi.com" className="hero-quick-btn"><FaEnvelope /> Email Us</a>
          <a href="https://wa.me/919999999999" className="hero-quick-btn whatsapp" target="_blank" rel="noopener noreferrer"><FaWhatsapp /> WhatsApp</a>
        </div>
      </div>

      <div className="contact-container">
        {/* Info Cards */}
        <div className="contact-info-grid">
          {[
            { icon: <FaEnvelope />, title: "Email Us", value: "hello@fasalsaathi.demo", sub: "We reply within 24 hours", color: "#10b981" },
            { icon: <FaPhone />, title: "Call Us", value: "+91 00000 00000", sub: "Mon–Sat, 9 AM – 6 PM", color: "#6366f1" },
            { icon: <FaMapMarkerAlt />, title: "Our Office", value: "123 AgriTech Park, Delhi", sub: "India – 110001", color: "#f59e0b" },
            { icon: <FaClock />, title: "Support Hours", value: "9 AM – 6 PM", sub: "Monday to Saturday", color: "#ec4899" },
          ].map((item, i) => (
            <div className="contact-info-card" key={i} style={{ "--card-accent": item.color }}>
              <div className="info-card-icon" style={{ background: `${item.color}18`, color: item.color }}>{item.icon}</div>
              <div>
                <div className="info-card-title">{item.title}</div>
                <div className="info-card-value">{item.value}</div>
                <div className="info-card-sub">{item.sub}</div>
              </div>
              <div className="info-card-glow" />
            </div>
          ))}
        </div>

        {/* Main Grid */}
        <div className="contact-main-grid">
          {/* Form */}
          <div className="contact-form-card">
            {submitted ? (
              <div className="cf-success">
                <div className="cf-success-ring">
                  <FaCheckCircle className="cf-success-icon" />
                </div>
                <h2>Message Sent! 🌱</h2>
                <p>Thanks for reaching out. Our team will get back to you within <strong>24 hours</strong>.</p>
                <div className="cf-success-detail">
                  <span>📧 Confirmation sent to <strong>{form.email || "your email"}</strong></span>
                </div>
                <button className="cf-submit-btn" onClick={handleReset}>Send Another Message</button>
              </div>
            ) : (
              <>
                <div className="form-card-header">
                  <div className="form-header-icon-wrap"><FaHeadset /></div>
                  <div>
                    <h2>Send a Message</h2>
                    <p>Fill in the form and our team will respond shortly.</p>
                  </div>
                </div>

                {/* Topic chips */}
                <div className="cf-topic-section">
                  <label>What's this about? <span className="required">*</span></label>
                  <div className="cf-topic-chips">
                    {TOPICS.map((t) => (
                      <button
                        key={t} type="button"
                        className={`cf-chip ${form.topic === t ? "active" : ""}`}
                        onClick={() => { setForm({ ...form, topic: t }); setErrors({ ...errors, topic: "" }); }}
                      >{t}</button>
                    ))}
                  </div>
                  {errors.topic && <span className="cf-error">{errors.topic}</span>}
                </div>

                <form className="contact-form" onSubmit={handleSubmit} noValidate>
                  <div className="cf-row">
                    <div className={`cf-group ${focused === "name" ? "cf-focused" : ""}`}>
                      <label htmlFor="name">Full Name <span className="required">*</span></label>
                      <div className="cf-input-wrap">
                        <input id="name" name="name" type="text" placeholder="e.g. Ramesh Kumar"
                          value={form.name} onChange={handleChange}
                          onFocus={() => setFocused("name")} onBlur={() => setFocused(null)}
                          className={errors.name ? "cf-error-input" : ""} />
                        {form.name && !errors.name && <FaCheckCircle className="cf-valid-icon" />}
                      </div>
                      {errors.name && <span className="cf-error">{errors.name}</span>}
                    </div>
                    <div className={`cf-group ${focused === "email" ? "cf-focused" : ""}`}>
                      <label htmlFor="email">Email Address <span className="required">*</span></label>
                      <div className="cf-input-wrap">
                        <input id="email" name="email" type="email" placeholder="e.g. ramesh@example.com"
                          value={form.email} onChange={handleChange}
                          onFocus={() => setFocused("email")} onBlur={() => setFocused(null)}
                          className={errors.email ? "cf-error-input" : ""} />
                        {form.email && !errors.email && /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(form.email) && <FaCheckCircle className="cf-valid-icon" />}
                      </div>
                      {errors.email && <span className="cf-error">{errors.email}</span>}
                    </div>
                  </div>

                  <div className={`cf-group ${focused === "phone" ? "cf-focused" : ""}`}>
                    <label htmlFor="phone">Phone Number <span className="optional">(optional)</span></label>
                    <div className="cf-input-wrap">
                      <input id="phone" name="phone" type="tel" placeholder="e.g. +91 00000 00000"
                        value={form.phone} onChange={handleChange}
                        onFocus={() => setFocused("phone")} onBlur={() => setFocused(null)}
                        className={errors.phone ? "cf-error-input" : ""} />
                    </div>
                    {errors.phone && <span className="cf-error">{errors.phone}</span>}
                  </div>

                  <div className={`cf-group ${focused === "message" ? "cf-focused" : ""}`}>
                    <label htmlFor="message">
                      Message <span className="required">*</span>
                      <span className={`cf-char-count ${form.message.length > 450 ? "warn" : ""}`}>{form.message.length}/500</span>
                    </label>
                    <textarea id="message" name="message" rows={5}
                      placeholder="Describe your query or feedback in detail..."
                      value={form.message} onChange={handleChange} maxLength={500}
                      onFocus={() => setFocused("message")} onBlur={() => setFocused(null)}
                      className={errors.message ? "cf-error-input" : ""} />
                    {errors.message && <span className="cf-error">{errors.message}</span>}
                  </div>

                  {/* Progress bar */}
                  <div className="cf-progress-wrapper">
                    <div className="cf-progress">
                      <div className="cf-progress-bar" style={{ width: `${Math.min(100, ([form.name, form.email, form.topic, form.message].filter(Boolean).length / 4) * 100)}%` }} />
                    </div>
                    <span>{[form.name, form.email, form.topic, form.message].filter(Boolean).length} of 4 required fields filled</span>
                  </div>

                  <button type="submit" className="cf-submit-btn" disabled={loading}>
                    {loading ? <><span className="cf-spinner" /> Sending...</> : <><FaPaperPlane /> Send Message</>}
                  </button>
                </form>
              </>
            )}
          </div>

          {/* Side Panel */}
          <div className="contact-side-panel">
            <div className="contact-map">
              <div className="map-label"><FaMapMarkerAlt /> AgriTech Park, New Delhi</div>
              <iframe
                title="Fasal Saathi Location"
                src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d224345.83923192955!2d77.06889754725782!3d28.52758200617607!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x390cfd5b347eb62d%3A0x52c2b7494e204dce!2sNew%20Delhi%2C%20Delhi!5e0!3m2!1sen!2sin!4v1234567890"
                width="100%" height="220" style={{ border: 0 }}
                allowFullScreen loading="lazy" referrerPolicy="no-referrer-when-downgrade"
              />
            </div>

            <div className="contact-social-card">
              <h3>🌐 Follow Us</h3>
              <p>Stay updated with farming tips and platform news.</p>
              <div className="social-links">
                {[
                  { icon: <FaFacebook />, label: "Facebook", href: "https://facebook.com/fasalsaathi", color: "#1877f2" },
                  { icon: <FaTwitter />, label: "Twitter", href: "https://twitter.com/fasalsaathi", color: "#1da1f2" },
                  { icon: <FaInstagram />, label: "Instagram", href: "https://instagram.com/fasalsaathi", color: "#e1306c" },
                  { icon: <FaYoutube />, label: "YouTube", href: "https://youtube.com/@fasalsaathi", color: "#ff0000" },
                  { icon: <FaWhatsapp />, label: "WhatsApp", href: "https://wa.me/919999999999", color: "#25d366" },
                ].map((s) => (
                  <a key={s.label} href={s.href} className="social-link" style={{ "--social-color": s.color }} aria-label={s.label} target="_blank" rel="noopener noreferrer">
                    <span className="social-icon" style={{ color: s.color }}>{s.icon}</span>
                    <span>{s.label}</span>
                  </a>
                ))}
              </div>
            </div>

          </div>
        </div>

        {/* FAQ */}
        <div className="contact-faq">
          <div className="faq-header">
            <div>
              <h2>Frequently Asked Questions</h2>
               <p className="faq-subtitle">Quick answers to common questions about <span className="notranslate" translate="no">Fasal Saathi</span>.</p>
            </div>
            <div className="faq-badge">{FAQ_ITEMS.length} Questions</div>
          </div>
          <div className="faq-list">
            {FAQ_ITEMS.map((item, i) => (
              <div key={i} className={`faq-item ${openFaq === i ? "open" : ""}`}>
                <button className="faq-question" onClick={() => setOpenFaq(openFaq === i ? null : i)}>
                  <span className="faq-num">0{i + 1}</span>
                  <span>{item.q}</span>
                  <span className="faq-chevron">{openFaq === i ? "−" : "+"}</span>
                </button>
                <div className="faq-answer-wrap">
                  <div className="faq-answer">{item.a}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
