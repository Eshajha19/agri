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
        Is <span className="notranslate" translate="no">Fasal Saathi</span> available in my language?
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
        Can I use <span className="notranslate" translate="no">Fasal Saathi</span> offline?
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

      {/* HERO */}
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

        {/* FAQ SECTION */}
        <div className="contact-faq">
          <div className="faq-header">
            <div>
              <h2>Frequently Asked Questions</h2>

              <p className="faq-subtitle">
                Quick answers to common questions about{" "}
                <span className="notranslate" translate="no">
                  Fasal Saathi
                </span>
                .
              </p>
            </div>

            <div className="faq-badge">
              {FAQ_ITEMS.length} Questions
            </div>
          </div>

          <div className="faq-list">
            {FAQ_ITEMS.map((item, i) => (
              <div
                key={i}
                className={`faq-item ${openFaq === i ? "open" : ""}`}
              >
                {/* FULL CLICKABLE BUTTON */}
                <button
                  type="button"
                  className="faq-question"
                  onClick={() => toggleFaq(i)}
                  aria-expanded={openFaq === i}
                >
                  <span className="faq-num">
                    {String(i + 1).padStart(2, "0")}
                  </span>

                  <span className="faq-title">
                    {item.q}
                  </span>

                  {/* FIXED ICON */}
                  <span className="faq-chevron">
                    {openFaq === i ? "−" : "+"}
                  </span>
                </button>

                {/* EXPANDABLE CONTENT */}
                <div
                  className={`faq-answer-wrap ${
                    openFaq === i ? "show" : ""
                  }`}
                >
                  <div className="faq-answer">
                    {item.a}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}