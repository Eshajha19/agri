/**
 * ConsultationHistory.jsx
 *
 * Security / UX fix: removed all MOCK_CONSULTATIONS fallback behaviour.
 *
 * Previously the component silently replaced real (empty) Firestore
 * results with four hardcoded fake consultations, including:
 *  - Fictional expert names and fabricated session notes.
 *  - A "scheduled" consultation with "Dr. Ramesh Kumar" dated tomorrow,
 *    which a new farmer could mistake for a real upcoming appointment.
 *  - randomuser.me avatar URLs that load third-party tracking pixels.
 *  - The same fake data was shown on any Firestore error, so a network
 *    blip caused real users to see fabricated history.
 *
 * Now:
 *  - Empty Firestore result → proper empty-state UI ("No consultations yet").
 *  - Firestore error → error-state UI with a retry button; no fake data.
 *  - Unauthenticated / anonymous user → prompt to sign in; no fake data.
 *  - Avatar images fall back to a local placeholder instead of randomuser.me.
 */
import React, { useState, useEffect, useCallback } from "react";
import { useAdvisorStore } from "../stores/advisorStore";
import { db, auth } from "../lib/firebase";
import { collection, getDocs, query, where, orderBy } from "firebase/firestore";
import { toast } from "react-toastify";
import {
  X,
  Calendar,
  Clock,
  Video,
  PhoneCall,
  CheckCircle,
  XCircle,
  AlertCircle,
  Play,
  Star,
  UserCircle,
} from "lucide-react";

// Local SVG data-URI avatar — no third-party requests, no tracking pixels.
const FALLBACK_AVATAR =
  "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 40 40'%3E" +
  "%3Ccircle cx='20' cy='20' r='20' fill='%23e0e7ef'/%3E" +
  "%3Ccircle cx='20' cy='16' r='7' fill='%23b0bec5'/%3E" +
  "%3Cellipse cx='20' cy='34' rx='11' ry='8' fill='%23b0bec5'/%3E%3C/svg%3E";

function ConsultationHistory({ onClose }) {
  const { setShowTeleConsultation, setActiveConsultation } = useAdvisorStore();
  const user = auth?.currentUser;

  const [consultations, setConsultations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [filter, setFilter] = useState("all");
  const [selectedConsultation, setSelectedConsultation] = useState(null);
  const [rating, setRating] = useState(0);
  const [feedback, setFeedback] = useState("");

  const loadConsultations = useCallback(async () => {
    setLoading(true);
    setError("");

    // Unauthenticated or anonymous — show a sign-in prompt, not fake data.
    if (!user || user.isAnonymous) {
      setConsultations([]);
      setLoading(false);
      return;
    }

    try {
      const consultationsRef = collection(db, "consultations");
      const q = query(
        consultationsRef,
        where("userId", "==", user.uid),
        orderBy("createdAt", "desc")
      );
      const snapshot = await getDocs(q);
      const data = snapshot.docs.map((doc) => ({ id: doc.id, ...doc.data() }));
      // Empty result is a valid state — show the empty-state UI, not fake data.
      setConsultations(data);
    } catch (err) {
      console.error("Error loading consultations:", err);
      // Surface the error so the user knows something went wrong and can retry.
      // Do NOT fall back to mock data — that would show fabricated history.
      setError("Failed to load consultations. Please check your connection and try again.");
      setConsultations([]);
    } finally {
      setLoading(false);
    }
  }, [user]);

  useEffect(() => {
    loadConsultations();
  }, [loadConsultations]);

  const filteredConsultations = consultations.filter((c) => {
    if (filter === "all")       return true;
    if (filter === "upcoming")  return c.status === "scheduled";
    if (filter === "completed") return c.status === "completed";
    if (filter === "cancelled") return c.status === "cancelled";
    return true;
  });

  const getStatusIcon = (status) => {
    switch (status) {
      case "scheduled":   return <Calendar   size={16} className="status-scheduled"  />;
      case "completed":   return <CheckCircle size={16} className="status-completed"  />;
      case "cancelled":   return <XCircle    size={16} className="status-cancelled"  />;
      default:            return <AlertCircle size={16} className="status-pending"    />;
    }
  };

  const getStatusLabel = (status) => {
    switch (status) {
      case "scheduled":   return "Upcoming";
      case "completed":   return "Completed";
      case "cancelled":   return "Cancelled";
      case "in-progress": return "In Progress";
      default:            return status;
    }
  };

  const handleStartConsultation = (consultation) => {
    setActiveConsultation(consultation);
    setShowTeleConsultation(true);
  };

  const handleRatingSubmit = () => {
    toast.success("Thank you for your feedback!");
    setSelectedConsultation(null);
    setRating(0);
    setFeedback("");
  };

  const formatDate = (dateStr) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString("en-US", {
      weekday: "short",
      day: "numeric",
      month: "short",
      year: "numeric",
    });
  };

  // ── Render helpers ──────────────────────────────────────────────────────

  const renderBody = () => {
    if (loading) {
      return (
        <div className="loading-state">
          <div className="spinner" />
          <p>Loading consultations…</p>
        </div>
      );
    }

    // Unauthenticated user — prompt to sign in.
    if (!user || user.isAnonymous) {
      return (
        <div className="empty-state">
          <UserCircle size={48} />
          <p>Sign in to view your consultations</p>
          <span>Your consultation history will appear here once you are logged in.</span>
        </div>
      );
    }

    // Firestore error — show error with retry, not fake data.
    if (error) {
      return (
        <div className="empty-state error-state">
          <AlertCircle size={48} />
          <p>Could not load consultations</p>
          <span>{error}</span>
          <button className="retry-btn" onClick={loadConsultations}>
            Retry
          </button>
        </div>
      );
    }

    // Authenticated user with no consultations yet.
    if (filteredConsultations.length === 0) {
      return (
        <div className="empty-state">
          <Calendar size={48} />
          <p>No consultations found</p>
          <span>Book a consultation with an expert to get started.</span>
        </div>
      );
    }

    return (
      <div className="consultations-list">
        {filteredConsultations.map((consultation) => (
          <div
            key={consultation.id}
            className={`consultation-card ${consultation.status}`}
          >
            <div className="consultation-header">
              <div className="expert-info-mini">
                <img
                  src={FALLBACK_AVATAR}
                  alt={consultation.expertName || "Expert"}
                  className="expert-avatar-mini"
                />
                <div>
                  <h4>{consultation.expertName}</h4>
                  <p>{consultation.expertSpecialization}</p>
                </div>
              </div>
              <div className={`status-badge ${consultation.status}`}>
                {getStatusIcon(consultation.status)}
                <span>{getStatusLabel(consultation.status)}</span>
              </div>
            </div>

            <div className="consultation-details">
              <div className="detail-item">
                <Calendar size={14} />
                <span>{formatDate(consultation.date)}</span>
              </div>
              <div className="detail-item">
                <Clock size={14} />
                <span>{consultation.time}</span>
              </div>
              <div className="detail-item">
                {consultation.type === "video"
                  ? <Video size={14} />
                  : <PhoneCall size={14} />}
                <span>{consultation.type === "video" ? "Video Call" : "Audio Call"}</span>
              </div>
            </div>

            {consultation.notes && (
              <div className="consultation-notes">
                <strong>Notes:</strong> {consultation.notes}
              </div>
            )}

            {consultation.status === "scheduled" && (
              <div className="consultation-actions">
                <button
                  className="start-btn"
                  onClick={() => handleStartConsultation(consultation)}
                >
                  <Play size={16} /> Join Consultation
                </button>
                <button className="reschedule-btn">Reschedule</button>
              </div>
            )}

            {consultation.status === "completed" && (
              <div className="consultation-actions">
                <button
                  className="feedback-btn"
                  onClick={() => setSelectedConsultation(consultation)}
                >
                  <Star size={16} /> Give Feedback
                </button>
              </div>
            )}
          </div>
        ))}
      </div>
    );
  };

  return (
    <div className="consultation-history">
      <div className="history-header">
        <h2>
          <Calendar className="header-icon" /> Consultation History
        </h2>
        <button className="close-btn" onClick={onClose} aria-label="Close">
          <X size={20} />
        </button>
      </div>

      <div className="history-filters">
        {["all", "upcoming", "completed", "cancelled"].map((f) => (
          <button
            key={f}
            className={`filter-btn ${filter === f ? "active" : ""}`}
            onClick={() => setFilter(f)}
          >
            {f.charAt(0).toUpperCase() + f.slice(1)}
          </button>
        ))}
      </div>

      {renderBody()}

      {selectedConsultation && (
        <div className="feedback-modal">
          <div className="feedback-content">
            <h3>Rate Your Consultation</h3>
            <p>How was your session with {selectedConsultation.expertName}?</p>

            <div className="rating-stars">
              {[1, 2, 3, 4, 5].map((star) => (
                <button
                  key={star}
                  className={`star-btn ${rating >= star ? "active" : ""}`}
                  onClick={() => setRating(star)}
                  aria-label={`Rate ${star} star${star > 1 ? "s" : ""}`}
                >
                  <Star size={32} fill={rating >= star ? "#ffd700" : "none"} />
                </button>
              ))}
            </div>

            <textarea
              placeholder="Share your experience…"
              value={feedback}
              onChange={(e) => setFeedback(e.target.value)}
              rows={3}
            />

            <div className="feedback-actions">
              <button className="cancel-btn" onClick={() => setSelectedConsultation(null)}>
                Cancel
              </button>
              <button className="submit-btn" onClick={handleRatingSubmit}>
                Submit Feedback
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default ConsultationHistory;
