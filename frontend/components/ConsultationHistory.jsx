import React, { useState, useEffect } from "react";
import { useAdvisorStore } from "../stores/advisorStore";
import { db, auth } from "../lib/firebase";
import { collection, getDocs, query, where, orderBy } from "firebase/firestore";
import { toast } from "react-toastify";
import {
  X,
  Calendar,
  Clock,
  Video,
  Phone,
  PhoneCall,
  CheckCircle,
  XCircle,
  AlertCircle,
  Play,
  Star,
} from "lucide-react";

function ConsultationHistory({ onClose, onStartConsultation: _onStartConsultation }) {
  const { setShowTeleConsultation, setActiveConsultation } = useAdvisorStore();
  const user = auth?.currentUser;
  const [consultations, setConsultations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState("all");
  const [selectedConsultation, setSelectedConsultation] = useState(null);
  const [showRating, setShowRating] = useState(false);
  const [rating, setRating] = useState(0);
  const [feedback, setFeedback] = useState("");

  useEffect(() => {
    loadConsultations();
  }, []);

  const loadConsultations = async () => {
    setLoading(true);
    try {
      if (!user || user.isAnonymous) {
        setConsultations(MOCK_CONSULTATIONS);
        setLoading(false);
        return;
      }

      const consultationsRef = collection(db, "consultations");
      const q = query(
        consultationsRef,
        where("userId", "==", user.uid),
        orderBy("createdAt", "desc")
      );

      const snapshot = await getDocs(q);
      const data = snapshot.docs.map((doc) => ({
        id: doc.id,
        ...doc.data(),
      }));

      if (data.length === 0) {
        setConsultations(MOCK_CONSULTATIONS);
      } else {
        setConsultations(data);
      }
    } catch (error) {
      console.error("Error loading consultations:", error);
      setConsultations(MOCK_CONSULTATIONS);
    } finally {
      setLoading(false);
    }
  };

  const filteredConsultations = consultations.filter((c) => {
    if (filter === "all") return true;
    if (filter === "upcoming") return c.status === "scheduled";
    if (filter === "completed") return c.status === "completed";
    if (filter === "cancelled") return c.status === "cancelled";
    return true;
  });

  const getStatusIcon = (status) => {
    switch (status) {
      case "scheduled":
        return <Calendar size={16} className="status-scheduled" />;
      case "completed":
        return <CheckCircle size={16} className="status-completed" />;
      case "cancelled":
        return <XCircle size={16} className="status-cancelled" />;
      default:
        return <AlertCircle size={16} className="status-pending" />;
    }
  };

  const getStatusLabel = (status) => {
    switch (status) {
      case "scheduled":
        return "Upcoming";
      case "completed":
        return "Completed";
      case "cancelled":
        return "Cancelled";
      case "in-progress":
        return "In Progress";
      default:
        return status;
    }
  };

  const handleStartConsultation = (consultation) => {
    setActiveConsultation(consultation);
    setShowTeleConsultation(true);
  };

  const handleRatingSubmit = () => {
    toast.success("Thank you for your feedback!");
    setShowRating(false);
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

  return (
    <div className="consultation-history">
        <div className="history-header">
          <h2>
            <Calendar className="header-icon" /> Consultation History
          </h2>
          <button className="close-btn" onClick={onClose}>
            <X size={20} />
          </button>
        </div>

        <div className="history-filters">
          <button
            className={`filter-btn ${filter === "all" ? "active" : ""}`}
            onClick={() => setFilter("all")}
          >
            All
          </button>
          <button
            className={`filter-btn ${filter === "upcoming" ? "active" : ""}`}
            onClick={() => setFilter("upcoming")}
          >
            Upcoming
          </button>
          <button
            className={`filter-btn ${filter === "completed" ? "active" : ""}`}
            onClick={() => setFilter("completed")}
          >
            Completed
          </button>
          <button
            className={`filter-btn ${filter === "cancelled" ? "active" : ""}`}
            onClick={() => setFilter("cancelled")}
          >
            Cancelled
          </button>
        </div>

        {loading ? (
          <div className="loading-state">
            <div className="spinner"></div>
            <p>Loading consultations...</p>
          </div>
        ) : filteredConsultations.length === 0 ? (
          <div className="empty-state">
            <Calendar size={48} />
            <p>No consultations found</p>
            <span>Book a consultation with an expert to get started</span>
          </div>
        ) : (
          <div className="consultations-list">
            {filteredConsultations.map((consultation) => (
              <div
                key={consultation.id}
                className={`consultation-card ${consultation.status}`}
              >
                <div className="consultation-header">
                  <div className="expert-info-mini">
                    <img
                      src={consultation.avatar || "https://randomuser.me/api/portraits/men/32.jpg"}
                      alt={consultation.expertName}
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
                    {consultation.type === "video" ? <Video size={14} /> : <PhoneCall size={14} />}
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
        )}

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
                  >
                    <Star size={32} fill={rating >= star ? "#ffd700" : "none"} />
                  </button>
                ))}
              </div>

              <textarea
                placeholder="Share your experience..."
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

const MOCK_CONSULTATIONS = [
  {
    id: "c1",
    expertId: "exp1",
    expertName: "Dr. Ramesh Kumar",
    expertSpecialization: "Crop Disease",
    date: new Date(Date.now() + 2 * 24 * 60 * 60 * 1000).toISOString().split("T")[0],
    time: "10:00",
    status: "scheduled",
    type: "video",
    avatar: "https://randomuser.me/api/portraits/men/32.jpg",
    notes: "Discussing pest control for tomato crops",
  },
  {
    id: "c2",
    expertId: "exp2",
    expertName: "Dr. Priya Sharma",
    expertSpecialization: "Fertilizers",
    date: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString().split("T")[0],
    time: "14:30",
    status: "completed",
    type: "video",
    duration: 1800,
    avatar: "https://randomuser.me/api/portraits/women/44.jpg",
    notes: "Recommended organic fertilizer for cotton",
  },
  {
    id: "c3",
    expertId: "exp3",
    expertName: "Er. Suresh Patil",
    expertSpecialization: "Irrigation",
    date: new Date(Date.now() - 14 * 24 * 60 * 60 * 1000).toISOString().split("T")[0],
    time: "11:00",
    status: "completed",
    type: "audio",
    duration: 900,
    avatar: "https://randomuser.me/api/portraits/men/45.jpg",
    notes: "Drip irrigation system maintenance",
  },
  {
    id: "c4",
    expertId: "exp5",
    expertName: "Dr. Mahendra Singh",
    expertSpecialization: "Soil Health",
    date: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString().split("T")[0],
    time: "09:00",
    status: "cancelled",
    type: "video",
    avatar: "https://randomuser.me/api/portraits/men/67.jpg",
    notes: "Soil testing results discussion",
  },
];

export default ConsultationHistory;