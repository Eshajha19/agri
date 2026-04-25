import React, { useState, useEffect } from "react";
import { collection, getDocs, deleteDoc, doc, query, orderBy } from "firebase/firestore";
import { db, auth } from "./lib/firebase";
import { Star, Trash2, RefreshCw, MessageSquare, TrendingUp, Users, Award, ShieldAlert } from "lucide-react";
import "./AdminFeedback.css";
import Loader from "./Loader";

// ── ADMIN ACCESS CONTROL ──────────────────────────────────────────────────────
// Add your admin email(s) here. Only these emails can access the admin panel.
const ADMIN_EMAILS = ["tushar.18246@gmail.com"];

const CATEGORY_LABELS = {
  general: "💬 General",
  feature: "✨ Feature Request",
  bug: "🐛 Bug Report",
  ui: "🎨 UI/UX",
  accuracy: "🎯 AI Accuracy",
  other: "📌 Other",
};

export default function AdminFeedback() {
  const [feedbacks, setFeedbacks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [authChecking, setAuthChecking] = useState(true);
  const [filter, setFilter] = useState("all");
  const [sortBy, setSortBy] = useState("newest");

  useEffect(() => {
    // Check if auth is resolved
    const unsubscribe = auth.onAuthStateChanged(() => {
      setAuthChecking(false);
    });
    return () => unsubscribe();
  }, []);

  const currentUserEmail = auth?.currentUser?.email;
  const isAdmin = ADMIN_EMAILS.includes(currentUserEmail);

  const fetchFeedbacks = async () => {
    setLoading(true);
    try {
      const q = query(collection(db, "feedback"), orderBy("createdAt", "desc"));
      const snapshot = await getDocs(q);
      const data = snapshot.docs.map((snap) => ({ id: snap.id, ...snap.data() }));
      setFeedbacks(data);
    } catch (err) {
      console.error("Error fetching feedback:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!authChecking && isAdmin) {
      fetchFeedbacks();
    }
  }, [authChecking, isAdmin]);

  // Block non-admins immediately, but wait for auth to check
  if (authChecking) {
    return (
      <Loader fullPage={true} message="Authenticating Admin..." />
    );
  }

  if (!isAdmin) {
    return (
      <div className="admin-fb-page" style={{ display: "flex", alignItems: "center", justifyContent: "center", minHeight: "60vh" }}>
        <div style={{ textAlign: "center", padding: "40px" }}>
          <ShieldAlert size={72} style={{ color: "#ef4444", marginBottom: "16px" }} />
          <h2 style={{ fontSize: "24px", fontWeight: 800, color: "#0f172a", margin: "0 0 12px" }}>
            Access Denied
          </h2>
          <p style={{ color: "#64748b", maxWidth: "400px" }}>
            You do not have permission to view this page. This area is restricted to administrators only.
          </p>
        </div>
      </div>
    );
  }

  const handleDelete = async (id) => {
    if (!window.confirm("Delete this feedback?")) return;
    try {
      await deleteDoc(doc(db, "feedback", id));
      setFeedbacks((prev) => prev.filter((f) => f.id !== id));
    } catch (err) {
      console.error("Error deleting:", err);
    }
  };

  const filtered = feedbacks.filter((f) => filter === "all" || f.category === filter);

  const getSafeDate = (dateVal) => {
    if (!dateVal) return null;
    let d;
    if (typeof dateVal.toDate === 'function') {
      d = dateVal.toDate();
    } else if (dateVal.seconds) {
      d = new Date(dateVal.seconds * 1000);
    } else {
      d = new Date(dateVal);
    }
    return isNaN(d) ? null : d;
  };

  const sorted = [...filtered].sort((a, b) => {
    if (sortBy === "highest") return (b.rating || 0) - (a.rating || 0);
    if (sortBy === "lowest") return (a.rating || 0) - (b.rating || 0);
    const dateA = getSafeDate(a.createdAt) || new Date(0);
    const dateB = getSafeDate(b.createdAt) || new Date(0);
    return dateB - dateA;
  });

  const avgRating = feedbacks.length
    ? (feedbacks.reduce((s, f) => s + (f.rating || 0), 0) / feedbacks.length).toFixed(1)
    : "N/A";

  const ratingCounts = [5, 4, 3, 2, 1].map((r) => ({
    star: r,
    count: feedbacks.filter((f) => f.rating === r).length,
  }));

  return (
    <div className="admin-fb-page">
      <div className="admin-fb-header">
        <div>
          <h1>📋 Feedback Dashboard</h1>
          <p>All submissions from Fasal Saathi users</p>
        </div>
        <button className="refresh-btn" onClick={fetchFeedbacks} disabled={loading}>
          <RefreshCw size={16} className={loading ? "spin" : ""} /> Refresh
        </button>
      </div>

      {loading ? (
        <Loader message="Fetching feedback data..." />
      ) : (
        <>
          {/* Summary Cards */}
          <div className="admin-stats-row">
        <div className="admin-stat-card">
          <Users size={28} className="admin-stat-icon green" />
          <div>
            <div className="admin-stat-value">{feedbacks.length}</div>
            <div className="admin-stat-label">Total Responses</div>
          </div>
        </div>
        <div className="admin-stat-card">
          <Star size={28} className="admin-stat-icon yellow" />
          <div>
            <div className="admin-stat-value">{avgRating}</div>
            <div className="admin-stat-label">Average Rating</div>
          </div>
        </div>
        <div className="admin-stat-card">
          <Award size={28} className="admin-stat-icon purple" />
          <div>
            <div className="admin-stat-value">
              {feedbacks.filter((f) => f.rating >= 4).length}
            </div>
            <div className="admin-stat-label">Positive Ratings</div>
          </div>
        </div>
        <div className="admin-stat-card">
          <TrendingUp size={28} className="admin-stat-icon blue" />
          <div>
            <div className="admin-stat-value">
              {feedbacks.filter((f) => f.category === "feature").length}
            </div>
            <div className="admin-stat-label">Feature Requests</div>
          </div>
        </div>
      </div>

      {/* Rating Distribution */}
      <div className="rating-distribution">
        <h3>Rating Distribution</h3>
        {ratingCounts.map(({ star, count }) => (
          <div key={star} className="rating-bar-row">
            <span className="rbar-label">{star} ⭐</span>
            <div className="rbar-track">
              <div
                className="rbar-fill"
                style={{ width: feedbacks.length ? `${(count / feedbacks.length) * 100}%` : "0%" }}
              />
            </div>
            <span className="rbar-count">{count}</span>
          </div>
        ))}
      </div>

      {/* Filters */}
      <div className="admin-controls">
        <div className="filter-chips">
          {["all", "general", "feature", "bug", "ui", "accuracy", "other"].map((cat) => (
            <button
              key={cat}
              className={`filter-chip ${filter === cat ? "active" : ""}`}
              onClick={() => setFilter(cat)}
            >
              {cat === "all" ? "All" : CATEGORY_LABELS[cat]}
            </button>
          ))}
        </div>
        <select className="sort-select" value={sortBy} onChange={(e) => setSortBy(e.target.value)}>
          <option value="newest">Newest First</option>
          <option value="highest">Highest Rating</option>
          <option value="lowest">Lowest Rating</option>
        </select>
      </div>

      {/* Feedback Cards */}
      {sorted.length === 0 ? (
        <div className="admin-empty">
          <MessageSquare size={48} />
          <p>No feedback found.</p>
        </div>
      ) : (
        <div className="feedback-cards-grid">
          {sorted.map((fb) => (
            <div key={fb.id} className="fb-admin-card">
              <div className="fb-card-top">
                <div className="fb-user-info">
                  <div className="fb-avatar">{(fb.name?.[0] || "A").toUpperCase()}</div>
                  <div>
                    <div className="fb-user-name">{fb.name || "Anonymous"}</div>
                    <div className="fb-user-meta">
                      {fb.cropType && <span>🌾 {fb.cropType}</span>}
                      {fb.location && <span>📍 {fb.location}</span>}
                    </div>
                  </div>
                </div>
                <div className="fb-card-actions">
                  <span className="cat-tag">{CATEGORY_LABELS[fb.category] || fb.category}</span>
                  <button className="del-btn" onClick={() => handleDelete(fb.id)}>
                    <Trash2 size={14} />
                  </button>
                </div>
              </div>

              <div className="fb-stars">
                {[1, 2, 3, 4, 5].map((s) => (
                  <Star
                    key={s}
                    size={16}
                    fill={s <= fb.rating ? "#f59e0b" : "none"}
                    stroke={s <= fb.rating ? "#f59e0b" : "#cbd5e1"}
                  />
                ))}
                <span className="fb-rating-text">{fb.rating}/5</span>
              </div>

              <p className="fb-message">"{fb.message}"</p>

              <div className="fb-date">
                {getSafeDate(fb.createdAt)
                  ? getSafeDate(fb.createdAt).toLocaleDateString("en-IN", {
                    day: "numeric", month: "short", year: "numeric",
                  })
                  : "Unknown Date"}
              </div>
            </div>
          ))}
        </div>
      )}
      </>
      )}
    </div>
  );
}
