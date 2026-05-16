import React, { useEffect, useMemo, useState, useCallback } from "react";
import {
  FaSearch, FaExternalLinkAlt, FaInfoCircle, FaUserCheck,
  FaCheckCircle, FaFilter, FaArrowRight, FaClipboardList,
  FaBell, FaCalendarAlt, FaChevronDown, FaChevronUp, FaSpinner,
} from "react-icons/fa";
import { MdOutlineWorkspacePremium, MdOutlineWaterDrop } from "react-icons/md";
import { auth, db, isFirebaseConfigured } from "./lib/firebase";
import {
  collection, addDoc, query, where, onSnapshot,
  updateDoc, doc, serverTimestamp, getDoc,
} from "firebase/firestore";
import "./GovernmentSchemes.css";

// ── Scheme data ───────────────────────────────────────────────────────────────
const SCHEMES_DATA = [
  {
    id: 1, title: "PM-KISAN", fullName: "Pradhan Mantri Kisan Samman Nidhi",
    category: "Financial Support", icon: "💰", featured: true,
    benefits: "Fixed income support of Rs.6,000 per year in three equal installments.",
    eligibility: "All landholding farmer families (some exclusions for high-income groups).",
    link: "https://pmkisan.gov.in/",
    deadline: "Rolling — apply anytime",
    documents: ["Aadhaar Card", "Land ownership records", "Bank account details", "Mobile number linked to Aadhaar"],
    eligibilityRules: { anyFarmer: true },
  },
  {
    id: 2, title: "PMFBY", fullName: "Pradhan Mantri Fasal Bima Yojana",
    category: "Insurance", icon: "🛡️", featured: true,
    benefits: "Comprehensive insurance cover against crop failure due to natural risks.",
    eligibility: "All farmers growing notified crops in notified areas.",
    link: "https://pmfby.gov.in/",
    deadline: "Before sowing season cutoff",
    documents: ["Aadhaar Card", "Land records / Khasra", "Bank passbook", "Sowing certificate"],
    eligibilityRules: { anyFarmer: true },
  },
  {
    id: 3, title: "KCC", fullName: "Kisan Credit Card",
    category: "Credit", icon: "💳",
    benefits: "Timely credit for agriculture and allied activities with low interest rates.",
    eligibility: "Owner cultivators, tenant farmers, oral lessees, sharecroppers, and SHGs.",
    link: "https://www.myscheme.gov.in/schemes/kcc",
    deadline: "Rolling — apply at nearest bank",
    documents: ["Aadhaar Card", "PAN Card", "Land records", "Passport-size photo", "Bank account details"],
    eligibilityRules: { anyFarmer: true },
  },
  {
    id: 4, title: "PM-KMY", fullName: "Pradhan Mantri Kisan Maan Dhan Yojana",
    category: "Pension", icon: "👴",
    benefits: "Minimum fixed pension of Rs.3,000 per month upon reaching 60 years of age.",
    eligibility: "Small and Marginal Farmers aged between 18 to 40 years.",
    link: "https://maandhan.in/",
    deadline: "Rolling enrollment",
    documents: ["Aadhaar Card", "Bank account / Jan Dhan account", "Land records (up to 2 hectares)"],
    eligibilityRules: { anyFarmer: true },
  },
  {
    id: 5, title: "Soil Health Card", fullName: "Soil Health Card Scheme",
    category: "Resources", icon: "🧪",
    benefits: "Detailed report on soil nutrient status and fertilizer recommendations.",
    eligibility: "All farmers in India can get soil samples tested every 2 years.",
    link: "https://www.soilhealth.dac.gov.in/",
    deadline: "Rolling — visit nearest KVK",
    documents: ["Aadhaar Card", "Land details"],
    eligibilityRules: { anyFarmer: true },
  },
  {
    id: 6, title: "PMKSY", fullName: "Pradhan Mantri Krishi Sinchai Yojana",
    category: "Irrigation", icon: "💧", featured: true,
    benefits: "Subsidies for micro-irrigation systems to improve water efficiency.",
    eligibility: "Farmers, SHGs, and Trusts focused on agriculture.",
    link: "https://pmksy.gov.in/",
    deadline: "State-wise — check portal",
    documents: ["Aadhaar Card", "Land records", "Bank account", "Quotation from supplier"],
    eligibilityRules: { anyFarmer: true },
  },
  {
    id: 7, title: "PKVY", fullName: "Paramparagat Krishi Vikas Yojana",
    category: "Organic Farming", icon: "🌱",
    benefits: "Financial assistance for organic cultivation and certification.",
    eligibility: "Groups of 20 or more farmers forming clusters.",
    link: "https://dmsouthwest.delhi.gov.in/scheme/paramparagat-krishi-vikas-yojana/",
    deadline: "Annual — check state agriculture dept.",
    documents: ["Group formation certificate", "Land records", "Aadhaar of all members"],
    eligibilityRules: { anyFarmer: true },
  },
  {
    id: 8, title: "NHM", fullName: "National Horticulture Mission",
    category: "Horticulture", icon: "🍎",
    benefits: "Support for cold storage, greenhouses, and high-value crops.",
    eligibility: "Farmers interested in fruits, vegetables, flowers, and spices.",
    link: "https://www.myscheme.gov.in/schemes/midh",
    deadline: "Annual — state horticulture dept.",
    documents: ["Aadhaar Card", "Land records", "Bank account", "Project proposal"],
    eligibilityRules: { crops: ["Tomato", "Onion", "Potato", "Fruits", "Vegetables"] },
  },
  {
    id: 9, title: "Rythu Bandhu", fullName: "Farmers Investment Support Scheme",
    category: "State Specific", icon: "🌾",
    benefits: "Investment support of Rs.5,000 per acre per season for inputs.",
    eligibility: "Farmers in Telangana owning agricultural land.",
    link: "https://rythubharosa.telangana.gov.in",
    deadline: "Seasonal — Kharif and Rabi",
    documents: ["Aadhaar Card", "Pattadar Passbook", "Bank account"],
    eligibilityRules: { states: ["Telangana"] },
  },
  {
    id: 10, title: "KALIA", fullName: "Krushak Assistance for Livelihood and Income Augmentation",
    category: "State Specific", icon: "🚜",
    benefits: "Financial assistance for small and marginal farmers.",
    eligibility: "Small and marginal farmers and landless agricultural households.",
    link: "https://www.myscheme.gov.in/schemes/kalia",
    deadline: "Rolling — Odisha residents",
    documents: ["Aadhaar Card", "Land records", "Bank account", "Residence proof"],
    eligibilityRules: { states: ["Odisha"] },
  },
];

const CATEGORIES = [
  "All", "Financial Support", "Insurance", "Credit", "Pension",
  "Resources", "Irrigation", "Organic Farming", "Horticulture", "State Specific",
];

const STATUS_LABELS = {
  not_applied:  { label: "Not Applied",  color: "#64748b", bg: "#f1f5f9" },
  applied:      { label: "Applied",      color: "#2563eb", bg: "#eff6ff" },
  under_review: { label: "Under Review", color: "#d97706", bg: "#fffbeb" },
  approved:     { label: "Approved",     color: "#16a34a", bg: "#f0fdf4" },
  rejected:     { label: "Rejected",     color: "#dc2626", bg: "#fef2f2" },
  disbursed:    { label: "Disbursed",    color: "#7c3aed", bg: "#f5f3ff" },
};

// ── Eligibility engine ────────────────────────────────────────────────────────
function computeEligibility(scheme, userProfile) {
  if (!userProfile) {
    return { score: 0, eligible: false, reason: "Complete your profile to check eligibility." };
  }
  const rules = scheme.eligibilityRules || {};
  if (rules.anyFarmer) {
    return { score: 100, eligible: true, reason: "You appear eligible for this scheme." };
  }
  if (rules.crops && userProfile.cropType) {
    const match = rules.crops.some(c =>
      userProfile.cropType.toLowerCase().includes(c.toLowerCase())
    );
    if (match) return { score: 90, eligible: true, reason: "Your crop type matches this scheme." };
    return { score: 30, eligible: false, reason: "Your crop type may not qualify. Check the official site." };
  }
  if (rules.states && userProfile.address) {
    const match = rules.states.some(s =>
      userProfile.address.toLowerCase().includes(s.toLowerCase())
    );
    if (match) return { score: 95, eligible: true, reason: "Your location matches this scheme." };
    return { score: 10, eligible: false, reason: "This scheme is state-specific and may not apply to your location." };
  }
  return { score: 60, eligible: true, reason: "Likely eligible — verify on the official portal." };
}

// ── Main component ────────────────────────────────────────────────────────────
export default function Schemes() {
  const [searchTerm, setSearchTerm]         = useState("");
  const [activeCategory, setActiveCategory] = useState("All");
  const [userProfile, setUserProfile]       = useState(null);
  const [applications, setApplications]     = useState({});
  const [expandedCard, setExpandedCard]     = useState(null);
  const [applyingId, setApplyingId]         = useState(null);
  const [refInput, setRefInput]             = useState("");
  const [activeTab, setActiveTab]           = useState("all");
  const [toastMsg, setToastMsg]             = useState(null);

  const currentUser = isFirebaseConfigured() ? auth?.currentUser : null;

  const showToast = useCallback((msg, type = "success") => {
    setToastMsg({ msg, type });
    setTimeout(() => setToastMsg(null), 3500);
  }, []);

  // Load user profile
  useEffect(() => {
    if (!currentUser || !isFirebaseConfigured()) return;
    getDoc(doc(db, "users", currentUser.uid))
      .then(snap => { if (snap.exists()) setUserProfile(snap.data()); })
      .catch(() => {});
  }, [currentUser]);

  // Real-time listener for scheme applications
  useEffect(() => {
    if (!currentUser || !isFirebaseConfigured()) return;
    const q = query(
      collection(db, "scheme_applications"),
      where("userId", "==", currentUser.uid)
    );
    const unsub = onSnapshot(q, snap => {
      const map = {};
      snap.docs.forEach(d => { map[d.data().schemeId] = { id: d.id, ...d.data() }; });
      setApplications(map);
    }, err => console.error("scheme_applications listener:", err));
    return () => unsub();
  }, [currentUser]);

  // Track a scheme application + add calendar reminder
  const handleTrackApplication = useCallback(async (scheme) => {
    if (!currentUser || !isFirebaseConfigured()) {
      showToast("Please log in to track applications.", "error");
      return;
    }
    if (applications[scheme.id]) {
      showToast("Already tracking this scheme.", "info");
      return;
    }
    setApplyingId(scheme.id);
    try {
      await addDoc(collection(db, "scheme_applications"), {
        userId:      currentUser.uid,
        schemeId:    scheme.id,
        schemeTitle: scheme.title,
        schemeName:  scheme.fullName,
        status:      "applied",
        referenceNo: refInput.trim() || "",
        appliedAt:   serverTimestamp(),
        updatedAt:   serverTimestamp(),
        notes:       "",
        deadline:    scheme.deadline,
      });
      // Add a 30-day follow-up reminder to the Farming Calendar
      await addDoc(collection(db, "activities"), {
        userId:      currentUser.uid,
        title:       `Follow up: ${scheme.title} application`,
        type:        "Landmark",
        time:        "10:00",
        description: `Check status of your ${scheme.fullName} application.`,
        date:        new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString(),
        completed:   false,
        createdAt:   new Date().toISOString(),
      });
      setRefInput("");
      setExpandedCard(null);
      showToast(`Tracking ${scheme.title}! A 30-day reminder has been added to your Farming Calendar.`);
    } catch (err) {
      console.error(err);
      showToast("Failed to save. Please try again.", "error");
    } finally {
      setApplyingId(null);
    }
  }, [currentUser, applications, refInput, showToast]);

  // Update application status
  const handleStatusChange = useCallback(async (schemeId, newStatus) => {
    const appData = applications[schemeId];
    if (!appData) return;
    try {
      await updateDoc(doc(db, "scheme_applications", appData.id), {
        status: newStatus,
        updatedAt: serverTimestamp(),
      });
      showToast("Status updated.");
    } catch (err) {
      console.error(err);
      showToast("Failed to update status.", "error");
    }
  }, [applications, showToast]);

  useEffect(() => { window.scrollTo(0, 0); }, []);

  const filteredSchemes = useMemo(() => {
    let list = SCHEMES_DATA.filter(s => {
      const q = searchTerm.toLowerCase();
      const matchSearch =
        s.title.toLowerCase().includes(q) ||
        s.fullName.toLowerCase().includes(q) ||
        s.category.toLowerCase().includes(q);
      const matchCat = activeCategory === "All" || s.category === activeCategory;
      return matchSearch && matchCat;
    });
    if (activeTab === "eligible") {
      list = list.filter(s => computeEligibility(s, userProfile).eligible);
    } else if (activeTab === "tracked") {
      list = list.filter(s => applications[s.id]);
    }
    return list;
  }, [searchTerm, activeCategory, activeTab, userProfile, applications]);

  const featuredSchemes = SCHEMES_DATA.filter(s => s.featured);

  return (
    <div className="schemes-page">
      {/* Toast notification */}
      {toastMsg && (
        <div className={`schemes-toast schemes-toast--${toastMsg.type}`} role="alert" aria-live="polite">
          {toastMsg.msg}
        </div>
      )}

      {/* HERO */}
      <section className="schemes-hero">
        <div className="hero-badge">
          <MdOutlineWorkspacePremium /> Trusted Government Schemes
        </div>
        <h1>Empowering Farmers Through <span>Smart Support</span></h1>
        <p>Discover agricultural schemes for insurance, irrigation, pension, loans, subsidies, and income support designed for Indian farmers.</p>
        <div className="hero-stats">
          <div className="stat-card"><h3>{SCHEMES_DATA.length}+</h3><p>Government Schemes</p></div>
          <div className="stat-card"><h3>100%</h3><p>Official Resources</p></div>
          <div className="stat-card"><h3>{Object.keys(applications).length}</h3><p>Tracked by You</p></div>
        </div>
      </section>

      {/* Eligibility banner */}
      {currentUser && userProfile && (
        <div className="eligibility-banner">
          <FaUserCheck className="elig-icon" aria-hidden="true" />
          <div>
            <strong>Personalised Eligibility Active</strong>
            <span>
              Showing results based on your profile:
              {userProfile.cropType ? ` ${userProfile.cropType}` : " Farmer"}
              {userProfile.address ? ` · ${userProfile.address}` : " · India"}
            </span>
          </div>
        </div>
      )}

      {/* FEATURED */}
      <section className="featured-section">
        <div className="section-heading">
          <h2>Featured Schemes</h2>
          <p>Popular schemes farmers apply for most frequently.</p>
        </div>
        <div className="featured-grid">
          {featuredSchemes.map(scheme => {
            const elig = computeEligibility(scheme, userProfile);
            return (
              <div className="featured-card" key={scheme.id}>
                <div className="featured-top">
                  <span className="featured-icon">{scheme.icon}</span>
                  <span className={`featured-tag ${elig.eligible ? "" : "featured-tag--low"}`}>
                    <FaCheckCircle aria-hidden="true" />
                    {elig.eligible ? "Eligible" : "Check Eligibility"}
                  </span>
                </div>
                <h3>{scheme.title}</h3>
                <p>{scheme.fullName}</p>
                <a href={scheme.link} target="_blank" rel="noopener noreferrer" className="featured-btn">
                  Explore Scheme <FaArrowRight aria-hidden="true" />
                </a>
              </div>
            );
          })}
        </div>
      </section>

      {/* CONTROLS */}
      <section className="schemes-controls">
        <div className="schemes-search">
          <FaSearch className="search-icon" aria-hidden="true" />
          <input
            type="text"
            placeholder="Search schemes, benefits, categories..."
            value={searchTerm}
            onChange={e => setSearchTerm(e.target.value)}
            aria-label="Search government schemes"
          />
        </div>
        <div className="filter-title"><FaFilter aria-hidden="true" /> Filter Categories</div>
        <div className="schemes-filter-chips">
          {CATEGORIES.map(cat => (
            <button
              key={cat}
              className={`filter-chip ${activeCategory === cat ? "active" : ""}`}
              onClick={() => setActiveCategory(cat)}
            >
              {cat}
            </button>
          ))}
        </div>

        {/* View tabs */}
        <div className="scheme-view-tabs" role="tablist">
          {[
            { key: "all",      label: "All Schemes" },
            { key: "eligible", label: "Eligible for Me" },
            { key: "tracked",  label: `My Applications (${Object.keys(applications).length})` },
          ].map(tab => (
            <button
              key={tab.key}
              role="tab"
              aria-selected={activeTab === tab.key}
              className={`view-tab ${activeTab === tab.key ? "active" : ""}`}
              onClick={() => setActiveTab(tab.key)}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </section>

      {/* SCHEME GRID */}
      <section className="schemes-grid">
        {filteredSchemes.length > 0 ? filteredSchemes.map(scheme => {
          const elig       = computeEligibility(scheme, userProfile);
          const appData    = applications[scheme.id];
          const isOpen     = expandedCard === scheme.id;
          const statusInfo = appData ? STATUS_LABELS[appData.status] : null;

          return (
            <div
              className={`scheme-card${appData ? " scheme-card--tracked" : ""}`}
              key={scheme.id}
            >
              <div className="scheme-header">
                <div className="scheme-icon">{scheme.icon}</div>
                <span className="scheme-category">{scheme.category}</span>
              </div>

              <div className="scheme-content">
                <h2>{scheme.title}</h2>
                <p className="scheme-fullname">{scheme.fullName}</p>

                {/* Eligibility score bar */}
                <div className="elig-score-row">
                  <span className={`elig-badge ${elig.eligible ? "elig-badge--yes" : "elig-badge--no"}`}>
                    {elig.eligible ? "Likely Eligible" : "May Not Qualify"}
                  </span>
                  <div className="elig-bar-wrap" title={`Eligibility score: ${elig.score}%`} aria-label={`Eligibility ${elig.score}%`}>
                    <div
                      className="elig-bar"
                      style={{ width: `${elig.score}%`, background: elig.eligible ? "#22c55e" : "#f59e0b" }}
                    />
                  </div>
                  <span className="elig-pct">{elig.score}%</span>
                </div>
                <p className="elig-reason">{elig.reason}</p>

                {/* Application status */}
                {appData && statusInfo && (
                  <div className="app-status-row">
                    <span
                      className="app-status-badge"
                      style={{ color: statusInfo.color, background: statusInfo.bg }}
                    >
                      {statusInfo.label}
                    </span>
                    {appData.referenceNo && (
                      <span className="app-ref">Ref: {appData.referenceNo}</span>
                    )}
                    <select
                      className="status-select"
                      value={appData.status}
                      onChange={e => handleStatusChange(scheme.id, e.target.value)}
                      aria-label="Update application status"
                    >
                      {Object.entries(STATUS_LABELS).map(([k, v]) => (
                        <option key={k} value={k}>{v.label}</option>
                      ))}
                    </select>
                  </div>
                )}

                <div className="scheme-info">
                  <div className="info-box">
                    <h4><FaInfoCircle aria-hidden="true" /> Benefits</h4>
                    <p>{scheme.benefits}</p>
                  </div>
                  <div className="info-box">
                    <h4><FaUserCheck aria-hidden="true" /> Eligibility</h4>
                    <p>{scheme.eligibility}</p>
                  </div>
                </div>

                {/* Expand toggle */}
                <button
                  className="expand-btn"
                  onClick={() => setExpandedCard(isOpen ? null : scheme.id)}
                  aria-expanded={isOpen}
                  aria-controls={`scheme-details-${scheme.id}`}
                >
                  <FaClipboardList aria-hidden="true" />
                  {isOpen ? "Hide Details" : "Documents and Track"}
                  {isOpen ? <FaChevronUp aria-hidden="true" /> : <FaChevronDown aria-hidden="true" />}
                </button>

                {isOpen && (
                  <div id={`scheme-details-${scheme.id}`} className="scheme-expanded">
                    {/* Document checklist */}
                    <div className="doc-checklist">
                      <h4><FaClipboardList aria-hidden="true" /> Required Documents</h4>
                      <ul>
                        {scheme.documents.map((docItem, i) => (
                          <li key={i}>
                            <FaCheckCircle className="doc-check" aria-hidden="true" />
                            {docItem}
                          </li>
                        ))}
                      </ul>
                    </div>

                    {/* Deadline */}
                    <div className="deadline-row">
                      <FaCalendarAlt aria-hidden="true" />
                      <strong>Deadline:</strong> {scheme.deadline}
                    </div>

                    {/* Track form or already-tracked state */}
                    {!appData ? (
                      <div className="track-form">
                        <input
                          type="text"
                          placeholder="Reference / Application No. (optional)"
                          value={refInput}
                          onChange={e => setRefInput(e.target.value)}
                          className="ref-input"
                          aria-label="Reference number"
                        />
                        <button
                          className="btn-track"
                          onClick={() => handleTrackApplication(scheme)}
                          disabled={applyingId === scheme.id || !currentUser}
                          aria-label={currentUser ? "Track this application" : "Login to track"}
                        >
                          {applyingId === scheme.id
                            ? <FaSpinner className="spin" aria-hidden="true" />
                            : <FaBell aria-hidden="true" />}
                          {currentUser ? "Track My Application" : "Login to Track"}
                        </button>
                        <p className="track-hint">
                          Tracking adds a 30-day follow-up reminder to your Farming Calendar.
                        </p>
                      </div>
                    ) : (
                      <div className="already-tracked">
                        <FaCheckCircle className="tracked-icon" aria-hidden="true" />
                        <span>
                          Application tracked since{" "}
                          {appData.appliedAt?.toDate
                            ? appData.appliedAt.toDate().toLocaleDateString("en-IN")
                            : "recently"}.
                        </span>
                      </div>
                    )}
                  </div>
                )}
              </div>

              <div className="scheme-footer">
                <a
                  href={scheme.link}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="btn-visit"
                >
                  Apply on Official Website <FaExternalLinkAlt size={13} aria-hidden="true" />
                </a>
              </div>
            </div>
          );
        }) : (
          <div className="no-schemes">
            <MdOutlineWaterDrop size={50} aria-hidden="true" />
            <h3>No schemes found</h3>
            <p>Try changing category filters or search keywords.</p>
          </div>
        )}
      </section>
    </div>
  );
}
