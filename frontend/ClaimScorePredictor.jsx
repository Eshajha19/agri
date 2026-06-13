import React, { useState, useContext, useEffect, useRef } from "react";
import { AuthContext } from "./AuthContext";
import { ClipLoader } from "react-spinners";
import "./ClaimScorePredictor.css";

const API = process.env.REACT_APP_API_URL || "";

export default function ClaimScorePredictor() {
  const { user } = useContext(AuthContext);
  const initialized = useRef(false);

  const [step, setStep] = useState("form");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const [form, setForm] = useState({
    incidentType: "",
    incidentDate: "",
    damageDescription: "",
    claimedAmount: "",
  });

  const handleChange = (e) =>
    setForm((prev) => ({ ...prev, [e.target.name]: e.target.value }));

  useEffect(() => {
    if (initialized.current) return;
    initialized.current = true;
  }, []);

  const handlePredict = async (e) => {
    e.preventDefault();
    if (!user) { setError("Please log in first"); return; }
    setLoading(true);
    setError("");
    try {
      const res = await fetch(`${API}/api/claims/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          uid: user.uid,
          incident_type: form.incidentType,
          incident_date: form.incidentDate,
          damage_description: form.damageDescription,
          claimed_amount: parseFloat(form.claimedAmount),
        }),
      });
      if (!res.ok) { setError("Prediction failed"); setLoading(false); return; }
      const data = await res.json();
      setResult(data);
      setStep("result");
    } catch {
      setError("Network error");
    } finally {
      setLoading(false);
    }
  };

  const getColor = (score) => {
    if (score >= 75) return "#2e7d32";
    if (score >= 50) return "#f9a825";
    if (score >= 25) return "#e65100";
    return "#c62828";
  };

  const getLabelBadge = (prob) => {
    const map = {
      High: "badge-high",
      Moderate: "badge-moderate",
      Low: "badge-low",
      "Very Low": "badge-verylow",
    };
    return map[prob] || "badge-low";
  };

  return (
    <div className="csp-container">
      <div className="csp-card">
        <h2>Claim Success Predictor</h2>
        <p className="csp-subtitle">Estimate your claim approval probability before submitting.</p>

        {error && <div className="csp-error">{error}</div>}

        {step === "form" && (
          <form onSubmit={handlePredict} className="csp-form">
            <div className="csp-grid">
              <label>Incident Type
                <select name="incidentType" value={form.incidentType} onChange={handleChange} required>
                  <option value="">Select…</option>
                  <option value="flood">Flood</option>
                  <option value="drought">Drought</option>
                  <option value="hail">Hail</option>
                  <option value="pest">Pest Attack</option>
                  <option value="fire">Fire</option>
                  <option value="other">Other</option>
                </select>
              </label>
              <label>Incident Date
                <input name="incidentDate" type="date" value={form.incidentDate} onChange={handleChange} required />
              </label>
            </div>
            <label>Damage Description
              <textarea name="damageDescription" rows="3" value={form.damageDescription} onChange={handleChange} required />
            </label>
            <label>Claimed Amount (₹)
              <input name="claimedAmount" type="number" step="0.01" value={form.claimedAmount} onChange={handleChange} required />
            </label>
            <button type="submit" className="csp-submit" disabled={loading}>
              {loading ? <><ClipLoader size={16} color="#fff" /> Predicting…</> : "Predict Approval"}
            </button>
          </form>
        )}

        {step === "result" && result && (
          <div className="csp-result">
            <div className="csp-score-ring" style={{ "--score-color": getColor(result.claim_strength_score) }}>
              <svg viewBox="0 0 120 120">
                <circle cx="60" cy="60" r="52" fill="none" stroke="#e0e0e0" strokeWidth="10" />
                <circle
                  cx="60" cy="60" r="52"
                  fill="none" stroke="var(--score-color)"
                  strokeWidth="10"
                  strokeLinecap="round"
                  strokeDasharray={`${result.claim_strength_score * 3.267} 326.7`}
                  transform="rotate(-90 60 60)"
                />
              </svg>
              <div className="csp-score-text">
                <span className="csp-score-value">{result.claim_strength_score}%</span>
                <span className="csp-score-label">Strength</span>
              </div>
            </div>

            <div className={`csp-badge ${getLabelBadge(result.approval_probability)}`}>
              {result.approval_label}
            </div>

            <div className="csp-factors">
              <h3>Factors Considered</h3>
              {result.factors.map((f, i) => (
                <div key={i} className="csp-factor-row">
                  <div className="csp-factor-header">
                    <span className="csp-factor-name">{f.name}</span>
                    <span className="csp-factor-score" style={{ color: getColor(f.score) }}>
                      {f.score}/100
                    </span>
                  </div>
                  <div className="csp-factor-bar">
                    <div className="csp-factor-fill" style={{ width: `${f.score}%`, background: getColor(f.score) }} />
                  </div>
                  <ul className="csp-factor-details">
                    {f.details.map((d, j) => (
                      <li key={j}>{d}</li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>

            <button className="csp-back" onClick={() => setStep("form")}>
              Adjust &amp; Re-predict
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
