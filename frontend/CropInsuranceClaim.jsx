import React, { useState, useRef, useCallback } from "react";
import "./CropInsuranceClaim.css";
import apiClient from "./services/api";

// ── Constants ──────────────────────────────────────────────────────────────

const STEPS = [
  { label: "Farm Details", icon: "🌾" },
  { label: "Upload Photos", icon: "📷" },
  { label: "AI Assessment", icon: "🤖" },
  { label: "Claim Report", icon: "📋" },
];

const SEASONS = ["Kharif", "Rabi", "Zaid", "Annual"];
const DAMAGE_CAUSES = [
  "Flood",
  "Drought",
  "Hailstorm",
  "Cyclone",
  "Pest Attack",
  "Disease Outbreak",
  "Unseasonal Rain",
  "Fire",
  "Landslide",
  "Other",
];

const SEVERITY_ICONS = { Low: "✅", Medium: "⚠️", High: "🔴" };

// ── Component ──────────────────────────────────────────────────────────────

export default function CropInsuranceClaim() {
  const [step, setStep] = useState(0);
  const [form, setForm] = useState({
    farmer_name: "",
    crop_type: "",
    season: "Kharif",
    location: "",
    farm_area: "",
    damage_cause: "Flood",
  });
  const [images, setImages] = useState([]); // [{file, preview}]
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [claimResult, setClaimResult] = useState(null); // full API response
  const fileInputRef = useRef(null);

  // ── Form helpers ───────────────────────────────────────────────────────

  const handleChange = (e) =>
    setForm((prev) => ({ ...prev, [e.target.name]: e.target.value }));

  const formValid = () =>
    form.farmer_name.trim() &&
    form.crop_type.trim() &&
    form.location.trim() &&
    form.farm_area.trim();

  // ── Image handling ─────────────────────────────────────────────────────

  const addFiles = useCallback((files) => {
    const valid = Array.from(files).filter((f) =>
      f.type.startsWith("image/")
    );
    const remaining = 5 - images.length;
    const toAdd = valid.slice(0, remaining).map((file) => ({
      file,
      preview: URL.createObjectURL(file),
    }));
    setImages((prev) => [...prev, ...toAdd]);
  }, [images.length]);

  const removeImage = (idx) => {
    URL.revokeObjectURL(images[idx].preview);
    setImages((prev) => prev.filter((_, i) => i !== idx));
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    addFiles(e.dataTransfer.files);
  };

  // ── Step navigation ────────────────────────────────────────────────────

  const goNext = () => {
    setError("");
    if (step === 0 && !formValid()) {
      setError("Please fill in all required fields.");
      return;
    }
    if (step === 1 && images.length === 0) {
      setError("Please upload at least one damage photo.");
      return;
    }
    if (step === 1) {
      submitClaim();
      return;
    }
    setStep((s) => s + 1);
  };

  const goBack = () => setStep((s) => Math.max(0, s - 1));

  // ── API call ───────────────────────────────────────────────────────────

  const submitClaim = async () => {
    setLoading(true);
    setError("");
    setStep(2); // jump to loading/assessment step

    try {
      const fd = new FormData();
      Object.entries(form).forEach(([k, v]) => fd.append(k, v));
      images.forEach(({ file }) => fd.append("images", file));

      const resp = await apiClient.post("/api/insurance/claim", fd, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setClaimResult(resp.data);
      setStep(3); // success / report step
    } catch (err) {
      const msg =
        err?.response?.data?.detail ||
        "Failed to submit claim. Please try again.";
      setError(typeof msg === "string" ? msg : JSON.stringify(msg));
      setStep(1); // go back to upload step on error
    } finally {
      setLoading(false);
    }
  };

  const downloadPDF = async () => {
    if (!claimResult?.claim?.claim_id) return;
    try {
      const resp = await apiClient.get(
        `/api/insurance/claim/${claimResult.claim.claim_id}/export`,
        { responseType: "blob" }
      );
      const url = URL.createObjectURL(new Blob([resp.data]));
      const a = document.createElement("a");
      a.href = url;
      a.download = `InsuranceClaim_${claimResult.claim.claim_id}.pdf`;
      a.click();
      URL.revokeObjectURL(url);
    } catch {
      setError("Failed to download PDF. Please try again.");
    }
  };

  const startNew = () => {
    setStep(0);
    setForm({
      farmer_name: "",
      crop_type: "",
      season: "Kharif",
      location: "",
      farm_area: "",
      damage_cause: "Flood",
    });
    images.forEach(({ preview }) => URL.revokeObjectURL(preview));
    setImages([]);
    setClaimResult(null);
    setError("");
  };

  // ── Render ─────────────────────────────────────────────────────────────

  return (
    <div className="insurance-page">
      {/* Hero */}
      <div className="insurance-hero">
        <div className="hero-badge">🛡️ Crop Protection</div>
        <h1>Crop Insurance Claim Assistant</h1>
        <p>
          Document crop damage with AI-powered analysis and generate a
          structured insurance claim report in minutes.
        </p>
      </div>

      {/* Stepper */}
      <div className="insurance-stepper">
        {STEPS.map((s, i) => (
          <div
            key={i}
            className={`step-item ${i === step ? "active" : ""} ${
              i < step ? "completed" : ""
            }`}
          >
            <div className="step-circle">
              {i < step ? "✓" : i + 1}
            </div>
            <span className="step-label">
              {s.icon} {s.label}
            </span>
          </div>
        ))}
      </div>

      <div className="insurance-container">
        {/* Error Banner */}
        {error && (
          <div className="error-banner">⚠️ {error}</div>
        )}

        {/* ── Step 0: Farm Details ── */}
        {step === 0 && (
          <div className="insurance-card">
            <h2 className="card-title">🌾 Farm & Crop Details</h2>
            <div className="form-grid">
              <div className="form-group">
                <label className="form-label">Farmer Name *</label>
                <input
                  id="farmer_name"
                  name="farmer_name"
                  className="form-input"
                  placeholder="Enter your full name"
                  value={form.farmer_name}
                  onChange={handleChange}
                  maxLength={100}
                />
              </div>
              <div className="form-group">
                <label className="form-label">Crop Type *</label>
                <input
                  id="crop_type"
                  name="crop_type"
                  className="form-input"
                  placeholder="e.g. Rice, Wheat, Cotton"
                  value={form.crop_type}
                  onChange={handleChange}
                  maxLength={50}
                />
              </div>
              <div className="form-group">
                <label className="form-label">Season</label>
                <select
                  id="season"
                  name="season"
                  className="form-select"
                  value={form.season}
                  onChange={handleChange}
                >
                  {SEASONS.map((s) => (
                    <option key={s}>{s}</option>
                  ))}
                </select>
              </div>
              <div className="form-group">
                <label className="form-label">Cause of Damage *</label>
                <select
                  id="damage_cause"
                  name="damage_cause"
                  className="form-select"
                  value={form.damage_cause}
                  onChange={handleChange}
                >
                  {DAMAGE_CAUSES.map((c) => (
                    <option key={c}>{c}</option>
                  ))}
                </select>
              </div>
              <div className="form-group full">
                <label className="form-label">Farm Location *</label>
                <input
                  id="location"
                  name="location"
                  className="form-input"
                  placeholder="Village / District / State"
                  value={form.location}
                  onChange={handleChange}
                  maxLength={150}
                />
              </div>
              <div className="form-group">
                <label className="form-label">Farm Area *</label>
                <input
                  id="farm_area"
                  name="farm_area"
                  className="form-input"
                  placeholder="e.g. 2.5 acres"
                  value={form.farm_area}
                  onChange={handleChange}
                  maxLength={50}
                />
              </div>
            </div>
            <div className="btn-actions">
              <button
                id="btn-next-step0"
                className="btn-primary"
                onClick={goNext}
              >
                Next: Upload Photos →
              </button>
            </div>
          </div>
        )}

        {/* ── Step 1: Upload Photos ── */}
        {step === 1 && (
          <div className="insurance-card">
            <h2 className="card-title">📷 Upload Damage Photos</h2>
            <p style={{ fontSize: "0.88rem", color: "#94a3b8", marginBottom: "1.25rem" }}>
              Upload up to 5 clear photos of the damaged crop. The AI will
              analyze each image to estimate damage severity.
            </p>

            <div
              className={`upload-zone ${dragging ? "dragging" : ""}`}
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                multiple
                style={{ display: "none" }}
                onChange={(e) => addFiles(e.target.files)}
                id="image-upload-input"
              />
              <div className="upload-icon">📸</div>
              <h3>Drag & drop photos here</h3>
              <p>or click to browse — JPEG, PNG, WebP (max 10 MB each)</p>
              <p style={{ marginTop: "0.35rem", color: "#10b981", fontSize: "0.8rem" }}>
                {images.length}/5 photos selected
              </p>
            </div>

            {images.length > 0 && (
              <div className="image-previews">
                {images.map((img, i) => (
                  <div key={i} className="preview-item">
                    <img src={img.preview} alt={`Damage photo ${i + 1}`} />
                    <button
                      className="preview-remove"
                      onClick={(e) => { e.stopPropagation(); removeImage(i); }}
                      aria-label={`Remove photo ${i + 1}`}
                      id={`remove-photo-${i}`}
                    >
                      ✕
                    </button>
                  </div>
                ))}
              </div>
            )}

            <div className="btn-actions">
              <button id="btn-back-step1" className="btn-secondary" onClick={goBack}>
                ← Back
              </button>
              <button
                id="btn-submit-claim"
                className="btn-primary"
                onClick={goNext}
                disabled={images.length === 0}
              >
                🤖 Analyze & Submit Claim
              </button>
            </div>
          </div>
        )}

        {/* ── Step 2: AI Assessment (loading) ── */}
        {step === 2 && (
          <div className="insurance-card">
            <div className="loading-overlay">
              <div className="spinner" />
              <p>Analyzing damage photos with AI...</p>
              <p style={{ fontSize: "0.8rem", color: "#64748b" }}>
                This may take a few seconds
              </p>
            </div>
          </div>
        )}

        {/* ── Step 3: Report + Result ── */}
        {step === 3 && claimResult && (() => {
          const { claim, applicable_schemes } = claimResult;
          const sev = claim.damage_severity;
          return (
            <>
              {/* Success header */}
              <div className="insurance-card">
                <div className="success-header">
                  <div className="success-icon">🎉</div>
                  <h2>Claim Submitted Successfully</h2>
                  <p>Your claim has been recorded. Reference ID:</p>
                  <div className="claim-id-badge">
                    🔖 {claim.claim_id}
                  </div>
                </div>

                {/* Damage Assessment Result */}
                <h3 className="card-title" style={{ marginBottom: "1rem" }}>
                  🤖 AI Damage Assessment
                </h3>
                <div className="damage-result">
                  <div className={`severity-badge-large ${sev}`}>
                    <span className="severity-icon">{SEVERITY_ICONS[sev]}</span>
                    <span className="severity-label">Severity</span>
                    <span className="severity-text">{sev}</span>
                  </div>
                  <div>
                    <div className="damage-stats">
                      <div className="stat-box">
                        <div className="stat-label">Estimated Loss</div>
                        <div className="stat-value">
                          {claim.estimated_loss_pct}
                          <span className="stat-unit">%</span>
                        </div>
                        <div className="progress-bar-wrap">
                          <div className="progress-bar-track">
                            <div
                              className={`progress-bar-fill ${sev}`}
                              style={{ width: `${claim.estimated_loss_pct}%` }}
                            />
                          </div>
                        </div>
                      </div>
                      <div className="stat-box">
                        <div className="stat-label">AI Confidence</div>
                        <div className="stat-value">
                          {claim.confidence_score}
                          <span className="stat-unit">%</span>
                        </div>
                        <div className="progress-bar-wrap">
                          <div className="progress-bar-track">
                            <div
                              className="progress-bar-fill Medium"
                              style={{ width: `${claim.confidence_score}%` }}
                            />
                          </div>
                        </div>
                      </div>
                    </div>
                    {claim.damage_description && (
                      <div className="damage-description">
                        <strong>Damage Description</strong>
                        {claim.damage_description}
                      </div>
                    )}
                    {claim.treatment_hint && (
                      <div className="damage-description" style={{ marginTop: "0.75rem" }}>
                        <strong>💡 Recovery Guidance</strong>
                        {claim.treatment_hint}
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Claim Summary */}
              <div className="insurance-card">
                <h3 className="card-title">📋 Claim Summary</h3>
                <div className="claim-summary-grid">
                  {[
                    ["Farmer", claim.farmer_name],
                    ["Crop", claim.crop_type],
                    ["Season", claim.season],
                    ["Location", claim.location],
                    ["Farm Area", claim.farm_area],
                    ["Damage Cause", claim.damage_cause],
                    ["Photos Submitted", `${claim.image_count} image(s)`],
                    ["Submitted At", claim.submitted_at],
                    ["Status", claim.status],
                  ].map(([label, val]) => (
                    <div key={label} className="summary-row">
                      <span className="summary-label">{label}</span>
                      <span className="summary-value">{val}</span>
                    </div>
                  ))}
                </div>

                <div className="btn-actions">
                  <button id="btn-new-claim" className="btn-secondary" onClick={startNew}>
                    + New Claim
                  </button>
                  <button id="btn-download-pdf" className="btn-primary" onClick={downloadPDF}>
                    ⬇️ Download PDF Report
                  </button>
                </div>
              </div>

              {/* Insurance Schemes */}
              {applicable_schemes?.length > 0 && (
                <div className="insurance-card">
                  <h3 className="card-title">🏛️ Applicable Insurance Schemes</h3>
                  <div className="schemes-grid">
                    {applicable_schemes.map((scheme) => (
                      <div key={scheme.id} className="scheme-card">
                        <div className="scheme-header">
                          <span className="scheme-icon">{scheme.icon}</span>
                          <span className="scheme-name">{scheme.name}</span>
                        </div>
                        <div className="scheme-field">
                          <div className="scheme-field-label">Coverage</div>
                          <div className="scheme-field-value">{scheme.coverage}</div>
                        </div>
                        <div className="scheme-field">
                          <div className="scheme-field-label">Farmer Premium</div>
                          <div className="scheme-field-value">{scheme.premium_farmer}</div>
                        </div>
                        <div className="scheme-field">
                          <div className="scheme-field-label">Eligibility</div>
                          <div className="scheme-field-value">{scheme.eligibility}</div>
                        </div>
                        <div className="scheme-field">
                          <div className="scheme-field-label">Claim Process</div>
                          <div className="scheme-field-value">{scheme.claim_process}</div>
                        </div>
                        <a
                          href={scheme.portal}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="scheme-link"
                          id={`scheme-link-${scheme.id}`}
                        >
                          🔗 Apply / Learn More ↗
                        </a>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          );
        })()}
      </div>
    </div>
  );
}
