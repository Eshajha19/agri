import React, { useState, useEffect, useContext } from "react";
import { AuthContext } from "./AuthContext";
import { ClipLoader } from "react-spinners";
import "./InsuranceClaim.css";

const API = process.env.REACT_APP_API_URL || "";

export default function InsuranceClaim() {
  const { user } = useContext(AuthContext);
  const [autofill, setAutofill] = useState(null);
  const [loading, setLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);

  const [form, setForm] = useState({
    cropName: "",
    sowingDate: "",
    farmLocation: "",
    farmArea: "",
    village: "",
    district: "",
    incidentType: "",
    incidentDate: "",
    damageDescription: "",
    claimedAmount: "",
  });

  useEffect(() => {
    if (!user) return;
    setLoading(true);
    fetch(`${API}/api/claims/autofill/${user.uid}`)
      .then((r) => r.json())
      .then((data) => {
        setAutofill(data);
        if (data.fill_percentage > 0) {
          setForm((prev) => ({
            ...prev,
            cropName: data.crop_name || prev.cropName,
            sowingDate: data.sowing_date || prev.sowingDate,
            farmLocation: data.farm_location || prev.farmLocation,
            farmArea: data.farm_area || prev.farmArea,
            village: data.village || prev.village,
            district: data.district || prev.district,
          }));
        }
      })
      .catch(() => setAutofill({ fill_percentage: 0 }))
      .finally(() => setLoading(false));
  }, [user]);

  const handleChange = (e) =>
    setForm((prev) => ({ ...prev, [e.target.name]: e.target.value }));

  const handleSubmit = async (e) => {
    e.preventDefault();
    setSubmitting(true);
    try {
      await fetch(`${API}/api/claims/submit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          uid: user.uid,
          crop_name: form.cropName,
          sowing_date: form.sowingDate,
          farm_location: form.farmLocation,
          farm_area: parseFloat(form.farmArea),
          village: form.village,
          district: form.district,
          incident_type: form.incidentType,
          incident_date: form.incidentDate,
          damage_description: form.damageDescription,
          claimed_amount: parseFloat(form.claimedAmount),
          weather_event_ids: [],
        }),
      });
      setSubmitted(true);
    } catch {
      alert("Failed to submit claim. Try again.");
    } finally {
      setSubmitting(false);
    }
  };

  if (submitted) {
    return (
      <div className="ic-container">
        <div className="ic-card ic-success">
          <h2>Claim Submitted Successfully</h2>
          <p>Your insurance claim has been recorded. You will be notified of the assessment.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="ic-container">
      <div className="ic-card">
        <h2>File an Insurance Claim</h2>

        {loading && (
          <div className="ic-loading">
            <ClipLoader size={24} /> <span>Auto-filling from your profile…</span>
          </div>
        )}

        {autofill && autofill.fill_percentage > 0 && (
          <div className="ic-autofill-banner">
            Auto-filled {autofill.fill_percentage}% of fields from your farm profile.
            {autofill.weather_events?.length > 0 && (
              <span> &bull; {autofill.weather_events.length} recent weather event(s) found.</span>
            )}
          </div>
        )}

        <form onSubmit={handleSubmit} className="ic-form">
          <div className="ic-grid">
            <label>Crop Name<input name="cropName" value={form.cropName} onChange={handleChange} required /></label>
            <label>Sowing Date<input name="sowingDate" type="date" value={form.sowingDate} onChange={handleChange} required /></label>
            <label>Farm Location<input name="farmLocation" value={form.farmLocation} onChange={handleChange} required /></label>
            <label>Farm Area (acres)<input name="farmArea" type="number" step="0.01" value={form.farmArea} onChange={handleChange} required /></label>
            <label>Village<input name="village" value={form.village} onChange={handleChange} required /></label>
            <label>District<input name="district" value={form.district} onChange={handleChange} required /></label>
            <label>Incident Type<select name="incidentType" value={form.incidentType} onChange={handleChange} required>
              <option value="">Select…</option>
              <option value="flood">Flood</option>
              <option value="drought">Drought</option>
              <option value="hail">Hail</option>
              <option value="pest">Pest Attack</option>
              <option value="fire">Fire</option>
              <option value="other">Other</option>
            </select></label>
            <label>Incident Date<input name="incidentDate" type="date" value={form.incidentDate} onChange={handleChange} required /></label>
          </div>
          <label>Damage Description<textarea name="damageDescription" rows="3" value={form.damageDescription} onChange={handleChange} required /></label>
          <label>Claimed Amount (₹)<input name="claimedAmount" type="number" step="0.01" value={form.claimedAmount} onChange={handleChange} required /></label>
          <button type="submit" className="ic-submit" disabled={submitting}>
            {submitting ? "Submitting…" : "Submit Claim"}
          </button>
        </form>
      </div>
    </div>
  );
}
