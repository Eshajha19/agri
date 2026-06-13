import React, { useState } from "react";
import "./CompensationCalculator.css";

const API = process.env.REACT_APP_API_URL || "";

const CROPS = ["Paddy", "Wheat", "Maize", "Cotton", "Sugarcane", "Soybean", "Groundnut", "Potato", "Tomato", "Pulses", "Vegetables", "Fruits"];
const POLICIES = ["Basic", "Standard", "Premium", "Comprehensive"];

export default function CompensationCalculator() {
  const [form, setForm] = useState({ cropType: "", area: "", damage: "", policy: "" });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleChange = (e) =>
    setForm((prev) => ({ ...prev, [e.target.name]: e.target.value }));

  const canCalculate = form.cropType && form.area && form.damage && form.policy;

  const handleCalculate = async (e) => {
    e.preventDefault();
    if (!canCalculate) return;
    setLoading(true);
    setError("");
    try {
      const res = await fetch(`${API}/api/compensation/calculate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          crop_type: form.cropType,
          cultivated_area: parseFloat(form.area),
          damage_percentage: parseFloat(form.damage),
          policy_type: form.policy,
        }),
      });
      if (!res.ok) { setError("Calculation failed"); setLoading(false); return; }
      setResult(await res.json());
    } catch {
      setError("Network error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="cc-container">
      <div className="cc-card">
        <h2>Compensation Calculator</h2>
        <p className="cc-subtitle">Estimate your insurance payout before filing a claim.</p>

        {error && <div className="cc-error">{error}</div>}

        <form onSubmit={handleCalculate} className="cc-form">
          <div className="cc-grid">
            <label>Crop Type
              <select name="cropType" value={form.cropType} onChange={handleChange} required>
                <option value="">Select…</option>
                {CROPS.map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
            </label>
            <label>Cultivated Area (acres)
              <input name="area" type="number" step="0.01" min="0.01" placeholder="e.g. 2.5" value={form.area} onChange={handleChange} required />
            </label>
            <label>Damage Percentage (%)
              <input name="damage" type="number" step="1" min="0" max="100" placeholder="e.g. 40" value={form.damage} onChange={handleChange} required />
            </label>
            <label>Policy Type
              <select name="policy" value={form.policy} onChange={handleChange} required>
                <option value="">Select…</option>
                {POLICIES.map((p) => <option key={p} value={p.toLowerCase()}>{p}</option>)}
              </select>
            </label>
          </div>
          <button type="submit" className="cc-submit" disabled={!canCalculate || loading}>
            {loading ? "Calculating…" : "Calculate Compensation"}
          </button>
        </form>

        {result && (
          <div className="cc-result">
            <div className="cc-amount">
              <span className="cc-amount-label">Estimated Compensation</span>
              <span className="cc-amount-value">₹{result.estimated_compensation.toLocaleString("en-IN", { maximumFractionDigits: 2 })}</span>
            </div>
            <div className="cc-breakdown">
              <h3>Calculation Breakdown</h3>
              <table>
                <tbody>
                  {result.breakdown.map((row, i) => (
                    <tr key={i}>
                      <td className="cc-bd-label">{row.label}</td>
                      <td className="cc-bd-value">{row.value}</td>
                    </tr>
                  ))}
                  <tr className="cc-bd-total">
                    <td>Estimated Compensation</td>
                    <td>₹{result.estimated_compensation.toLocaleString("en-IN", { maximumFractionDigits: 2 })}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
