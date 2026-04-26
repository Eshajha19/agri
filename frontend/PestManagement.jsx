import React, { useState, useEffect } from "react";
import "./PestManagement.css";

export default function PestManagement({ onClose }) {
  const [crop, setCrop] = useState("");
  const [symptoms, setSymptoms] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Handle escape key
  useEffect(() => {
    const handleEsc = (e) => {
      if (e.key === "Escape") {
        onClose?.();
      }
    };
    window.addEventListener("keydown", handleEsc);
    return () => window.removeEventListener("keydown", handleEsc);
  }, [onClose]);

  const handleGetAdvice = async () => {
    if (!crop.trim() || !symptoms.trim()) {
      setError("Please provide both crop name and symptoms observed.");
      return;
    }

    setLoading(true);
    setError(null);

    const apiKey = import.meta.env.VITE_GEMINI_API_KEY;
    if (!apiKey) {
      setError("API key is missing. Please configure VITE_GEMINI_API_KEY.");
      setLoading(false);
      return;
    }

    const prompt = `You are an expert agricultural scientist specializing in pest management. 
    A farmer is growing ${crop} and has noticed the following symptoms: "${symptoms}".
    
    Identify the most likely pest(s) causing this and provide an action plan.
    
    Reply ONLY in this exact JSON format:
    {
      "likelyPest": "Name of the pest(s)",
      "severity": "High/Medium/Low risk",
      "organicTreatment": "Organic or home remedies",
      "chemicalTreatment": "Recommended chemical pesticides (if severe)",
      "prevention": "Tips to prevent future infestations"
    }`;

    try {
      const response = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key=${apiKey}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            contents: [{
              parts: [{ text: prompt }]
            }]
          })
        }
      );

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.error?.message || `HTTP error ${response.status}`);
      }

      const data = await response.json();
      const text = data.candidates?.[0]?.content?.parts?.[0]?.text;

      if (!text) {
        throw new Error("Invalid response from AI service");
      }

      const clean = text.replace(/```json|```/g, "").trim();
      const parsed = JSON.parse(clean);
      setResult(parsed);
    } catch (err) {
      console.error("Pest Management error:", err);
      setError(err.message || "Failed to generate advice. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="pest-management-modal">
      <button
        className="close-btn"
        onClick={onClose}
        aria-label="Close"
      >
        ✕
      </button>

      <h2 className="modal-title">
        🐛 Pest Management Advice
      </h2>
      <p className="modal-subtitle">Describe what's affecting your crop, and our AI will provide pest control solutions.</p>

      <div className="input-group">
        <label>Crop Name</label>
        <input
          type="text"
          placeholder="e.g., Tomato, Wheat, Cotton..."
          value={crop}
          onChange={(e) => setCrop(e.target.value)}
          className="pm-input"
        />
      </div>

      <div className="input-group">
        <label>Observed Symptoms / Pest Appearance</label>
        <textarea
          placeholder="e.g., White spots on leaves, small green insects under leaves, curled yellow leaves..."
          value={symptoms}
          onChange={(e) => setSymptoms(e.target.value)}
          className="pm-textarea"
          rows={3}
        />
      </div>

      <button
        onClick={handleGetAdvice}
        disabled={loading}
        className={`pm-btn ${loading ? 'loading' : ''}`}
      >
        {loading ? "⏳ Analyzing..." : "🔍 Get Advice"}
      </button>

      {error && (
        <p className="pm-error">{error}</p>
      )}

      {result && (
        <div className="pm-result">
          <p className="result-item">
            <strong>🐛 Likely Pest:</strong> <span className="highlight-pest">{result.likelyPest}</span>
          </p>
          <p className="result-item">
            <strong>⚠️ Severity:</strong> <span className={`severity-badge ${result.severity?.toLowerCase() || 'medium'}`}>{result.severity}</span>
          </p>
          <div className="result-box organic">
            <h4>🌱 Organic Treatment</h4>
            <p>{result.organicTreatment}</p>
          </div>
          <div className="result-box chemical">
            <h4>🧪 Chemical Treatment</h4>
            <p>{result.chemicalTreatment}</p>
          </div>
          <div className="result-box prevention">
            <h4>🛡️ Prevention Tips</h4>
            <p>{result.prevention}</p>
          </div>
        </div>
      )}
    </div>
  );
}
