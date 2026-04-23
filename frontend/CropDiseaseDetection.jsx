import { useState, useEffect } from "react";

export default function CropDiseaseDetection({ onClose }) {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Cleanup object URL on unmount or when image changes
  useEffect(() => {
    return () => {
      if (preview) {
        URL.revokeObjectURL(preview);
      }
    };
  }, [preview]);

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

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    // Revoke previous preview URL to avoid memory leak
    if (preview) {
      URL.revokeObjectURL(preview);
    }
    setImage(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
    setError(null);
  };

  const handleDetect = async () => {
    if (!image) return;
    setLoading(true);
    setError(null);

    const apiKey = import.meta.env.VITE_GEMINI_API_KEY;
    if (!apiKey) {
      setError("API key is missing. Please configure VITE_GEMINI_API_KEY.");
      setLoading(false);
      return;
    }

    const toBase64 = (file) =>
      new Promise((res, rej) => {
        const reader = new FileReader();
        reader.onload = () => res(reader.result.split(",")[1]);
        reader.onerror = rej;
        reader.readAsDataURL(file);
      });

    let base64;
    try {
      base64 = await toBase64(image);
    } catch (error) {
      console.error("Image read error:", error);
      setError("Failed to read image. Please try another file.");
      setLoading(false);
      return;
    }

    const prompt = `You are an expert agricultural scientist. 
    Analyze this crop image and identify:
    1. Disease name (if any)
    2. Confidence level (High/Medium/Low)
    3. Suggested treatment
    4. Prevention tips
    
    Reply in this exact JSON format:
    {
      "disease": "disease name or Healthy",
      "confidence": "High/Medium/Low",
      "treatment": "treatment steps",
      "prevention": "prevention tips"
    }`;

    try {
      const response = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${apiKey}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            contents: [{
              parts: [
                { text: prompt },
                { inline_data: { mime_type: image.type, data: base64 } }
              ]
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
      console.error("Disease detection error:", err);
      setError(err.message || "Detection failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ 
      maxWidth: "500px", 
      margin: "40px auto", 
      padding: "24px", 
      background: "#fff", 
      borderRadius: "16px", 
      boxShadow: "0 4px 20px rgba(0,0,0,0.1)",
      position: "relative"
    }}>
      <button
        className="close-btn"
        onClick={onClose}
        aria-label="Close"
      >
        ✕
      </button>

      <h2 style={{ color: "#16a34a", marginBottom: "20px", fontSize: "24px", paddingRight: "40px" }}>
        🌿 Crop Disease Detection
      </h2>

      <input
        type="file"
        accept="image/*"
        onChange={handleImageChange}
        style={{ marginBottom: "16px", width: "100%" }}
      />

      {preview && (
        <img src={preview} alt="Preview"
          style={{ width: "100%", borderRadius: "12px", marginBottom: "16px", 
            maxHeight: "250px", objectFit: "cover" }} />
      )}

      <button
        onClick={handleDetect}
        disabled={!image || loading}
        style={{ width: "100%", padding: "12px", backgroundColor: loading ? "#86efac" : "#16a34a",
          color: "white", border: "none", borderRadius: "8px", 
          fontSize: "16px", cursor: image ? "pointer" : "not-allowed" }}>
        {loading ? "⏳ Detecting..." : "🔍 Detect Disease"}
      </button>

      {error && (
        <p style={{ color: "red", marginTop: "12px", textAlign: "center" }}>{error}</p>
      )}

      {result && (
        <div style={{ marginTop: "20px", padding: "16px", 
          background: "#f0fdf4", borderRadius: "12px", border: "1px solid #bbf7d0" }}>
          <p style={{ fontSize: "18px", fontWeight: "bold", color: "#111" }}>
            🦠 Disease: <span style={{ color: result.disease === "Healthy" ? "#16a34a" : "#dc2626" }}>
              {result.disease}
            </span>
          </p>
          <p style={{ color: "#555", marginTop: "8px" }}>
            📊 Confidence: <strong>{result.confidence}</strong>
          </p>
          <p style={{ color: "#555", marginTop: "8px" }}>
            💊 Treatment: {result.treatment}
          </p>
          <p style={{ color: "#555", marginTop: "8px" }}>
            🛡️ Prevention: {result.prevention}
          </p>
        </div>
      )}
    </div>
  );
}
