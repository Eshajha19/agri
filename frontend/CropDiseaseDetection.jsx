import { useState, useEffect } from "react";
import "./CropDiseaseDetection.css";

export default function CropDiseaseDetection({ onClose }) {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    return () => preview && URL.revokeObjectURL(preview);
  }, [preview]);

  useEffect(() => {
    const handleEsc = (e) => e.key === "Escape" && onClose?.();
    window.addEventListener("keydown", handleEsc);

    return () => window.removeEventListener("keydown", handleEsc);
  }, [onClose]);

  const handleImageChange = (e) => {
    const file = e.target.files[0];

    if (!file) return;

    if (!file.type.startsWith("image/")) {
      setError("⚠️ Please upload a valid image file.");
      return;
    }

    if (file.size > 5 * 1024 * 1024) {
      setError("⚠️ Image size should be less than 5MB.");
      return;
    }

    if (preview) URL.revokeObjectURL(preview);

    setImage(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
    setError(null);
  };

  const handleDetect = async () => {
    if (!image || loading) return;

    setLoading(true);
    setError(null);

    const apiKey = import.meta.env.VITE_GEMINI_API_KEY;

    if (!apiKey) {
      setError("⚠️ API key not configured.");
      setLoading(false);
      return;
    }

    const toBase64 = (file) =>
      new Promise((res, rej) => {
        const reader = new FileReader();
        reader.onload = () => res(reader.result.split(",")[1]);
        reader.onerror = () => rej("Image reading failed");
        reader.readAsDataURL(file);
      });

    try {
      const base64 = await toBase64(image);

      const prompt = `
You are an agricultural expert.

Analyze the crop image and return ONLY valid JSON:

{
  "disease": "disease name or Healthy",
  "confidence": "High/Medium/Low",
  "severity": "Low/Moderate/High",
  "description": "short explanation",
  "treatment": [
    "step 1",
    "step 2"
  ],
  "prevention": [
    "tip 1",
    "tip 2"
  ]
}
`;

      const response = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key=${apiKey}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            contents: [
              {
                parts: [
                  { text: prompt },
                  {
                    inline_data: {
                      mime_type: image.type,
                      data: base64,
                    },
                  },
                ],
              },
            ],
          }),
        }
      );

      if (!response.ok) {
        throw new Error(`API Error (${response.status})`);
      }

      const data = await response.json();

      const text =
        data?.candidates?.[0]?.content?.parts?.[0]?.text;

      if (!text) throw new Error("Empty AI response");

      let parsed;

      try {
        parsed = JSON.parse(
          text.replace(/```json|```/g, "").trim()
        );
      } catch {
        throw new Error("Invalid AI response format");
      }

      setResult(parsed);
    } catch (err) {
      console.error(err);
      setError(err.message || "❌ Detection failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="crop-disease-container">
      {/* Header */}
      <div className="crop-header">
        <button
          onClick={onClose}
          className="close-btn"
          aria-label="Close"
        >
          ✕
        </button>

        <h2>🌿 Crop Disease Detection</h2>

        <p>
          Upload a crop leaf image and get instant AI-powered
          disease analysis.
        </p>
      </div>

      {/* Body */}
      <div className="crop-body">
        {/* Upload */}
        <label className="upload-box">
          <input
            type="file"
            accept="image/*"
            onChange={handleImageChange}
            hidden
          />

          <div className="upload-icon">📷</div>

          <p>Click to upload crop image</p>

          <span>JPG, PNG • Max 5MB</span>
        </label>

        {/* Preview */}
        {preview && (
          <div className="preview-wrapper">
            <img
              src={preview}
              alt="Crop preview"
              className="preview-image"
            />
          </div>
        )}

        {/* Button */}
        <button
          onClick={handleDetect}
          disabled={!image || loading}
          className="detect-btn"
        >
          {loading
            ? "⏳ Analyzing Crop..."
            : "🔍 Detect Disease"}
        </button>

        {/* Error */}
        {error && <div className="error-box">{error}</div>}

        {/* Result */}
        {result && (
          <div className="result-card">
            <div className="result-top">
              <h3
                className={
                  result.disease === "Healthy"
                    ? "healthy"
                    : "disease"
                }
              >
                {result.disease === "Healthy"
                  ? "✅ Healthy Crop"
                  : `🦠 ${result.disease}`}
              </h3>

              <span className="confidence-badge">
                {result.confidence} Confidence
              </span>
            </div>

            {result.description && (
              <div className="result-section">
                <h4>📖 Description</h4>
                <p>{result.description}</p>
              </div>
            )}

            {result.severity && (
              <div className="result-section">
                <h4>⚠️ Severity</h4>

                <span
                  className={`severity-badge ${result.severity.toLowerCase()}`}
                >
                  {result.severity}
                </span>
              </div>
            )}

            {result.treatment && (
              <div className="result-section">
                <h4>💊 Recommended Treatment</h4>

                <ul>
                  {Array.isArray(result.treatment) ? (
                    result.treatment.map((item, index) => (
                      <li key={index}>{item}</li>
                    ))
                  ) : (
                    <li>{result.treatment}</li>
                  )}
                </ul>
              </div>
            )}

            {result.prevention && (
              <div className="result-section">
                <h4>🛡️ Prevention Tips</h4>

                <ul>
                  {Array.isArray(result.prevention) ? (
                    result.prevention.map((item, index) => (
                      <li key={index}>{item}</li>
                    ))
                  ) : (
                    <li>{result.prevention}</li>
                  )}
                </ul>
              </div>
            )}
          </div>
        )}

        {!image && (
          <div className="empty-text">
            Upload a crop image to begin AI disease detection
          </div>
        )}
      </div>
    </div>
  );
}