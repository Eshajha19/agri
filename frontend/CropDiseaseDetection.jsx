import { useState, useEffect } from "react";

export default function CropDiseaseDetection({ onClose }) {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Cleanup preview URL
  useEffect(() => {
    return () => preview && URL.revokeObjectURL(preview);
  }, [preview]);

  // ESC close
  useEffect(() => {
    const handleEsc = (e) => e.key === "Escape" && onClose?.();
    window.addEventListener("keydown", handleEsc);
    return () => window.removeEventListener("keydown", handleEsc);
  }, [onClose]);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // ✅ File validation
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

      const prompt = `You are an agricultural expert. Analyze this crop image and return ONLY valid JSON:

{
  "disease": "disease name or Healthy",
  "confidence": "High/Medium/Low",
  "treatment": "clear treatment steps",
  "prevention": "practical prevention tips"
}`;

      const response = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key=${apiKey}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
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
      const text = data.candidates?.[0]?.content?.parts?.[0]?.text;

      if (!text) throw new Error("Empty response from AI");

      // ✅ Safer JSON parsing
      let parsed;
      try {
        parsed = JSON.parse(text.replace(/```json|```/g, "").trim());
      } catch {
        throw new Error("Invalid AI response format");
      }

      // ✅ Validate required fields
      if (!parsed.disease || !parsed.confidence) {
        throw new Error("Incomplete analysis result");
      }

      setResult(parsed);

    } catch (err) {
      console.error(err);
      setError(err.message || "❌ Detection failed. Try again.");
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

      {/* Close Button */}
      <button
        className="close-btn"
        onClick={onClose}
        aria-label="Close"
      >
        ✕
      </button>

      {/* Header */}
      <h2 style={{ color: "#16a34a", marginBottom: "20px", fontSize: "24px", paddingRight: "40px" }}>
        🌿 Crop Disease Detection
      </h2>

      {/* Upload */}
      <input
        type="file"
        accept="image/*"
        onChange={handleImageChange}
        style={{ marginBottom: "16px", width: "100%" }}
      />

      {/* Preview */}
      {preview && (
        <img 
          src={preview} 
          alt="Selected crop preview"
          style={{ 
            width: "100%", 
            borderRadius: "12px", 
            marginBottom: "16px",
            maxHeight: "250px", 
            objectFit: "cover" 
          }} 
        />
      )}

      {/* Button */}
      <button
        onClick={handleDetect}
        disabled={!image || loading}
        style={{ 
          width: "100%", 
          padding: "12px", 
          backgroundColor: loading ? "#86efac" : "#16a34a",
          color: "white", 
          border: "none", 
          borderRadius: "8px", 
          fontSize: "16px", 
          cursor: !image || loading ? "not-allowed" : "pointer",
          opacity: !image || loading ? 0.7 : 1
        }}
      >
        {loading ? "⏳ Analyzing image..." : "🔍 Detect Disease"}
      </button>

      {/* Error */}
      {error && (
        <p style={{ color: "red", marginTop: "12px", textAlign: "center" }}>
          {error}
        </p>
      )}

      {/* Result */}
      {result && (
        <div style={{ 
          marginTop: "20px", 
          padding: "16px", 
          background: "#f0fdf4", 
          borderRadius: "12px", 
          border: "1px solid #bbf7d0" 
        }}>
          <p style={{ fontSize: "18px", fontWeight: "bold", color: "#111" }}>
            🦠 Disease: 
            <span style={{ 
              color: result.disease === "Healthy" ? "#16a34a" : "#dc2626",
              marginLeft: "6px"
            }}>
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

      {/* Empty */}
      {!image && (
        <p style={{ textAlign: "center", color: "#999", marginTop: "12px" }}>
          Upload a crop image to begin detection
        </p>
      )}
    </div>
  );
}
