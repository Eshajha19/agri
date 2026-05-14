import { useState, useEffect, useRef } from "react";
import * as tf from '@tensorflow/tfjs';
import * as mobilenet from '@tensorflow-models/mobilenet';
import { getDiseaseInfo, saveDiseaseHistory, getDiseaseHistory } from './utils/diseaseDatabase';
import { useTranslation } from 'react-i18next';

export default function CropDiseaseDetection({ onClose }) {
  const { t, i18n } = useTranslation();
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [model, setModel] = useState(null);
  const [modelLoading, setModelLoading] = useState(true);
  const [showHistory, setShowHistory] = useState(false);
  const [history, setHistory] = useState([]);
  const [isUsingAI, setIsUsingAI] = useState(true);
  const fileInputRef = useRef(null);
  const dropZoneRef = useRef(null);

  // Load TensorFlow.js model
  useEffect(() => {
    const loadModel = async () => {
      try {
        await tf.ready();
        const loadedModel = await mobilenet.load();
        setModel(loadedModel);
        setModelLoading(false);
      } catch (err) {
        setModelLoading(false);
      }
    };
    loadModel();
  }, []);

  // Load disease history
  useEffect(() => {
    setHistory(getDiseaseHistory());
  }, []);

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

  const handleImageChange = (file) => {
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

  const handleFileInput = (e) => {
    const file = e.target.files[0];
    handleImageChange(file);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    
    const files = e.dataTransfer.files;
    if (files && files[0]) {
      handleImageChange(files[0]);
    }
  };

  const detectWithTensorFlow = async () => {
    if (!model || !preview) return null;

    try {
      const img = new Image();
      img.src = preview;
      await new Promise((resolve) => { img.onload = resolve; });

      const predictions = await model.classify(img);
      
      // Map MobileNet predictions to disease categories
      const diseaseMapping = {
        'leaf': 'leaf_spot',
        'plant': 'leaf_blight',
        'flower': 'powdery_mildew',
        'vegetable': 'early_blight',
        'fruit': 'anthracnose'
      };

      const topPrediction = predictions[0];
      const detectedDisease = Object.keys(diseaseMapping).find(key => 
        topPrediction.className.toLowerCase().includes(key)
      ) || 'leaf_spot';

      const diseaseKey = diseaseMapping[detectedDisease] || 'leaf_spot';
      const confidence = Math.round(topPrediction.probability * 100);
      
      const diseaseInfo = getDiseaseInfo(diseaseKey, i18n.language);
      
      return {
        disease: diseaseInfo.disease,
        confidence: confidence > 70 ? 'High' : confidence > 40 ? 'Medium' : 'Low',
        confidenceScore: confidence,
        treatment: diseaseInfo.treatment,
        prevention: diseaseInfo.prevention,
        pesticides: diseaseInfo.pesticides,
        organic: diseaseInfo.organic,
        method: 'tensorflow'
      };
    } catch (error) {
      return null;
    }
  };

  const handleDetect = async () => {
    if (!image || loading) return;

    setLoading(true);
    setError(null);

    try {
      let detectionResult = null;

      if (isUsingAI) {
        // Use Gemini AI API
        const apiKey = import.meta.env.VITE_GEMINI_API_KEY;
        if (!apiKey) {
          throw new Error("⚠️ API key not configured.");
        }

        const toBase64 = (file) =>
          new Promise((res, rej) => {
            const reader = new FileReader();
            reader.onload = () => res(reader.result.split(",")[1]);
            reader.onerror = () => rej("Image reading failed");
            reader.readAsDataURL(file);
          });

        const base64 = await toBase64(image);

        const prompt = `You are an agricultural expert. Analyze this crop image and return ONLY valid JSON:

{
  "disease": "disease name or Healthy",
  "confidence": "High/Medium/Low",
  "treatment": "clear treatment steps",
  "prevention": "practical prevention tips",
  "pesticides": ["pesticide1", "pesticide2"],
  "organic": ["organic1", "organic2"]
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

        let parsed;
        try {
          parsed = JSON.parse(text.replace(/```json|```/g, "").trim());
        } catch {
          throw new Error("Invalid AI response format");
        }

        if (!parsed.disease || !parsed.confidence) {
          throw new Error("Incomplete analysis result");
        }

        detectionResult = { ...parsed, method: 'ai' };
      } else {
        // Use TensorFlow.js
        detectionResult = await detectWithTensorFlow();
        if (!detectionResult) {
          throw new Error("TensorFlow detection failed");
        }
      }

      // Save to history
      const historyEntry = saveDiseaseHistory(detectionResult);
      if (historyEntry) {
        setHistory(prev => [historyEntry, ...prev]);
      }

      setResult(detectionResult);

    } catch (err) {
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

      {/* Header with controls */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <h2 style={{ color: "#16a34a", fontSize: "24px" }}>
          🌿 Crop Disease Detection
        </h2>
        <div style={{ display: 'flex', gap: '10px' }}>
          <button
            onClick={() => setShowHistory(!showHistory)}
            style={{
              padding: '6px 12px',
              backgroundColor: '#f3f4f6',
              border: '1px solid #d1d5db',
              borderRadius: '6px',
              fontSize: '12px',
              cursor: 'pointer'
            }}
          >
            📋 History ({history.length})
          </button>
          <button
            className="close-btn"
            onClick={onClose}
            aria-label="Close"
          >
            ✕
          </button>
        </div>
      </div>

      {/* Detection Method Toggle */}
      <div style={{ marginBottom: '16px', padding: '12px', backgroundColor: '#f9fafb', borderRadius: '8px' }}>
        <label style={{ fontSize: '14px', fontWeight: '500', marginBottom: '8px', display: 'block' }}>
          Detection Method:
        </label>
        <div style={{ display: 'flex', gap: '10px' }}>
          <button
            onClick={() => setIsUsingAI(true)}
            style={{
              padding: '8px 16px',
              backgroundColor: isUsingAI ? '#16a34a' : '#e5e7eb',
              color: isUsingAI ? 'white' : '#374151',
              border: 'none',
              borderRadius: '6px',
              fontSize: '14px',
              cursor: 'pointer'
            }}
          >
            🤖 AI Analysis
          </button>
          <button
            onClick={() => setIsUsingAI(false)}
            disabled={modelLoading}
            style={{
              padding: '8px 16px',
              backgroundColor: !isUsingAI ? '#16a34a' : '#e5e7eb',
              color: !isUsingAI ? 'white' : '#374151',
              border: 'none',
              borderRadius: '6px',
              fontSize: '14px',
              cursor: modelLoading ? 'not-allowed' : 'pointer',
              opacity: modelLoading ? 0.5 : 1
            }}
          >
            🧠 TensorFlow {modelLoading ? '(Loading...)' : ''}
          </button>
        </div>
      </div>

      {/* Upload Area with Drag & Drop */}
      <div
        ref={dropZoneRef}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        style={{
          border: `2px dashed ${isDragging ? '#16a34a' : '#d1d5db'}`,
          borderRadius: '12px',
          padding: '32px',
          textAlign: 'center',
          cursor: 'pointer',
          backgroundColor: isDragging ? '#f0fdf4' : '#f9fafb',
          marginBottom: '16px',
          transition: 'all 0.3s ease'
        }}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileInput}
          style={{ display: 'none' }}
        />
        <div style={{ fontSize: '48px', marginBottom: '12px' }}>📸</div>
        <p style={{ fontSize: '16px', color: '#374151', marginBottom: '8px' }}>
          {isDragging ? 'Drop your image here' : 'Click to upload or drag & drop'}
        </p>
        <p style={{ fontSize: '14px', color: '#6b7280' }}>
          Supports: JPG, PNG, GIF (Max 5MB)
        </p>
      </div>

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
          padding: "20px", 
          background: "#f0fdf4", 
          borderRadius: "12px", 
          border: "1px solid #bbf7d0" 
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            <h3 style={{ fontSize: "18px", fontWeight: "bold", color: "#111", margin: 0 }}>
              🦠 Disease: 
              <span style={{ 
                color: result.disease.toLowerCase().includes('healthy') ? "#16a34a" : "#dc2626",
                marginLeft: "6px"
              }}>
                {result.disease}
              </span>
            </h3>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ fontSize: '14px', color: '#6b7280' }}>Confidence:</span>
              <div style={{
                width: '60px',
                height: '8px',
                backgroundColor: '#e5e7eb',
                borderRadius: '4px',
                overflow: 'hidden'
              }}>
                <div style={{
                  width: `${result.confidenceScore || (result.confidence === 'High' ? 80 : result.confidence === 'Medium' ? 50 : 20)}%`,
                  height: '100%',
                  backgroundColor: result.confidence === 'High' ? '#16a34a' : result.confidence === 'Medium' ? '#f59e0b' : '#ef4444',
                  transition: 'width 0.3s ease'
                }} />
              </div>
              <span style={{ fontSize: '14px', fontWeight: 'bold', color: '#374151' }}>
                {result.confidence}
              </span>
            </div>
          </div>

          <div style={{ marginBottom: '16px' }}>
            <h4 style={{ fontSize: '16px', fontWeight: '600', color: '#111', marginBottom: '8px' }}>
              💊 Treatment:
            </h4>
            <p style={{ color: '#555', lineHeight: '1.5', margin: 0 }}>
              {result.treatment}
            </p>
          </div>

          <div style={{ marginBottom: '16px' }}>
            <h4 style={{ fontSize: '16px', fontWeight: '600', color: '#111', marginBottom: '8px' }}>
              🛡️ Prevention:
            </h4>
            <p style={{ color: '#555', lineHeight: '1.5', margin: 0 }}>
              {result.prevention}
            </p>
          </div>

          {(result.pesticides || result.organic) && (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
              {result.pesticides && (
                <div>
                  <h4 style={{ fontSize: '14px', fontWeight: '600', color: '#111', marginBottom: '8px' }}>
                    🧪 Recommended Pesticides:
                  </h4>
                  <ul style={{ margin: 0, paddingLeft: '16px' }}>
                    {result.pesticides.map((pesticide, index) => (
                      <li key={index} style={{ color: '#555', fontSize: '14px', marginBottom: '4px' }}>
                        {pesticide}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {result.organic && (
                <div>
                  <h4 style={{ fontSize: '14px', fontWeight: '600', color: '#111', marginBottom: '8px' }}>
                    🌿 Organic Options:
                  </h4>
                  <ul style={{ margin: 0, paddingLeft: '16px' }}>
                    {result.organic.map((option, index) => (
                      <li key={index} style={{ color: '#555', fontSize: '14px', marginBottom: '4px' }}>
                        {option}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}

          <div style={{ marginTop: '16px', fontSize: '12px', color: '#6b7280', textAlign: 'center' }}>
            Detection method: {result.method === 'ai' ? '🤖 AI Analysis' : '🧠 TensorFlow.js'}
          </div>
        </div>
      )}

      {/* Disease History */}
      {showHistory && (
        <div style={{
          marginTop: '20px',
          padding: '16px',
          backgroundColor: '#f8fafc',
          borderRadius: '12px',
          border: '1px solid #e2e8f0'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
            <h3 style={{ fontSize: '16px', fontWeight: '600', color: '#111', margin: 0 }}>
              � Detection History
            </h3>
            {history.length > 0 && (
              <button
                onClick={() => {
                  if (confirm('Clear all disease history?')) {
                    localStorage.removeItem('diseaseHistory');
                    setHistory([]);
                  }
                }}
                style={{
                  padding: '4px 8px',
                  fontSize: '12px',
                  backgroundColor: '#ef4444',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer'
                }}
              >
                Clear All
              </button>
            )}
          </div>
          
          {history.length === 0 ? (
            <p style={{ textAlign: 'center', color: '#6b7280', fontSize: '14px' }}>
              No detection history yet
            </p>
          ) : (
            <div style={{ maxHeight: '200px', overflowY: 'auto' }}>
              {history.map((entry) => (
                <div
                  key={entry.id}
                  style={{
                    padding: '8px',
                    marginBottom: '8px',
                    backgroundColor: 'white',
                    borderRadius: '6px',
                    fontSize: '12px'
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontWeight: '500', color: '#111' }}>
                      {entry.disease}
                    </span>
                    <span style={{ color: '#6b7280' }}>
                      {new Date(entry.timestamp).toLocaleDateString()}
                    </span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '4px' }}>
                    <span style={{
                      padding: '2px 6px',
                      backgroundColor: entry.confidence === 'High' ? '#dcfce7' : entry.confidence === 'Medium' ? '#fef3c7' : '#fee2e2',
                      color: entry.confidence === 'High' ? '#166534' : entry.confidence === 'Medium' ? '#92400e' : '#991b1b',
                      borderRadius: '4px',
                      fontSize: '10px'
                    }}>
                      {entry.confidence}
                    </span>
                    <span style={{ color: '#6b7280', fontSize: '10px' }}>
                      {entry.method === 'ai' ? '🤖' : '🧠'}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
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
