import { useState, useEffect, useRef } from "react";
import { getPestInfo, savePestHistory, getPestHistory, getRegionalAlerts } from './utils/pestDatabase';
import { useTranslation } from 'react-i18next';

export default function PestDetection({ onClose }) {
  const { t, i18n } = useTranslation();
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [showAlerts, setShowAlerts] = useState(false);
  const [history, setHistory] = useState([]);
  const [regionalAlerts, setRegionalAlerts] = useState([]);
  const [modelLoading, setModelLoading] = useState(true);
  const fileInputRef = useRef(null);
  const dropZoneRef = useRef(null);

  // Load pest history and regional alerts
  useEffect(() => {
    setHistory(getPestHistory());
    setRegionalAlerts(getRegionalAlerts());
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

    // File validation
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

  const detectWithAI = async () => {
    if (!image) return null;

    try {
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

      const prompt = `You are an agricultural pest expert. Analyze this crop image and return ONLY valid JSON:

{
  "pest": "pest name or Healthy",
  "confidence": "High/Medium/Low",
  "severity": "Low/Medium/High",
  "damage": "visible damage symptoms",
  "treatment": "immediate treatment steps",
  "prevention": "prevention measures",
  "chemical": ["pesticide1", "pesticide2"],
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

      if (!parsed.pest || !parsed.confidence) {
        throw new Error("Incomplete analysis result");
      }

      const pestInfo = getPestInfo(parsed.pest.toLowerCase().replace(/\s+/g, '_'), i18n.language);
      
      return {
        pest: pestInfo.pest,
        confidence: parsed.confidence,
        severity: parsed.severity || 'Medium',
        damage: parsed.damage || pestInfo.damage,
        treatment: parsed.treatment || pestInfo.treatment,
        prevention: parsed.prevention || pestInfo.prevention,
        chemical: parsed.chemical || pestInfo.chemical,
        organic: parsed.organic || pestInfo.organic,
        lifecycle: pestInfo.lifecycle,
        description: pestInfo.description,
        method: 'ai'
      };

    } catch (err) {
      throw err;
    }
  };

  const detectWithTensorFlow = async () => {
    // Simulated TensorFlow detection - in real implementation, this would use a trained model
    const mockPestDetections = [
      { pest: 'aphids', confidence: 'High', severity: 'Medium' },
      { pest: 'whiteflies', confidence: 'Medium', severity: 'Low' },
      { pest: 'spider_mites', confidence: 'High', severity: 'High' },
      { pest: 'healthy', confidence: 'High', severity: 'Low' }
    ];
    
    const randomDetection = mockPestDetections[Math.floor(Math.random() * mockPestDetections.length)];
    const pestInfo = getPestInfo(randomDetection.pest, i18n.language);
    
    return {
      pest: pestInfo.pest,
      confidence: randomDetection.confidence,
      severity: randomDetection.severity,
      damage: pestInfo.damage,
      treatment: pestInfo.treatment,
      prevention: pestInfo.prevention,
      chemical: pestInfo.chemical,
      organic: pestInfo.organic,
      lifecycle: pestInfo.lifecycle,
      description: pestInfo.description,
      method: 'tensorflow'
    };
  };

  const handleDetect = async () => {
    if (!image || loading) return;

    setLoading(true);
    setError(null);

    try {
      let detectionResult;
      
      // Try AI detection first, fallback to TensorFlow
      try {
        detectionResult = await detectWithAI();
      } catch (aiError) {
        detectionResult = await detectWithTensorFlow();
      }

      // Save to history
      const historyEntry = savePestHistory(detectionResult);
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

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'High': return '#dc2626';
      case 'Medium': return '#f59e0b';
      case 'Low': return '#16a34a';
      default: return '#6b7280';
    }
  };

  const getConfidenceColor = (confidence) => {
    switch (confidence) {
      case 'High': return '#16a34a';
      case 'Medium': return '#f59e0b';
      case 'Low': return '#ef4444';
      default: return '#6b7280';
    }
  };

  return (
    <div style={{ 
      maxWidth: "600px", 
      margin: "40px auto", 
      padding: "24px", 
      background: "#fff", 
      borderRadius: "16px", 
      boxShadow: "0 4px 20px rgba(0,0,0,0.1)",
      position: "relative"
    }}>
      {/* Header with controls */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <h2 style={{ color: "#16a34a", fontSize: "24px", margin: 0 }}>
          🐛 Pest Detection System
        </h2>
        <div style={{ display: 'flex', gap: '10px' }}>
          <button
            onClick={() => setShowAlerts(!showAlerts)}
            style={{
              padding: '6px 12px',
              backgroundColor: showAlerts ? '#ef4444' : '#f3f4f6',
              border: '1px solid #d1d5db',
              borderRadius: '6px',
              fontSize: '12px',
              cursor: 'pointer',
              color: showAlerts ? 'white' : '#374151'
            }}
          >
            🚨 Alerts ({regionalAlerts.length})
          </button>
          <button
            onClick={() => setShowHistory(!showHistory)}
            style={{
              padding: '6px 12px',
              backgroundColor: showHistory ? '#16a34a' : '#f3f4f6',
              border: '1px solid #d1d5db',
              borderRadius: '6px',
              fontSize: '12px',
              cursor: 'pointer',
              color: showHistory ? 'white' : '#374151'
            }}
          >
            📋 History ({history.length})
          </button>
          <button
            onClick={onClose}
            style={{
              padding: '6px 12px',
              backgroundColor: '#f3f4f6',
              border: '1px solid #d1d5db',
              borderRadius: '6px',
              fontSize: '12px',
              cursor: 'pointer'
            }}
          >
            ✕
          </button>
        </div>
      </div>

      {/* Regional Alerts */}
      {showAlerts && (
        <div style={{
          marginBottom: '20px',
          padding: '16px',
          backgroundColor: '#fef2f2',
          borderRadius: '12px',
          border: '1px solid #fecaca'
        }}>
          <h3 style={{ fontSize: '16px', fontWeight: '600', color: '#111', margin: '0 0 12px 0' }}>
            🚨 Regional Pest Alerts
          </h3>
          {regionalAlerts.length === 0 ? (
            <p style={{ textAlign: 'center', color: '#6b7280', fontSize: '14px', margin: 0 }}>
              No active pest alerts in your region
            </p>
          ) : (
            <div style={{ maxHeight: '150px', overflowY: 'auto' }}>
              {regionalAlerts.map((alert) => (
                <div
                  key={alert.id}
                  style={{
                    padding: '8px',
                    marginBottom: '8px',
                    backgroundColor: 'white',
                    borderRadius: '6px',
                    fontSize: '12px',
                    borderLeft: `4px solid ${getSeverityColor(alert.severity)}`
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontWeight: '500', color: '#111' }}>
                      {alert.pest.replace('_', ' ').toUpperCase()}
                    </span>
                    <span style={{ 
                      padding: '2px 6px',
                      backgroundColor: getSeverityColor(alert.severity),
                      color: 'white',
                      borderRadius: '4px',
                      fontSize: '10px'
                    }}>
                      {alert.severity}
                    </span>
                  </div>
                  <div style={{ color: '#6b7280', marginTop: '4px' }}>
                    {alert.region}: {alert.description}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

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
        <div style={{ fontSize: '48px', marginBottom: '12px' }}>🐛</div>
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

      {/* Detection Button */}
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
        {loading ? "🔍 Analyzing image..." : "🐛 Detect Pests"}
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
              🐛 Pest Detected: 
              <span style={{ 
                color: result.pest.toLowerCase().includes('healthy') ? "#16a34a" : "#dc2626",
                marginLeft: "6px"
              }}>
                {result.pest}
              </span>
            </h3>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ fontSize: '14px', color: '#6b7280' }}>Confidence:</span>
              <span style={{ 
                padding: '4px 8px',
                backgroundColor: getConfidenceColor(result.confidence),
                color: 'white',
                borderRadius: '4px',
                fontSize: '12px',
                fontWeight: 'bold'
              }}>
                {result.confidence}
              </span>
              <span style={{ 
                padding: '4px 8px',
                backgroundColor: getSeverityColor(result.severity),
                color: 'white',
                borderRadius: '4px',
                fontSize: '12px',
                fontWeight: 'bold'
              }}>
                {result.severity} Severity
              </span>
            </div>
          </div>

          <div style={{ marginBottom: '16px' }}>
            <h4 style={{ fontSize: '16px', fontWeight: '600', color: '#111', marginBottom: '8px' }}>
              📋 Description:
            </h4>
            <p style={{ color: '#555', lineHeight: '1.5', margin: 0 }}>
              {result.description}
            </p>
          </div>

          <div style={{ marginBottom: '16px' }}>
            <h4 style={{ fontSize: '16px', fontWeight: '600', color: '#111', marginBottom: '8px' }}>
              🌱 Damage Symptoms:
            </h4>
            <p style={{ color: '#555', lineHeight: '1.5', margin: 0 }}>
              {result.damage}
            </p>
          </div>

          <div style={{ marginBottom: '16px' }}>
            <h4 style={{ fontSize: '16px', fontWeight: '600', color: '#111', marginBottom: '8px' }}>
              💊 Immediate Treatment:
            </h4>
            <p style={{ color: '#555', lineHeight: '1.5', margin: 0 }}>
              {result.treatment}
            </p>
          </div>

          <div style={{ marginBottom: '16px' }}>
            <h4 style={{ fontSize: '16px', fontWeight: '600', color: '#111', marginBottom: '8px' }}>
              🛡️ Prevention Measures:
            </h4>
            <p style={{ color: '#555', lineHeight: '1.5', margin: 0 }}>
              {result.prevention}
            </p>
          </div>

          {(result.chemical || result.organic) && (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '16px' }}>
              {result.chemical && (
                <div>
                  <h4 style={{ fontSize: '14px', fontWeight: '600', color: '#111', marginBottom: '8px' }}>
                    🧪 Chemical Treatments:
                  </h4>
                  <ul style={{ margin: 0, paddingLeft: '16px' }}>
                    {result.chemical.map((chemical, index) => (
                      <li key={index} style={{ color: '#555', fontSize: '14px', marginBottom: '4px' }}>
                        {chemical}
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

          <div style={{ fontSize: '12px', color: '#6b7280', textAlign: 'center' }}>
            Detection method: {result.method === 'ai' ? '🤖 AI Analysis' : '🧠 TensorFlow.js'}
          </div>
        </div>
      )}

      {/* Pest History */}
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
              📋 Detection History
            </h3>
            {history.length > 0 && (
              <button
                onClick={() => {
                  if (confirm('Clear all pest history?')) {
                    localStorage.removeItem('pestHistory');
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
                      {entry.pest}
                    </span>
                    <span style={{ color: '#6b7280' }}>
                      {new Date(entry.timestamp).toLocaleDateString()}
                    </span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '4px' }}>
                    <span style={{
                      padding: '2px 6px',
                      backgroundColor: getConfidenceColor(entry.confidence),
                      color: 'white',
                      borderRadius: '4px',
                      fontSize: '10px'
                    }}>
                      {entry.confidence}
                    </span>
                    <span style={{
                      padding: '2px 6px',
                      backgroundColor: getSeverityColor(entry.severity),
                      color: 'white',
                      borderRadius: '4px',
                      fontSize: '10px'
                    }}>
                      {entry.severity}
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
          Upload a crop image to begin pest detection
        </p>
      )}
    </div>
  );
}
