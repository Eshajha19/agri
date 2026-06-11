import { useEffect, useRef, useState } from "react";
import apiClient from "./services/api";
import { getDiseaseInfo, saveDiseaseHistory, getDiseaseHistory } from "./utils/diseaseDatabase";
import { useTranslation } from "react-i18next";
import {
  Leaf,
  History,
  Camera,
  Upload,
  Search,
  Loader2,
  Bug,
  Pill,
  Shield,
  FlaskConical,
  Sprout,
  Sparkles,
  AlertTriangle,
  X,
} from "lucide-react";

const MAX_HISTORY_ITEMS = 25;

const CROP_OPTIONS = [
  "tomato",
  "potato",
  "cotton",
  "rice",
  "wheat",
  "maize",
  "cucumber",
  "chili",
  "brinjal",
  "generic",
];

const normalizeDiseaseKey = (value) => {
  const normalized = String(value || "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");

  const aliases = {
    healthy_plant: "healthy",
    healthy: "healthy",
    leaf_spot: "leaf_spot",
    leaf_blight: "early_blight",
    early_blight: "early_blight",
    late_blight: "late_blight",
    powdery_mildew: "powdery_mildew",
    rust: "rust",
    bacterial_spot: "bacterial_spot",
    mosaic_virus: "mosaic_virus",
    downy_mildew: "downy_mildew",
    anthracnose: "anthracnose",
    root_rot: "root_rot",
  };

  return aliases[normalized] || normalized || "leaf_spot";
};

const confidenceLabel = (score) => {
  if (score >= 80) return "High";
  if (score >= 55) return "Medium";
  return "Low";
};

const fileToBase64 = (file) =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",")[1]);
    reader.onerror = () => reject(new Error("Unable to read the selected image"));
    reader.readAsDataURL(file);
  });

const loadImage = (src) =>
  new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Unable to inspect the image"));
    image.src = src;
  });

const extractLocalAnalysis = async (preview, cropType, language) => {
  const image = await loadImage(preview);
  const canvas = document.createElement("canvas");
  const size = 72;

  canvas.width = size;
  canvas.height = size;

  const context = canvas.getContext("2d", { willReadFrequently: true });

  if (!context) {
    throw new Error("Canvas analysis is not available in this browser");
  }

  context.drawImage(image, 0, 0, size, size);

  const { data } = context.getImageData(0, 0, size, size);

  let totalRed = 0;
  let totalGreen = 0;
  let totalBlue = 0;
  let saturationSum = 0;
  let brightnessSum = 0;
  let darkPixels = 0;
  let yellowPixels = 0;
  let brownPixels = 0;
  let whitePixels = 0;
  let varianceAccumulator = 0;

  for (let index = 0; index < data.length; index += 4) {
    const red = data[index];
    const green = data[index + 1];
    const blue = data[index + 2];

    totalRed += red;
    totalGreen += green;
    totalBlue += blue;

    const maxChannel = Math.max(red, green, blue);
    const minChannel = Math.min(red, green, blue);

    const saturation =
      maxChannel === 0
        ? 0
        : ((maxChannel - minChannel) / maxChannel) * 255;

    saturationSum += saturation;
    brightnessSum += maxChannel;

    if (maxChannel < 90) darkPixels += 1;
    if (red > 150 && green > 130 && blue < 115) yellowPixels += 1;
    if (red > green + 20 && green > blue + 10 && red < 180) brownPixels += 1;
    if (red > 210 && green > 210 && blue > 210) whitePixels += 1;

    const average = (red + green + blue) / 3;

    varianceAccumulator +=
      ((red - average) ** 2) +
      ((green - average) ** 2) +
      ((blue - average) ** 2);
  }

  const pixelCount = data.length / 4;

  const meanRed = totalRed / pixelCount;
  const meanGreen = totalGreen / pixelCount;
  const meanBlue = totalBlue / pixelCount;
  const meanSaturation = saturationSum / pixelCount;
  const meanBrightness = brightnessSum / pixelCount;
  const textureScore = varianceAccumulator / pixelCount;
  const darkRatio = darkPixels / pixelCount;
  const yellowRatio = yellowPixels / pixelCount;
  const brownRatio = brownPixels / pixelCount;
  const whiteRatio = whitePixels / pixelCount;

  const scores = {
    healthy:
      Math.max(
        0,
        1.2 -
          Math.abs(meanSaturation - 70) / 70 -
          Math.abs(meanBrightness - 180) / 180 -
          textureScore / 20000
      ),

    powdery_mildew:
      Math.max(
        0,
        (150 - meanSaturation) / 150 +
          (230 - meanBrightness) / 230 +
          whiteRatio * 2.0
      ),

    rust:
      Math.max(
        0,
        yellowRatio * 2.2 +
          Math.max(0, meanRed + meanGreen - 2 * meanBlue) / 255
      ),

    early_blight:
      Math.max(
        0,
        brownRatio * 2.0 +
          Math.max(0, meanRed - meanGreen) / 255 +
          textureScore / 22000
      ),

    late_blight:
      Math.max(
        0,
        darkRatio * 2.2 +
          Math.max(0, 150 - meanBrightness) / 150 +
          textureScore / 20000
      ),

    bacterial_spot:
      Math.max(
        0,
        brownRatio * 1.6 +
          Math.max(0, meanRed - meanGreen) / 255 +
          darkRatio
      ),

    mosaic_virus:
      Math.max(
        0,
        Math.abs(meanRed - meanGreen) / 255 +
          Math.abs(meanGreen - meanBlue) / 255 +
          Math.max(0, 110 - meanSaturation) / 110
      ),

    downy_mildew:
      Math.max(
        0,
        Math.max(0, 150 - meanSaturation) / 150 +
          Math.max(0, 210 - meanBrightness) / 210
      ),

    anthracnose:
      Math.max(
        0,
        brownRatio * 1.8 +
          darkRatio +
          textureScore / 22000
      ),

    root_rot:
      Math.max(
        0,
        darkRatio * 2.0 +
          Math.max(0, 130 - meanBrightness) / 130 +
          Math.max(0, 80 - meanSaturation) / 80
      ),

    leaf_spot:
      Math.max(
        0,
        brownRatio * 1.2 +
          darkRatio +
          textureScore / 24000
      ),
  };

  const ranked = Object.entries(scores).sort((a, b) => b[1] - a[1]);

  const [bestKey, bestScore] = ranked[0];
  const runnerUp = ranked[1]?.[1] || 0;

  const confidenceScore = Math.max(
    42,
    Math.min(
      96,
      52 + bestScore * 18 + (bestScore - runnerUp) * 14
    )
  );

  const diseaseInfo = getDiseaseInfo(bestKey, language);

  return {
    diseaseKey: bestKey,
    disease: diseaseInfo.disease,
    severity:
      confidenceScore >= 80
        ? "High"
        : confidenceScore >= 55
          ? "Medium"
          : "Low",

    confidence: confidenceLabel(confidenceScore),
    confidenceScore: Math.round(confidenceScore),
    treatment: diseaseInfo.treatment,
    prevention: diseaseInfo.prevention,
    pesticides: diseaseInfo.pesticides,
    organic: diseaseInfo.organic,
    method: "local-vision",
  };
};

export default function CropDiseaseDetection({ onClose }) {
  const { i18n } = useTranslation();

  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [cropType, setCropType] = useState("generic");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const HISTORY_STORAGE_KEY = "diseaseHistory";
  const [history, setHistory] = useState([]);

  const fileInputRef = useRef(null);
  const mountedRef = useRef(true);
  const requestIdRef = useRef(0);
  const currentImageRef = useRef(null);
  const lastHistorySignatureRef = useRef("");
  const detectionAbortRef = useRef(false);
  const historySyncTimeoutRef = useRef(null);

  useEffect(() => {
    mountedRef.current = true;

    try {
      const storedHistory = getDiseaseHistory();

      if (Array.isArray(storedHistory)) {
        const cleanedHistory = storedHistory
          .filter(
            (entry) =>
              entry &&
              typeof entry === "object" &&
              entry.id &&
              entry.disease
          )
          .slice(0, MAX_HISTORY_ITEMS);

        setHistory(cleanedHistory);

        lastHistorySignatureRef.current = JSON.stringify(
          cleanedHistory.map((item) => item.id)
        );
      } else {
        setHistory([]);
      }
    } catch (error) {
      console.warn("Failed to load disease history");
      setHistory([]);
    }

    return () => {
      mountedRef.current = false;
    };
  }, []);

  useEffect(() => {
    return () => {
      detectionAbortRef.current = true;

      if (preview) {
        URL.revokeObjectURL(preview);
      }

      if (historySyncTimeoutRef.current) {
        clearTimeout(historySyncTimeoutRef.current);
      }
    };
  }, [preview]);

  useEffect(() => {
    const syncHistoryState = () => {
      if (historySyncTimeoutRef.current) {
        clearTimeout(historySyncTimeoutRef.current);
      }

      historySyncTimeoutRef.current = setTimeout(() => {
        if (!mountedRef.current) return;

        try {
          const latestHistory = getDiseaseHistory();

          if (!Array.isArray(latestHistory)) {
            setHistory([]);
            return;
          }

          const cleanedHistory = latestHistory
            .filter(
              (entry) =>
                entry &&
                typeof entry === "object" &&
                entry.id &&
                entry.disease
            )
            .slice(0, MAX_HISTORY_ITEMS);

          const signature = JSON.stringify(
            cleanedHistory.map((item) => item.id)
          );

          if (signature !== lastHistorySignatureRef.current) {
            lastHistorySignatureRef.current = signature;
            setHistory(cleanedHistory);
          }
        } catch (error) {
          console.warn("History synchronization skipped:", error);
        }
      }, 180);
    };

    window.addEventListener("storage", syncHistoryState);

    return () => {
      window.removeEventListener("storage", syncHistoryState);

      if (historySyncTimeoutRef.current) {
        clearTimeout(historySyncTimeoutRef.current);
      }
    };
  }, []);

  const handleImageChange = (file) => {
    if (!file) return;

    const allowedTypes = [
      "image/jpeg",
      "image/png",
      "image/webp",
      "image/jpg",
    ];

    if (!allowedTypes.includes(file.type)) {
      setError("Please upload a valid image file (JPEG, PNG, or WebP).");
      return;
    }

    const MAX_IMAGE_SIZE_BYTES = 5 * 1024 * 1024;

    if (file.size > MAX_IMAGE_SIZE_BYTES) {
      setError("Image size should be less than 5MB.");
      return;
    }

    const objectUrl = URL.createObjectURL(file);

    setImage(file);
    requestIdRef.current++;
    currentImageRef.current = file.name;
    detectionAbortRef.current = true;

    setPreview((previous) => {
      if (previous) {
        URL.revokeObjectURL(previous);
      }

      return objectUrl;
    });
    setResult(null);
    setError(null);
  };

  const handleDetect = async () => {
    if (!image || loading) return;

    setLoading(true);
    setError(null);

    const requestId = ++requestIdRef.current;
    const imageName = image.name;

    detectionAbortRef.current = false;

    try {
      const base64 = await fileToBase64(image);

      let analysis = null;

      try {
        const response = await apiClient.post(
          "/api/crop-disease/analyze-image",
          {
            image_base64: base64,
            mime_type: image.type,
            crop_type: cropType,
          },
          { skipGlobalLoader: true }
        );

        analysis = response.data?.analysis || response.data;
      } catch (apiError) {
        analysis = await extractLocalAnalysis(
          preview,
          cropType,
          i18n.language
        );
      }

      const detectionResult = {
        ...analysis,
        cropType,
      };

      try {
        const historyEntry = saveDiseaseHistory(detectionResult);

        if (historyEntry) {
          setHistory((previous) => {
            const filtered = previous.filter(
              (item) =>
                item &&
                item.id !== historyEntry.id &&
                item.disease !== historyEntry.disease
            );

            const updatedHistory = [
              historyEntry,
              ...filtered,
            ].slice(0, MAX_HISTORY_ITEMS);

            lastHistorySignatureRef.current = JSON.stringify(
              updatedHistory.map((item) => item.id)
            );

            return updatedHistory;
          });
        }
      } catch (storageError) {
        console.warn(
          "Disease history could not be saved:",
          storageError
        );
      }

      if (
        !mountedRef.current ||
        detectionAbortRef.current ||
        requestId !== requestIdRef.current ||
        imageName !== currentImageRef.current
      ) {
        return;
      }

      setResult(detectionResult);
    } catch (err) {
      if (
        mountedRef.current &&
        requestId === requestIdRef.current
      ) {
        setError(
          err?.message || "Detection failed. Try again."
        );
      }
    } finally {
      if (
        mountedRef.current &&
        requestId === requestIdRef.current
      ) {
        setLoading(false);
      }
    }
  };

  const resetSelection = () => {
    requestIdRef.current++;
    detectionAbortRef.current = true;
    currentImageRef.current = null;

    if (preview) {
      URL.revokeObjectURL(preview);
    }

    setImage(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  const resultConfidenceWidth = result?.confidenceScore || (result?.confidence === "High" ? 84 : result?.confidence === "Medium" ? 62 : 38);
  const containerStyle = {
    width: "100%",
    maxWidth: "860px",
    margin: "0 auto",
    padding: "clamp(18px, 3vw, 28px)",
    borderRadius: "28px",
    background: "linear-gradient(180deg, rgba(248,255,250,0.98) 0%, rgba(240,248,244,0.98) 100%)",
    boxShadow: "0 24px 60px rgba(15, 23, 42, 0.12)",
    border: "1px solid rgba(22, 163, 74, 0.12)",
    position: "relative",
  };

  const panelStyle = {
    background: "rgba(255,255,255,0.9)",
    border: "1px solid rgba(148,163,184,0.18)",
    borderRadius: "24px",
    padding: "20px",
    boxShadow: "0 12px 30px rgba(15, 23, 42, 0.06)",
  };

  const chipStyle = {
    display: "inline-flex",
    alignItems: "center",
    gap: "6px",
    padding: "8px 12px",
    borderRadius: "999px",
    background: "rgba(22, 163, 74, 0.09)",
    color: "#166534",
    fontSize: "13px",
    fontWeight: 600,
  };

  return (
    <div style={containerStyle}>
      <button
        onClick={onClose}
        aria-label="Close crop disease detection"
        style={{
          position: "absolute",
          top: "16px",
          right: "16px",
          width: "38px",
          height: "38px",
          borderRadius: "999px",
          border: "1px solid rgba(148,163,184,0.2)",
          background: "rgba(255,255,255,0.95)",
          cursor: "pointer",
          display: "grid",
          placeItems: "center",
        }}
      >
        <X size={16} />
      </button>

      <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "space-between", gap: "16px", marginBottom: "18px" }}>
        <div>
          <div style={chipStyle}>
            <Sparkles size={14} aria-hidden="true" /> AI Disease Scanner
          </div>
          <h2 style={{ margin: "12px 0 6px", fontSize: "clamp(24px, 3vw, 34px)", lineHeight: 1.1, color: "#0f172a" }}>
            <Leaf size={26} style={{ verticalAlign: "-5px", color: "#16a34a" }} aria-hidden="true" /> Crop Disease Detection
          </h2>
          <p style={{ margin: 0, color: "#475569", maxWidth: "62ch" }}>
            Upload a leaf or crop photo and get a disease prediction, confidence score, and practical treatment guidance.
          </p>
        </div>

        <button
          onClick={() => setShowHistory((current) => !current)}
          style={{
            alignSelf: "flex-start",
            border: "1px solid rgba(148,163,184,0.25)",
            background: "white",
            borderRadius: "16px",
            padding: "12px 14px",
            cursor: "pointer",
            display: "inline-flex",
            gap: "8px",
            alignItems: "center",
            color: "#0f172a",
            fontWeight: 600,
          }}
        >
          <History size={16} aria-hidden="true" /> History ({history.length})
        </button>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 1.1fr) minmax(280px, 0.9fr)", gap: "18px" }}>
        <div style={{ display: "grid", gap: "18px" }}>
          <div style={panelStyle}>
            <label htmlFor="crop-type" style={{ display: "block", marginBottom: "8px", fontWeight: 700, color: "#0f172a" }}>
              Crop type
            </label>
            <select
              id="crop-type"
              value={cropType}
              onChange={(event) => setCropType(event.target.value)}
              style={{
                width: "100%",
                borderRadius: "14px",
                border: "1px solid rgba(148,163,184,0.25)",
                padding: "12px 14px",
                fontSize: "15px",
                background: "white",
              }}
            >
              {CROP_OPTIONS.map((option) => (
                <option key={option} value={option}>
                  {option === "generic" ? "Not sure / general crop" : option.charAt(0).toUpperCase() + option.slice(1)}
                </option>
              ))}
            </select>

            <div
              onDragOver={(event) => {
                event.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={(event) => {
                event.preventDefault();
                setIsDragging(false);
              }}
              onDrop={(event) => {
                event.preventDefault();
                setIsDragging(false);
                const droppedFile = event.dataTransfer.files?.[0];
                if (droppedFile) {
                  handleImageChange(droppedFile);
                }
              }}
              onClick={() => fileInputRef.current?.click()}
              role="button"
              tabIndex={0}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  fileInputRef.current?.click();
                }
              }}
              style={{
                marginTop: "16px",
                borderRadius: "20px",
                border: `1.5px dashed ${isDragging ? "#16a34a" : "rgba(148,163,184,0.45)"}`,
                background: isDragging ? "rgba(240,253,244,0.9)" : "linear-gradient(180deg, rgba(248,250,252,0.95), rgba(255,255,255,0.95))",
                padding: "26px",
                textAlign: "center",
                cursor: "pointer",
                transition: "all 160ms ease",
              }}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                capture="environment"
                onChange={(event) => handleImageChange(event.target.files?.[0])}
                style={{ display: "none" }}
              />
              <div style={{ display: "grid", placeItems: "center", marginBottom: "10px" }}>
                {isDragging ? <Upload size={48} color="#16a34a" /> : <Camera size={48} color="#16a34a" />}
              </div>
              <div style={{ fontSize: "18px", fontWeight: 700, color: "#0f172a", marginBottom: "6px" }}>
                {isDragging ? "Drop the crop image here" : "Click, tap, or drag an image"}
              </div>
              <p style={{ margin: 0, color: "#64748b" }}>
                Gallery and camera uploads supported. Keep the image under 5MB for faster analysis.
              </p>
            </div>

            {preview && (
              <div style={{ marginTop: "16px", borderRadius: "20px", overflow: "hidden", border: "1px solid rgba(148,163,184,0.18)" }}>
                <img src={preview} alt="Selected crop" style={{ width: "100%", display: "block", maxHeight: "340px", objectFit: "cover" }} />
              </div>
            )}

            <div style={{ display: "flex", gap: "12px", marginTop: "16px", flexWrap: "wrap" }}>
              <button
                onClick={handleDetect}
                disabled={!image || loading}
                style={{
                  flex: "1 1 220px",
                  minHeight: "48px",
                  border: "none",
                  borderRadius: "16px",
                  background: loading ? "#86efac" : "linear-gradient(135deg, #16a34a, #15803d)",
                  color: "white",
                  fontSize: "16px",
                  fontWeight: 700,
                  cursor: !image || loading ? "not-allowed" : "pointer",
                  opacity: !image || loading ? 0.75 : 1,
                  display: "inline-flex",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: "8px",
                }}
              >
                {loading ? <Loader2 size={18} className="spinner" aria-hidden="true" /> : <Search size={18} aria-hidden="true" />}
                {loading ? "Analyzing image..." : "Detect disease"}
              </button>

              <button
                onClick={resetSelection}
                disabled={!image && !result}
                style={{
                  minHeight: "48px",
                  padding: "0 18px",
                  borderRadius: "16px",
                  border: "1px solid rgba(148,163,184,0.3)",
                  background: "white",
                  color: "#0f172a",
                  fontWeight: 600,
                  cursor: !image && !result ? "not-allowed" : "pointer",
                }}
              >
                Clear
              </button>
            </div>

            {error && (
              <div style={{ marginTop: "14px", borderRadius: "16px", background: "rgba(254,226,226,0.65)", color: "#991b1b", padding: "12px 14px", display: "flex", gap: "10px", alignItems: "flex-start" }}>
                <AlertTriangle size={18} style={{ flexShrink: 0, marginTop: "2px" }} aria-hidden="true" />
                <div>{error}</div>
              </div>
            )}
          </div>

          {result && (
            <div style={panelStyle}>
              <div style={{ display: "flex", justifyContent: "space-between", gap: "12px", flexWrap: "wrap", marginBottom: "14px" }}>
                <div>
                  <div style={{ display: "inline-flex", alignItems: "center", gap: "8px", marginBottom: "8px", color: "#166534", fontWeight: 700 }}>
                    <Bug size={18} aria-hidden="true" /> Prediction
                  </div>
                  <h3 style={{ margin: 0, fontSize: "clamp(22px, 3vw, 30px)", color: result.disease.toLowerCase().includes("healthy") ? "#166534" : "#b91c1c" }}>
                    {result.disease}
                  </h3>
                  <p style={{ margin: "6px 0 0", color: "#64748b" }}>
                    Source: {result.method === "gemini" ? "Gemini AI" : result.method === "local-vision" ? "Local vision fallback" : "Backend analysis"}
                  </p>
                </div>

                <div style={{ minWidth: "220px" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "6px", color: "#475569", fontSize: "14px", fontWeight: 600 }}>
                    <span>Confidence</span>
                    <span>{result.confidence} {result.confidenceScore ? `(${result.confidenceScore}%)` : ""}</span>
                  </div>
                  <div style={{ height: "10px", borderRadius: "999px", background: "#e2e8f0", overflow: "hidden" }}>
                    <div
                      style={{
                        width: `${resultConfidenceWidth}%`,
                        height: "100%",
                        borderRadius: "999px",
                        background: result.confidence === "High" ? "linear-gradient(90deg, #16a34a, #22c55e)" : result.confidence === "Medium" ? "linear-gradient(90deg, #f59e0b, #fbbf24)" : "linear-gradient(90deg, #ef4444, #f97316)",
                        transition: "width 220ms ease",
                      }}
                    />
                  </div>
                </div>
              </div>

              {Array.isArray(result.cues) && result.cues.length > 0 && (
                <div style={{ display: "flex", flexWrap: "wrap", gap: "8px", marginBottom: "16px" }}>
                  {result.cues.map((cue) => (
                    <span key={cue} style={{ padding: "8px 10px", borderRadius: "999px", background: "rgba(14,165,233,0.1)", color: "#075985", fontSize: "13px" }}>
                      {cue}
                    </span>
                  ))}
                </div>
              )}

              <div style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: "14px" }}>
                <div style={{ padding: "14px", borderRadius: "18px", background: "rgba(255,255,255,0.92)", border: "1px solid rgba(148,163,184,0.14)" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "8px", fontWeight: 700, marginBottom: "8px", color: "#0f172a" }}>
                    <Pill size={16} aria-hidden="true" /> Treatment
                  </div>
                  <p style={{ margin: 0, color: "#334155", lineHeight: 1.6 }}>{result.treatment}</p>
                </div>

                <div style={{ padding: "14px", borderRadius: "18px", background: "rgba(255,255,255,0.92)", border: "1px solid rgba(148,163,184,0.14)" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "8px", fontWeight: 700, marginBottom: "8px", color: "#0f172a" }}>
                    <Shield size={16} aria-hidden="true" /> Prevention
                  </div>
                  <p style={{ margin: 0, color: "#334155", lineHeight: 1.6 }}>{result.prevention}</p>
                </div>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: "14px", marginTop: "14px" }}>
                <div style={{ padding: "14px", borderRadius: "18px", background: "rgba(255,255,255,0.92)", border: "1px solid rgba(148,163,184,0.14)" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "8px", fontWeight: 700, marginBottom: "8px", color: "#0f172a" }}>
                    <FlaskConical size={16} aria-hidden="true" /> Suggested pesticides
                  </div>
                  {result.pesticides?.length ? (
                    <ul style={{ margin: 0, paddingLeft: "18px", color: "#334155" }}>
                      {result.pesticides.map((item) => (
                        <li key={item} style={{ marginBottom: "6px" }}>{item}</li>
                      ))}
                    </ul>
                  ) : (
                    <p style={{ margin: 0, color: "#64748b" }}>No chemical treatment recommended for this result.</p>
                  )}
                </div>

                <div style={{ padding: "14px", borderRadius: "18px", background: "rgba(255,255,255,0.92)", border: "1px solid rgba(148,163,184,0.14)" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "8px", fontWeight: 700, marginBottom: "8px", color: "#0f172a" }}>
                    <Sprout size={16} aria-hidden="true" /> Organic options
                  </div>
                  {result.organic?.length ? (
                    <ul style={{ margin: 0, paddingLeft: "18px", color: "#334155" }}>
                      {result.organic.map((item) => (
                        <li key={item} style={{ marginBottom: "6px" }}>{item}</li>
                      ))}
                    </ul>
                  ) : (
                    <p style={{ margin: 0, color: "#64748b" }}>No organic options listed.</p>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>

        {showHistory && (
          <div style={{ ...panelStyle, alignSelf: "start" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: "12px", marginBottom: "12px" }}>
              <h3 style={{ margin: 0, color: "#0f172a", fontSize: "18px", display: "flex", alignItems: "center", gap: "8px" }}>
                <History size={16} aria-hidden="true" /> Detection history
              </h3>
              <button
                onClick={() => {
                  if (typeof window !== "undefined" && window.confirm("Clear all disease history?")) {
                    try {
                      localStorage.removeItem(HISTORY_STORAGE_KEY);
                    } catch (err) {
                      console.error("Failed to clear history:", err);
                    }

                    lastHistorySignatureRef.current = "";
                    setHistory([]);
                  }
                }}
                disabled={history.length === 0}
                style={{
                  border: "none",
                  background: history.length === 0 ? "#cbd5e1" : "#ef4444",
                  color: "white",
                  borderRadius: "999px",
                  padding: "8px 12px",
                  fontWeight: 700,
                  cursor: history.length === 0 ? "not-allowed" : "pointer",
                }}
              >
                Clear
              </button>
            </div>

            {history.length === 0 ? (
              <p style={{ margin: 0, color: "#64748b" }}>No detection history yet.</p>
            ) : (
              <div style={{ display: "grid", gap: "10px", maxHeight: "520px", overflow: "auto", paddingRight: "4px" }}>
                {history.map((entry) => (
                  <div key={entry.id} style={{ padding: "12px", borderRadius: "16px", border: "1px solid rgba(148,163,184,0.14)", background: "white" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", gap: "10px", alignItems: "start" }}>
                      <div>
                        <div style={{ fontWeight: 700, color: "#0f172a" }}>{entry.disease}</div>
                        <div style={{ color: "#64748b", fontSize: "13px", marginTop: "3px" }}>{entry.method}</div>
                      </div>
                      <div style={{ color: "#64748b", fontSize: "12px" }}>{new Date(entry.timestamp).toLocaleDateString()}</div>
                    </div>
                    <div style={{ marginTop: "10px", display: "flex", gap: "8px", flexWrap: "wrap" }}>
                      <span style={{ padding: "6px 10px", borderRadius: "999px", background: entry.confidence === "High" ? "#dcfce7" : entry.confidence === "Medium" ? "#fef3c7" : "#fee2e2", color: entry.confidence === "High" ? "#166534" : entry.confidence === "Medium" ? "#92400e" : "#991b1b", fontSize: "12px", fontWeight: 700 }}>
                        {entry.confidence}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
