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
  const [history, setHistory] = useState([]);

  const fileInputRef = useRef(null);
  const mountedRef = useRef(true);
  const detectionAbortRef = useRef(false);
  const historySyncTimeoutRef = useRef(null);

  useEffect(() => {
    mountedRef.current = true;

    try {
      const storedHistory = getDiseaseHistory();

      if (Array.isArray(storedHistory)) {
        setHistory(storedHistory.slice(0, MAX_HISTORY_ITEMS));
      } else {
        setHistory([]);
      }
    } catch (error) {
      console.warn("Failed to restore disease history:", error);
      setHistory([]);
    }

    return () => {
      mountedRef.current = false;

      if (historySyncTimeoutRef.current) {
        clearTimeout(historySyncTimeoutRef.current);
      }
    };
  }, []);

  useEffect(() => {
    return () => {
      detectionAbortRef.current = true;

      if (preview) {
        URL.revokeObjectURL(preview);
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

          if (Array.isArray(latestHistory)) {
            setHistory(latestHistory.slice(0, MAX_HISTORY_ITEMS));
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

    if (!file.type.startsWith("image/")) {
      setError("Please upload a valid image file.");
      return;
    }

    if (file.size > 5 * 1024 * 1024) {
      setError("Image size should be less than 5MB.");
      return;
    }

    if (preview) {
      URL.revokeObjectURL(preview);
    }

    setImage(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
    setError(null);
  };

  const handleDetect = async () => {
    if (!image || loading) return;

    setLoading(true);
    setError(null);

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
            const nextHistory = [historyEntry, ...previous];
            return nextHistory.slice(0, MAX_HISTORY_ITEMS);
          });
        }
      } catch (storageError) {
        console.warn(
          "Disease history could not be saved:",
          storageError
        );
      }

      if (!mountedRef.current || detectionAbortRef.current) {
        return;
      }

      setResult(detectionResult);
    } catch (err) {
      setError(err?.message || "Detection failed. Try again.");
    } finally {
      if (mountedRef.current) {
        setLoading(false);
      }
    }
  };