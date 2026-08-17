"""Rule-based crop disease detection logic.

This module contains the shared disease analysis functions used by both
the platform router and the main monolithic application.
"""

import base64
import json
import os
import re
from typing import Any, Optional

from pydantic import BaseModel, Field, validator


_CROP_DISEASE_PROFILES = {
    "healthy": {
        "disease": "Healthy",
        "severity": "Low",
        "treatment": "No treatment is needed. Continue regular monitoring, irrigation, and nutrition management.",
        "prevention": "Keep a regular scouting schedule, avoid overwatering, and maintain balanced fertilization.",
        "pesticides": [],
        "organic": ["Crop rotation", "Balanced compost", "Consistent sanitation"],
    },
    "leaf_spot": {
        "disease": "Leaf Spot",
        "severity": "Medium",
        "treatment": "Remove the most affected leaves, improve airflow, and apply a suitable fungicide if symptoms spread.",
        "prevention": "Use resistant varieties, avoid overhead irrigation, and keep tools clean between fields.",
        "pesticides": ["Mancozeb", "Chlorothalonil", "Copper hydroxide"],
        "organic": ["Neem spray", "Baking soda solution", "Copper soap"],
    },
    "early_blight": {
        "disease": "Early Blight",
        "severity": "High",
        "treatment": "Remove infected foliage early, mulch to reduce soil splash, and apply a labeled fungicide when needed.",
        "prevention": "Rotate crops, stake plants for airflow, and avoid wetting leaves during irrigation.",
        "pesticides": ["Chlorothalonil", "Mancozeb", "Copper hydroxide"],
        "organic": ["Neem oil", "Baking soda spray", "Compost tea"],
    },
    "late_blight": {
        "disease": "Late Blight",
        "severity": "High",
        "treatment": "Destroy heavily infected plant tissue, improve drainage, and apply a targeted fungicide promptly.",
        "prevention": "Use disease-free seed, increase spacing, and avoid overhead watering in humid weather.",
        "pesticides": ["Metalaxyl", "Mefenoxam", "Copper hydroxide"],
        "organic": ["Copper sprays", "Bacillus subtilis", "Neem oil"],
    },
    "powdery_mildew": {
        "disease": "Powdery Mildew",
        "severity": "Medium",
        "treatment": "Remove infected growth, improve ventilation, and apply sulfur or another approved fungicide.",
        "prevention": "Space plants properly, water early in the day, and avoid excess nitrogen.",
        "pesticides": ["Sulfur", "Potassium bicarbonate", "Myclobutanil"],
        "organic": ["Milk spray", "Neem oil", "Baking soda solution"],
    },
    "rust": {
        "disease": "Rust",
        "severity": "Medium",
        "treatment": "Remove infected leaves and apply a fungicide before the disease spreads across the canopy.",
        "prevention": "Use resistant varieties, avoid overhead watering, and scout the crop after humid nights.",
        "pesticides": ["Azoxystrobin", "Tebuconazole", "Mancozeb"],
        "organic": ["Neem oil", "Copper sprays", "Sulfur dust"],
    },
    "bacterial_spot": {
        "disease": "Bacterial Spot",
        "severity": "High",
        "treatment": "Remove infected plant parts, avoid working in wet fields, and apply a copper-based bactericide if recommended locally.",
        "prevention": "Use certified seed, sanitize tools, and rotate away from infected solanaceous crops.",
        "pesticides": ["Copper hydroxide", "Fixed copper", "Streptomycin"],
        "organic": ["Copper soap", "Compost tea", "Bacillus subtilis"],
    },
    "mosaic_virus": {
        "disease": "Mosaic Virus",
        "severity": "High",
        "treatment": "Remove infected plants immediately because there is no cure, and control insect vectors around the plot.",
        "prevention": "Plant resistant varieties, control aphids and whiteflies, and keep weeds from hosting the virus.",
        "pesticides": ["Imidacloprid", "Thiamethoxam", "Dinotefuran"],
        "organic": ["Insecticidal soap", "Neem oil", "Row covers"],
    },
    "downy_mildew": {
        "disease": "Downy Mildew",
        "severity": "Medium",
        "treatment": "Improve ventilation, reduce humidity, and use a disease-specific fungicide when the infection is active.",
        "prevention": "Choose resistant cultivars, increase spacing, and avoid wet foliage overnight.",
        "pesticides": ["Metalaxyl", "Mefenoxam", "Copper hydroxide"],
        "organic": ["Bacillus subtilis", "Copper sprays", "Neem oil"],
    },
    "anthracnose": {
        "disease": "Anthracnose",
        "severity": "High",
        "treatment": "Prune infected tissue, improve sanitation, and apply a fungicide if the lesion pattern keeps expanding.",
        "prevention": "Rotate crops, avoid moving equipment through wet plants, and remove crop debris after harvest.",
        "pesticides": ["Chlorothalonil", "Mancozeb", "Tebuconazole"],
        "organic": ["Copper soap", "Neem oil", "Bacillus subtilis"],
    },
    "root_rot": {
        "disease": "Root Rot",
        "severity": "High",
        "treatment": "Improve drainage immediately, reduce irrigation frequency, and remove plants that have collapsed badly.",
        "prevention": "Use well-drained soil, avoid waterlogging, and rotate away from susceptible hosts.",
        "pesticides": ["Mefenoxam", "Metalaxyl", "Fosetyl-Al"],
        "organic": ["Beneficial microbes", "Neem oil", "Compost amendments"],
    },
}


def _normalise_disease_key(value: Optional[str]) -> str:
    if not value:
        return "leaf_spot"

    normalised = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    aliases = {
        "healthy_plant": "healthy",
        "healthy": "healthy",
        "leaf_spot": "leaf_spot",
        "leaf_blight": "early_blight",
        "early_blight": "early_blight",
        "late_blight": "late_blight",
        "powdery_mildew": "powdery_mildew",
        "rust": "rust",
        "bacterial_spot": "bacterial_spot",
        "mosaic_virus": "mosaic_virus",
        "downy_mildew": "downy_mildew",
        "anthracnose": "anthracnose",
        "root_rot": "root_rot",
    }
    return aliases.get(normalised, normalised if normalised in _CROP_DISEASE_PROFILES else "leaf_spot")


def _build_disease_response(
    disease_key: str,
    confidence_score: float,
    method: str,
    cues: Optional[list[str]] = None,
    crop_type: Optional[str] = None,
) -> dict:
    profile_key = _normalise_disease_key(disease_key)
    profile = _CROP_DISEASE_PROFILES.get(profile_key, _CROP_DISEASE_PROFILES["leaf_spot"])

    confidence_score = max(1, min(99, round(confidence_score, 2)))
    if confidence_score >= 80:
        confidence = "High"
    elif confidence_score >= 55:
        confidence = "Medium"
    else:
        confidence = "Low"

    result = {
        "cropType": crop_type,
        "diseaseKey": profile_key,
        "disease": profile["disease"],
        "severity": profile["severity"],
        "confidence": confidence,
        "confidenceScore": confidence_score,
        "treatment": profile["treatment"],
        "prevention": profile["prevention"],
        "pesticides": profile["pesticides"],
        "organic": profile["organic"],
        "method": method,
    }
    if cues:
        result["cues"] = cues
    return result


def _extract_json_object(text: str) -> dict:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?", "", candidate.strip(), flags=re.IGNORECASE).strip()
        candidate = candidate.rstrip("`").strip()

    match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
    if match:
        candidate = match.group(0)

    return json.loads(candidate)


def _heuristic_confidence(score: float, gap: float) -> float:
    return max(42.0, min(96.0, 54.0 + (score * 18.0) + (gap * 12.0)))


def _analyse_crop_disease_locally(image_bytes: bytes, crop_type: Optional[str] = None) -> dict:
    import cv2
    import numpy as np

    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Invalid image data")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mean_h = float(np.mean(hsv[:, :, 0]))
    mean_s = float(np.mean(hsv[:, :, 1]))
    mean_v = float(np.mean(hsv[:, :, 2]))
    texture = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_r = float(np.mean(rgb[:, :, 0]))
    mean_g = float(np.mean(rgb[:, :, 1]))
    mean_b = float(np.mean(rgb[:, :, 2]))

    cues = []
    candidates = {
        "healthy": max(0.0, 1.4 - abs(mean_s - 70.0) / 70.0 - abs(mean_v - 180.0) / 180.0 - texture / 180.0),
        "powdery_mildew": max(0.0, ((150.0 - mean_s) / 150.0) + ((220.0 - mean_v) / 220.0) + max(0.0, 1.0 - texture / 90.0)),
        "rust": max(0.0, ((mean_r + mean_g - 2.0 * mean_b) / 255.0) + (mean_h / 180.0) + (mean_s / 255.0) * 0.2),
        "early_blight": max(0.0, ((mean_r - mean_g) / 255.0) * 1.2 + texture / 140.0 + max(0.0, (140.0 - mean_v) / 140.0)),
        "late_blight": max(0.0, (170.0 - mean_v) / 170.0 + texture / 120.0 + max(0.0, (90.0 - mean_s) / 90.0)),
        "bacterial_spot": max(0.0, abs(mean_r - mean_g) / 255.0 + texture / 110.0 + max(0.0, (150.0 - mean_v) / 150.0)),
        "mosaic_virus": max(0.0, abs(mean_r - mean_g) / 255.0 + abs(mean_g - mean_b) / 255.0 + max(0.0, (110.0 - mean_s) / 110.0)),
        "downy_mildew": max(0.0, (160.0 - mean_s) / 160.0 + (190.0 - mean_v) / 190.0 + max(0.0, 1.0 - texture / 140.0)),
        "leaf_spot": max(0.0, texture / 100.0 + max(0.0, (135.0 - mean_v) / 135.0) + max(0.0, (90.0 - mean_s) / 90.0)),
        "anthracnose": max(0.0, texture / 120.0 + max(0.0, (140.0 - mean_v) / 140.0) + abs(mean_r - mean_g) / 255.0),
        "root_rot": max(0.0, (120.0 - mean_v) / 120.0 + texture / 150.0 + max(0.0, (70.0 - mean_s) / 70.0)),
    }

    crop_hint = _normalise_disease_key(crop_type)
    if crop_hint in {"early_blight", "late_blight", "bacterial_spot"}:
        candidates[crop_hint] += 0.25
    elif crop_hint == "healthy":
        candidates["healthy"] += 0.25

    ranked = sorted(candidates.items(), key=lambda item: item[1], reverse=True)
    disease_key, score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0

    if disease_key == "healthy":
        cues.append("Low texture variance and balanced color profile")
    elif disease_key == "powdery_mildew":
        cues.append("Bright surface with low saturation")
    elif disease_key in {"rust", "mosaic_virus"}:
        cues.append("Uneven color distribution across the leaf area")
    else:
        cues.append("Visible texture irregularities and discolored patches")

    if crop_type:
        cues.append(f"Crop hint: {crop_type}")

    confidence_score = _heuristic_confidence(score, max(0.0, score - runner_up))
    return _build_disease_response(disease_key, confidence_score, "local-heuristic", cues=cues, crop_type=crop_type)


def _coerce_backend_disease_result(payload: dict, crop_type: Optional[str] = None) -> dict:
    disease_key = _normalise_disease_key(payload.get("diseaseKey") or payload.get("disease"))
    confidence_value = payload.get("confidenceScore")

    if confidence_value is None:
        confidence_label = str(payload.get("confidence", "")).strip().lower()
        if confidence_label == "high":
            confidence_value = 84.0
        elif confidence_label == "medium":
            confidence_value = 64.0
        elif confidence_label == "low":
            confidence_value = 44.0
        else:
            confidence_value = 58.0

    result = _build_disease_response(
        disease_key,
        float(confidence_value),
        str(payload.get("method", "gemini")),
        cues=payload.get("cues") if isinstance(payload.get("cues"), list) else None,
        crop_type=crop_type,
    )

    for field in ("treatment", "prevention", "pesticides", "organic", "severity"):
        if payload.get(field):
            result[field] = payload[field]

    if payload.get("disease"):
        result["disease"] = payload["disease"]

    if payload.get("confidence") in {"High", "Medium", "Low"}:
        result["confidence"] = payload["confidence"]

    if payload.get("confidenceScore") is not None:
        result["confidenceScore"] = max(1, min(99, float(payload["confidenceScore"])))

    if payload.get("notes"):
        result["notes"] = payload["notes"]

    return result


class CropDiseaseImageRequest(BaseModel):
    image_base64: str = Field(..., min_length=10, description="Base64-encoded image data")
    mime_type: str = Field(..., pattern=r"^image/(jpeg|png|gif|webp)$", description="MIME type of the image")
    crop_type: Optional[str] = Field(default=None, max_length=50)

    @validator("image_base64")
    def validate_image_size(cls, value):
        if len(value) > 14000000:
            raise ValueError("Image payload size exceeds the maximum limit of 10MB")
        return value
