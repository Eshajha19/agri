"""Platform router for cross-cutting endpoints.

This module hosts endpoints that don't belong to a single domain router,
keeping main.py focused on application wiring only.
"""

import asyncio
import hashlib
import io
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Optional

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ed25519, padding
from fastapi import APIRouter, Form, HTTPException, Request, Response
from pydantic import BaseModel, Field, validator
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from slowapi import Limiter
from slowapi.util import get_remote_address

from backend.compute_rate_limit import enforce_compute_rate_limit
from backend.schemas import AlertTriggerRequest, RAGQuery
from backend.utils.numeric_validation import validate_numeric_bounds

router = APIRouter()

# ---------------------------------------------------------------------------
# Simple per-IP rate limiter for public endpoints (no Redis dependency)
# ---------------------------------------------------------------------------
_log_error_buckets: dict[str, list[float]] = {}

def _rate_limit_log_error(request: Request, limit: int = 30, window: int = 60) -> None:
    """Reject with 429 if *ip* exceeds *limit* requests within *window* seconds."""
    ip = "unknown"
    if request.client:
        ip = request.client.host
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        ip = forwarded.split(",")[0].strip()

    now = time.time()
    bucket = _log_error_buckets.setdefault(ip, [])
    # Prune timestamps outside the window
    bucket[:] = [t for t in bucket if now - t < window]
    if len(bucket) >= limit:
        raise HTTPException(status_code=429, detail="Too many error log requests. Please retry later.")
    bucket.append(now)
logger = logging.getLogger(__name__)
limiter = Limiter(key_func=get_remote_address)


class WhatsAppSubscribeRequest(BaseModel):
    phone_number: str = Field(..., min_length=7, max_length=20)
    name: str = Field(..., min_length=1, max_length=100)
    # user_id is accepted for backward-compatibility but is IGNORED.
    # The authoritative identity is always derived from the verified
    # Firebase ID token — never from client-supplied data.
    user_id: Optional[str] = None


class ReportRequest(BaseModel):
    name: str = Field(..., max_length=100, description="Full name of the farmer")
    crop: str = Field(..., max_length=50, description="Primary crop type")
    area: str = Field(..., max_length=50, description="Total farm area")
    profit: str = Field(..., max_length=50, description="Estimated season profit")
    season: str = Field(..., max_length=50, description="Farming season")

    @validator("name", "crop", "area", "profit", "season", pre=True)
    def sanitize_and_validate_input(cls, value):
        if isinstance(value, str):
            value = value.strip()
            if "|" in value:
                raise ValueError("Field value must not contain the '|' character.")
        return value


class ClientErrorReport(BaseModel):
    message: str = Field(..., min_length=1, max_length=500)
    source: Optional[str] = Field(default=None, max_length=200)
    stack: Optional[str] = Field(default=None, max_length=2000)
    level: str = Field(default="error", max_length=20)


class GeminiImageRequest(BaseModel):
    image_base64: str = Field(..., min_length=10, description="Base64-encoded image data")
    mime_type: str = Field(..., pattern=r"^image/(jpeg|png|gif|webp)$", description="MIME type of the image")
    prompt: str = Field(..., min_length=10, max_length=2000, description="Analysis prompt")

    @validator("image_base64")
    def validate_image_size(cls, value):
        if len(value) > 14000000:
            raise ValueError("Image payload size exceeds the maximum limit of 10MB")
        return value


class CropDiseaseImageRequest(BaseModel):
    image_base64: str = Field(..., min_length=10, description="Base64-encoded image data")
    mime_type: str = Field(..., pattern=r"^image/(jpeg|png|gif|webp)$", description="MIME type of the image")
    crop_type: Optional[str] = Field(default=None, max_length=50)

    @validator("image_base64")
    def validate_image_size(cls, value):
        if len(value) > 14000000:
            raise ValueError("Image payload size exceeds the maximum limit of 10MB")
        return value


class SimulationRequest(BaseModel):
    crop_type: str
    temp_delta: float = Field(..., ge=-5, le=5)
    rain_delta: float = Field(..., ge=-100, le=100)


class SeedVerifyRequest(BaseModel):
    code: str = Field(..., min_length=4, max_length=100)


verify_role_fn = None
get_signing_keys_fn = None
sanitise_log_field_fn = None
rag_generate_fn = None
subscriber_store = None
send_whatsapp_message_fn = None
format_alert_message_fn = None
weather_service = None
rbac_manager = None
permission_enum = None


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


def init_platform(
    verify_role,
    get_signing_keys,
    sanitise_log_field,
    rag_generate,
    subscribers,
    send_whatsapp_message,
    format_alert_message,
    weather_alert_service,
    rbac,
    permission,
):
    global verify_role_fn
    global get_signing_keys_fn
    global sanitise_log_field_fn
    global rag_generate_fn
    global subscriber_store
    global send_whatsapp_message_fn
    global format_alert_message_fn
    global weather_service
    global rbac_manager
    global permission_enum

    verify_role_fn = verify_role
    get_signing_keys_fn = get_signing_keys
    sanitise_log_field_fn = sanitise_log_field
    rag_generate_fn = rag_generate
    subscriber_store = subscribers
    send_whatsapp_message_fn = send_whatsapp_message
    format_alert_message_fn = format_alert_message
    weather_service = weather_alert_service
    rbac_manager = rbac
    permission_enum = permission


@router.get("/weather/alerts/history")
async def get_alerts_history(request: Request):
    """Return the most recent server-side weather alerts.

    Requires a valid authenticated session. The alert history is internal
    operational data and must not be exposed to unauthenticated callers.
    """
    if verify_role_fn is None:
        raise HTTPException(status_code=503, detail="Auth service unavailable")

    await verify_role_fn(request)

    if weather_service is None:
        raise HTTPException(status_code=503, detail="Weather service unavailable")

    recent_alerts = weather_service.alert_history[-50:]
    return {
        "success": True,
        "total_alerts": len(weather_service.alert_history),
        "recent_alerts": [alert.to_dict() for alert in recent_alerts],
    }


@router.post("/whatsapp/subscribe", dependencies=[Depends(verify_csrf_token_dependency)])
async def subscribe_whatsapp(data: WhatsAppSubscribeRequest, request: Request):
    """
    Subscribe the authenticated user to WhatsApp alerts.

    The subscriber's identity is derived exclusively from the verified
    Firebase ID token — never from the client-supplied user_id field.
    This prevents an attacker from overwriting another user's subscription
    by sending a known uid with an attacker-controlled phone number.
    """
    if verify_role_fn is None:
        raise HTTPException(status_code=500, detail="Auth service not initialized")
    if subscriber_store is None:
        raise HTTPException(status_code=500, detail="Subscriber store not initialized")

    token_data = await verify_role_fn(request)
    uid = token_data.get("uid")

    subscriber = {
        "phone_number": data.phone_number,
        "name": data.name,
        "subscribed_at": datetime.now().isoformat(),
    }

    try:
        subscriber_store.upsert(uid, subscriber)
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to save subscription. Please try again.",
        ) from exc

    if send_whatsapp_message_fn is not None:
        welcome_msg = (
            f"Namaste {data.name}!\n\n"
            "Welcome to Fasal Saathi WhatsApp Alerts. "
            "You will now receive real-time updates directly here."
        )
        await asyncio.to_thread(send_whatsapp_message_fn, data.phone_number, welcome_msg)

    return {"success": True, "message": "Successfully subscribed"}


@router.post("/whatsapp/trigger-alert", dependencies=[Depends(verify_csrf_token_dependency)])
async def trigger_whatsapp_alert(data: AlertTriggerRequest, request: Request):
    """
    Broadcast a WhatsApp alert to all subscribers.

    Requires authentication — admin or expert role only.

    Without this check any unauthenticated caller could send arbitrary
    messages to every subscribed farmer (social engineering attacks,
    fake pest warnings, fake market alerts) and consume Twilio API
    credits at will.
    """
    if verify_role_fn is None:
        raise HTTPException(status_code=500, detail="Auth service not initialized")
    if subscriber_store is None or send_whatsapp_message_fn is None or format_alert_message_fn is None:
        raise HTTPException(status_code=500, detail="WhatsApp dependencies not initialized")

    # RBAC: only admins and experts may broadcast alerts to all farmers.
    await verify_role_fn(request, required_roles=["admin", "expert"])

    subscribers = subscriber_store.get_all()
    results = []
    formatted_msg = format_alert_message_fn(data.alert_type, data.message)

    for user_id, info in subscribers.items():
        result = await asyncio.to_thread(send_whatsapp_message_fn, info["phone_number"], formatted_msg)
        results.append({"user_id": user_id, "success": result.get("success", False)})

    delivered = sum(1 for r in results if r["success"])
    return {"success": True, "results": results, "delivered": delivered, "total": len(results)}


@router.post("/reports/generate", dependencies=[Depends(verify_csrf_token_dependency)])
async def generate_signed_report(request: Request, data: ReportRequest):
    if verify_role_fn is None or get_signing_keys_fn is None:
        raise HTTPException(status_code=500, detail="Report dependencies not initialized")

    await verify_role_fn(request, required_roles=["expert", "admin"])
    private_key = get_signing_keys_fn()

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    pdf.setFont("Helvetica-Bold", 24)
    pdf.setFillColor(colors.green)
    pdf.drawCentredString(width / 2, height - 1 * inch, "FASAL SAATHI")

    pdf.setFont("Helvetica-Bold", 18)
    pdf.setFillColor(colors.black)
    pdf.drawCentredString(width / 2, height - 1.5 * inch, "CERTIFIED FINANCIAL FARM REPORT")

    pdf.setStrokeColor(colors.green)
    pdf.line(1 * inch, height - 1.7 * inch, width - 1 * inch, height - 1.7 * inch)

    pdf.setFont("Helvetica", 14)
    y = height - 2.5 * inch

    details = [
        ("Farmer Name:", data.name),
        ("Crop Type:", data.crop),
        ("Farm Area:", data.area),
        ("Season Profit:", f"Rs. {data.profit}"),
        ("Season:", data.season),
        ("Report Date:", datetime.now().strftime("%d %B, %Y")),
    ]

    for label, value in details:
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(1.5 * inch, y, label)
        pdf.setFont("Helvetica", 14)
        pdf.drawString(3.5 * inch, y, value)
        y -= 0.4 * inch

    y -= 0.5 * inch
    pdf.setStrokeColor(colors.black)
    pdf.rect(1 * inch, y - 1.5 * inch, width - 2 * inch, 1.8 * inch, stroke=1, fill=0)

    signing_payload = {
        "name": data.name,
        "crop": data.crop,
        "area": data.area,
        "profit": data.profit,
        "season": data.season,
        "date": datetime.now().date().isoformat(),
    }
    report_data_string = json.dumps(signing_payload, sort_keys=True)
    report_data_bytes = report_data_string.encode("utf-8")
    
    # Handle both Ed25519 and RSA private keys
    if isinstance(private_key, ed25519.Ed25519PrivateKey):
        signature = private_key.sign(report_data_bytes)
    else:
        # RSA key - requires padding and hash algorithm
        signature = private_key.sign(
            report_data_bytes,
            padding.PKCS1v15(),
            hashes.SHA256(),
        )
    sig_id = hashlib.sha256(signature).hexdigest()[:8].upper()

    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(1.2 * inch, y - 0.3 * inch, "DIGITAL CRYPTOGRAPHIC SIGNATURE")
    pdf.setFont("Helvetica", 12)
    pdf.drawString(1.2 * inch, y - 0.7 * inch, f"Signature ID: {sig_id}")
    pdf.setFont("Helvetica-Bold", 12)
    pdf.setFillColor(colors.green)
    pdf.drawString(1.2 * inch, y - 1.0 * inch, "Status: VERIFIED")

    pdf.showPage()
    pdf.save()

    pdf_content = buffer.getvalue()
    buffer.close()

    return Response(
        content=pdf_content,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=FasalSaathi_Report_{sig_id}.pdf"},
    )


@router.post("/log-error")
async def log_error(request: Request, body: ClientErrorReport):
    # Simple per-IP rate limiter: 30 requests per minute
    _rate_limit_log_error(request, limit=30, window=60)

    if sanitise_log_field_fn is None:
        raise HTTPException(status_code=500, detail="Log sanitizer not initialized")

    level = sanitise_log_field_fn(body.level).lower()
    message = sanitise_log_field_fn(body.message)
    source = sanitise_log_field_fn(body.source) if body.source else "unknown"
    stack = sanitise_log_field_fn(body.stack) if body.stack else ""

    log_fn = {
        "error": logger.error,
        "warn": logger.warning,
        "warning": logger.warning,
        "info": logger.info,
    }.get(level, logger.error)

    log_fn(
        "[ClientError] level=%s source=%s message=%s%s",
        level,
        source,
        message,
        f" stack={stack}" if stack else "",
    )
    return {"success": True}


@router.post("/rag/query")
async def rag_query(request: Request, body: RAGQuery):
    if rag_generate_fn is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not available")

    if verify_role_fn is None:
        raise HTTPException(status_code=500, detail="Auth service not initialized")

    token_data = await verify_role_fn(request)
    rate_limited = enforce_compute_rate_limit(
        request,
        scope="platform.rag_query",
        uid=(token_data or {}).get("uid"),
        limit=12,
        window_seconds=60,
    )
    if rate_limited is not None:
        return rate_limited

    try:
        return rag_generate_fn(body.query, top_k=body.top_k)
    except Exception as exc:
        logger.exception("RAG query failed")
        raise HTTPException(status_code=500, detail="RAG query failed") from exc


@router.post("/gemini/analyze-image")
async def gemini_analyze_image(request: Request, body: GeminiImageRequest):
    if verify_role_fn is None:
        raise HTTPException(status_code=500, detail="Auth service not initialized")

    # Require a valid Firebase ID token to prevent unauthenticated callers
    # from proxying arbitrary images through the server's GEMINI_API_KEY,
    # exhausting quota and incurring billing charges.
    token_data = await verify_role_fn(request)
    rate_limited = enforce_compute_rate_limit(
        request,
        scope="platform.gemini_analyze_image",
        uid=(token_data or {}).get("uid"),
        limit=5,
        window_seconds=60,
    )
    if rate_limited is not None:
        return rate_limited

    import httpx

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=503, detail="AI analysis service is not configured")

    gemini_url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.0-flash:generateContent?key={api_key}"
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": body.prompt},
                    {
                        "inline_data": {
                            "mime_type": body.mime_type,
                            "data": body.image_base64,
                        }
                    },
                ]
            }
        ]
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(gemini_url, json=payload)

        if response.status_code != 200:
            logger.warning("Gemini API returned %s: %s", response.status_code, response.text[:200])
            raise HTTPException(status_code=502, detail="AI analysis service returned an error")

        data = response.json()
        text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")

        if not text:
            raise HTTPException(status_code=502, detail="Empty response from AI analysis service")

        return {"text": text}

    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail="AI analysis service timed out") from exc


@router.post("/crop-disease/analyze-image")
async def analyze_crop_disease_image(request: Request, body: CropDiseaseImageRequest):
    if verify_role_fn is None:
        raise HTTPException(status_code=500, detail="Auth service not initialized")

    token_data = await verify_role_fn(request)
    rate_limited = enforce_compute_rate_limit(
        request,
        scope="platform.crop_disease_analyze_image",
        uid=(token_data or {}).get("uid"),
        limit=5,
        window_seconds=60,
    )
    if rate_limited is not None:
        return rate_limited

    import base64
    import httpx

    try:
        image_bytes = base64.b64decode(body.image_base64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 image data") from exc

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if api_key:
        gemini_url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-2.0-flash:generateContent?key={api_key}"
        )
        prompt = (
            "You are an agricultural plant pathologist. Inspect the crop image and return ONLY valid JSON with these keys: "
            "diseaseKey, disease, confidence, confidenceScore, severity, treatment, prevention, pesticides, organic, cues. "
            "Use diseaseKey values from this set when possible: healthy, leaf_spot, early_blight, late_blight, powdery_mildew, rust, bacterial_spot, mosaic_virus, downy_mildew, anthracnose, root_rot. "
            "Keep treatment and prevention concise and practical for farmers."
        )
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": body.mime_type,
                                "data": body.image_base64,
                            }
                        },
                    ]
                }
            ]
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(gemini_url, json=payload)

            if response.status_code == 200:
                data = response.json()
                text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                if text:
                    parsed = _extract_json_object(text)
                    return {
                        "success": True,
                        "analysis": _coerce_backend_disease_result(parsed, crop_type=body.crop_type),
                    }
        except Exception as exc:
            logger.warning("Gemini crop disease analysis failed; using local fallback: %s", exc)

    analysis = _analyse_crop_disease_locally(image_bytes, crop_type=body.crop_type)
    return {"success": True, "analysis": analysis}


@router.post("/simulate-climate")
async def simulate_climate(request: Request, data: SimulationRequest):
    if verify_role_fn is None:
        raise HTTPException(status_code=500, detail="Auth service not initialized")

    # Require a valid Firebase ID token to prevent unauthenticated callers
    # from consuming compute resources and to keep this route consistent with
    # the authenticated /api/knowledge/simulate-climate endpoint.
    token_data = await verify_role_fn(request)
    rate_limited = enforce_compute_rate_limit(
        request,
        scope="platform.simulate_climate",
        uid=(token_data or {}).get("uid"),
        limit=10,
        window_seconds=60,
    )
    if rate_limited is not None:
        return rate_limited

    # Validate that temp_delta and rain_delta are finite numbers.
    # Pydantic's ge/le constraints do not reject float('inf') or float('nan')
    # when the value is already a Python float, so we apply the shared utility
    # to block those edge cases before they reach the multiplication below.
    validate_numeric_bounds(
        {"temp_delta": data.temp_delta, "rain_delta": data.rain_delta},
        ["temp_delta", "rain_delta"],
    )

    sensitivities = {
        "rice": {"temp": -0.05, "rain": 0.02},
        "wheat": {"temp": -0.06, "rain": 0.03},
        "cotton": {"temp": -0.03, "rain": 0.01},
        "maize": {"temp": -0.07, "rain": 0.04},
        "sugarcane": {"temp": -0.02, "rain": 0.05},
        "soybean": {"temp": -0.04, "rain": 0.03},
        "potato": {"temp": -0.05, "rain": 0.04},
        "default": {"temp": -0.04, "rain": 0.02},
    }

    crop = data.crop_type.lower()
    coeff = sensitivities.get(crop, sensitivities["default"])

    yield_impact_temp = data.temp_delta * coeff["temp"]
    yield_impact_rain = (data.rain_delta / 100.0) * coeff["rain"]
    total_yield_impact = yield_impact_temp + yield_impact_rain
    profit_impact = total_yield_impact * 1.5
    suitability = max(0, min(100, 85 + (total_yield_impact * 100)))

    return {
        "crop_type": data.crop_type,
        "yield_impact_pct": round(total_yield_impact * 100, 2),
        "profit_impact_pct": round(profit_impact * 100, 2),
        "suitability_score": round(suitability, 1),
        "risk_level": "High" if total_yield_impact < -0.15 else "Medium" if total_yield_impact < -0.05 else "Low",
        "recommendation": "Switch to heat-tolerant varieties" if data.temp_delta > 2 else "Ensure adequate irrigation" if data.rain_delta < -20 else "Conditions remain viable",
    }


@router.post("/seeds/verify")
async def verify_seed(request: Request, data: SeedVerifyRequest):
    if rbac_manager is None or permission_enum is None:
        raise HTTPException(status_code=500, detail="RBAC not initialized")

    await rbac_manager.raise_if_unauthorized(request, [permission_enum.SEEDS_VERIFY], require_all=False)

    # ---------------------------------------------------------------------------
    # Seed registry — IMPORTANT LIMITATION
    #
    # This is a minimal static registry used for demonstration purposes only.
    # It contains two known codes: one authentic and one blacklisted counterfeit.
    # Any code NOT in this registry returns status="unverified" with a warning
    # that the seed could not be verified — NOT a clean "not found" that implies
    # the seed is safe. Farmers must be told to treat unverified codes with
    # caution and contact their local agricultural office for confirmation.
    #
    # Production deployment should replace this dict with a Firestore/database
    # lookup against a maintained registry of certified seed batches.
    # ---------------------------------------------------------------------------
    seed_registry = {
        "FS-RICE-2026-A1": {
            "status": "authentic",
            "crop": "Rice (IR-64)",
            "batch": "2026-A1",
            "manufacturer": "National Seeds Corporation (NSC)",
            "cert_body": "Central Seed Certification Board (CSCB)",
            "certified_on": "2025-10-01",
            "expires_on": "2027-03-31",
        },
        "FS-FAKE-2026-X9": {
            "status": "invalid",
            "crop": "Unknown",
            "batch": "2026-X9",
            "manufacturer": "Unknown",
            "cert_body": "N/A",
            "certified_on": "N/A",
            "expires_on": "N/A",
            "reason": "Blacklisted - reported counterfeit batch",
        },
    }

    code = data.code.upper().strip()
    entry = seed_registry.get(code)

    if entry is None:
        # Return "unverified" — NOT "not_found".
        # "Not found" implies the seed is safe but merely unknown.
        # "Unverified" correctly signals that the code could not be confirmed
        # as authentic, and the farmer should treat it with caution.
        return {
            "success": True,
            "code": code,
            "status": "unverified",
            "warning": (
                "This seed code was not found in the verified registry. "
                "This does NOT mean the seed is safe — it may be counterfeit, "
                "mislabelled, or from an unregistered batch. "
                "Do not use this seed until you have confirmed its authenticity "
                "with your local Krishi Vigyan Kendra (KVK) or agricultural office."
            ),
        }

    if entry["status"] == "invalid":
        return {
            "success": True,
            "code": code,
            "status": "invalid",
            "crop": entry["crop"],
            "batch": entry["batch"],
            "manufacturer": entry["manufacturer"],
            "cert_body": entry["cert_body"],
            "reason": entry.get("reason", "Batch is invalid or blacklisted"),
        }

    return {
        "success": True,
        "code": code,
        "status": "authentic",
        "crop": entry["crop"],
        "batch": entry["batch"],
        "manufacturer": entry["manufacturer"],
        "cert_body": entry["cert_body"],
        "certified_on": entry["certified_on"],
        "expires_on": entry["expires_on"],
    }
