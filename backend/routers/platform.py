"""Platform router for cross-cutting endpoints.

This module hosts endpoints that don't belong to a single domain router,
keeping main.py focused on application wiring only.
"""

import hashlib
import io
import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Form, HTTPException, Request, Response
from pydantic import BaseModel, Field, validator
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from error_utils import safe_detail
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

router = APIRouter()
logger = logging.getLogger(__name__)


class WhatsAppSubscribeRequest(BaseModel):
    phone_number: str = Field(..., min_length=7, max_length=20)
    name: str = Field(..., min_length=1, max_length=100)
    # user_id is accepted for backward-compatibility but is IGNORED.
    # The authoritative identity is always derived from the verified
    # Firebase ID token — never from client-supplied data.
    user_id: Optional[str] = None


class AlertTriggerRequest(BaseModel):
    alert_type: str = Field(..., pattern=r'^(weather|pest|advisory)$')
    message: str = Field(..., min_length=1, max_length=500)


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


class RAGQuery(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    top_k: int = Field(default=3, ge=1, le=5)

    @validator("query")
    def sanitize_and_normalize_query(cls, value):
        if not value or not isinstance(value, str):
            raise ValueError("Query must be a non-empty string.")

        value = re.sub(r"<script.*?>.*?</script>", "", value, flags=re.IGNORECASE | re.DOTALL)
        value = re.sub(r"</?script.*?>", "", value, flags=re.IGNORECASE)
        value = re.sub(r"on\w+\s*=", "", value, flags=re.IGNORECASE)
        value = re.sub(r"javascript:", "", value, flags=re.IGNORECASE)
        value = re.sub(r"data:", "", value, flags=re.IGNORECASE)
        value = re.sub(r"vbscript:", "", value, flags=re.IGNORECASE)
        value = re.sub(r"<[^>]*>", "", value)
        value = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", value)
        value = re.sub(r"[*_~`#]", "", value)
        value = re.sub(r"\s+", " ", value.strip())

        forbidden_patterns = [
            r"ignore\s+(?:all\s+)?previous\s+instructions",
            r"ignore\s+(?:the\s+)?system\s+prompt",
            r"override\s+system\s+constraints",
            r"developer\s+mode",
            r"bypass\s+safety\s+filter",
            r"disregard\s+(?:all\s+)?prior\s+instructions",
            r"act\s+as\s+(?:a\s+)?(?:different|unrestricted|unfiltered)\s+(?:ai|model|assistant)",
            r"pretend\s+(?:you\s+are|to\s+be)\s+(?:a\s+)?(?:different|unrestricted)",
            r"jailbreak",
            r"prompt\s+injection",
        ]

        lowered = value.lower()
        for pattern in forbidden_patterns:
            if re.search(pattern, lowered):
                raise ValueError("Query contains disallowed phrases or prompt injection attempts.")

        if len(value) < 3:
            raise ValueError("Query must be at least 3 characters long after sanitization.")

        return value


class GeminiImageRequest(BaseModel):
    image_base64: str = Field(..., min_length=10, description="Base64-encoded image data")
    mime_type: str = Field(..., pattern=r"^image/(jpeg|png|gif|webp)$", description="MIME type of the image")
    prompt: str = Field(..., min_length=10, max_length=2000, description="Analysis prompt")

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
async def get_alerts_history():
    if weather_service is None:
        raise HTTPException(status_code=503, detail="Weather service unavailable")

    recent_alerts = weather_service.alert_history[-50:]
    return {
        "success": True,
        "total_alerts": len(weather_service.alert_history),
        "recent_alerts": [alert.to_dict() for alert in recent_alerts],
    }


@router.post("/whatsapp/subscribe")
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
    uid = token_data["uid"]

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
        send_whatsapp_message_fn(data.phone_number, welcome_msg)

    return {"success": True, "message": "Successfully subscribed"}


@router.post("/whatsapp/trigger-alert")
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
        result = send_whatsapp_message_fn(info["phone_number"], formatted_msg)
        results.append({"user_id": user_id, "success": result.get("success", False)})

    delivered = sum(1 for r in results if r["success"])
    return {"success": True, "results": results, "delivered": delivered, "total": len(results)}


@router.post("/whatsapp/webhook")
async def whatsapp_webhook(Body: str = Form(...), From: str = Form(...)):
    """
    Handle incoming WhatsApp messages from Twilio.
    
    Processing is offloaded to a background Celery task to immediately
    acknowledge the webhook (preventing Twilio timeout/penalties under burst traffic)
    and process the message asynchronously.
    """
    if send_whatsapp_message_fn is None:
        raise HTTPException(status_code=500, detail="WhatsApp sender not initialized")

    sender_number = From.replace("whatsapp:", "")

    # Offload message processing to reliable background task queue
    from celery_worker import process_whatsapp_webhook_task
    process_whatsapp_webhook_task.delay(Body, sender_number)
    
    return {"status": "success"}


@router.post("/reports/generate")
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
    signature = private_key.sign(report_data_string.encode("utf-8"))
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
async def log_error(body: ClientErrorReport):
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
async def rag_query(body: RAGQuery):
    if rag_generate_fn is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not available")

    try:
        return rag_generate_fn(body.query, top_k=body.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=safe_detail(exc, 500))


@router.post("/gemini/analyze-image")
async def gemini_analyze_image(body: GeminiImageRequest):
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


@router.post("/simulate-climate")
async def simulate_climate(data: SimulationRequest):
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
