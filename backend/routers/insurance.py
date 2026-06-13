"""
Crop Insurance Claim Assistant Router

Provides endpoints for:
- Submitting a crop insurance claim with damage photo analysis
- Retrieving claim details
- Exporting a structured PDF claim report
- Listing applicable government insurance schemes
"""

from __future__ import annotations

import base64
import io
import json
import logging
import math
import os
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

router = APIRouter()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Injected dependencies (set via init_insurance)
# ---------------------------------------------------------------------------
verify_role_fn = None


def init_insurance(verify_role):
    global verify_role_fn
    verify_role_fn = verify_role


# ---------------------------------------------------------------------------
# In-memory claim store (replace with Firestore in production)
# ---------------------------------------------------------------------------
_claims: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DAMAGE_LABELS = {
    "Low": {"pct_range": "0–25%", "color": "#22c55e"},
    "Medium": {"pct_range": "26–60%", "color": "#f59e0b"},
    "High": {"pct_range": "61–100%", "color": "#ef4444"},
}

_INSURANCE_SCHEMES = [
    {
        "id": "pmfby",
        "name": "Pradhan Mantri Fasal Bima Yojana (PMFBY)",
        "coverage": "Yield losses due to non-preventable natural risks",
        "premium_farmer": "2% for Kharif, 1.5% for Rabi, 5% for commercial crops",
        "claim_process": "Notify insurer within 72 hours of crop damage",
        "eligibility": "All farmers growing notified crops in notified areas",
        "portal": "https://pmfby.gov.in",
        "icon": "🌾",
    },
    {
        "id": "rwbcis",
        "name": "Restructured Weather Based Crop Insurance Scheme (RWBCIS)",
        "coverage": "Weather parameter deviations (rainfall, temperature, humidity)",
        "premium_farmer": "2% for Kharif, 1.5% for Rabi, 5% for commercial crops",
        "claim_process": "Automatic — triggered by weather station data",
        "eligibility": "Farmers in areas with weather stations under the scheme",
        "portal": "https://agricoop.nic.in",
        "icon": "🌦️",
    },
    {
        "id": "nais",
        "name": "National Agricultural Insurance Scheme (NAIS)",
        "coverage": "Natural calamities, pest attacks, and diseases",
        "premium_farmer": "Varies by crop and region (1.5%–3.5%)",
        "claim_process": "Submit claim within 15 days of crop cutting experiments",
        "eligibility": "All farmers — both loanee and non-loanee",
        "portal": "https://agricoop.nic.in",
        "icon": "🏛️",
    },
    {
        "id": "mnais",
        "name": "Modified NAIS (MNAIS)",
        "coverage": "Post-harvest losses and localised calamities included",
        "premium_farmer": "Actuarial rates with government subsidy",
        "claim_process": "Submit through insurance company with supporting documents",
        "eligibility": "Notified crop growers in notified areas",
        "portal": "https://agricoop.nic.in",
        "icon": "📋",
    },
]


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class RejectionDetail(BaseModel):
    reason: str
    recommended_action: str


class ClaimSummary(BaseModel):
    claim_id: str
    farmer_name: str
    crop_type: str
    season: str
    location: str
    damage_cause: str
    damage_severity: str
    estimated_loss_pct: float
    confidence_score: float
    submitted_at: str
    image_count: int
    status: str
    rejection_details: Optional[List[RejectionDetail]] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _estimate_damage_from_analysis(analysis: dict) -> dict:
    """Map crop disease analysis result to insurance damage metrics."""
    severity = analysis.get("severity", "Medium")
    confidence_score = float(analysis.get("confidenceScore", 60))
    disease = analysis.get("disease", "Unknown damage")

    severity_to_loss = {
        "Low": 15.0,
        "Medium": 42.0,
        "High": 73.0,
    }
    estimated_loss_pct = severity_to_loss.get(severity, 42.0)

    # Adjust loss estimate by confidence
    if confidence_score < 50:
        estimated_loss_pct *= 0.8  # reduce estimate when confidence is low

    return {
        "damage_severity": severity,
        "estimated_loss_pct": round(estimated_loss_pct, 1),
        "confidence_score": round(confidence_score, 1),
        "damage_description": disease,
        "treatment_hint": analysis.get("treatment", ""),
    }


def _analyse_image_locally(image_bytes: bytes) -> dict:
    """Local heuristic fallback when Gemini is unavailable."""
    try:
        import cv2
        import numpy as np

        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image")

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_s = float(np.mean(hsv[:, :, 1]))
        mean_v = float(np.mean(hsv[:, :, 2]))
        texture = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Simple severity heuristic based on saturation + texture
        damage_score = (150 - mean_s) / 150 + texture / 200 + (150 - mean_v) / 150
        if damage_score > 1.5:
            severity = "High"
            confidence = 68.0
        elif damage_score > 0.8:
            severity = "Medium"
            confidence = 62.0
        else:
            severity = "Low"
            confidence = 58.0

        return {
            "severity": severity,
            "confidenceScore": confidence,
            "disease": "Crop damage detected (heuristic analysis)",
            "treatment": "Consult local agricultural office for detailed assessment.",
        }
    except Exception as exc:
        logger.warning("Local image analysis failed: %s", exc)
        return {
            "severity": "Medium",
            "confidenceScore": 50.0,
            "disease": "Unable to analyze image",
            "treatment": "Manual inspection recommended.",
        }


async def _analyse_damage_image(image_bytes: bytes, crop_type: str) -> dict:
    """Analyze damage image via Gemini API or local fallback."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if api_key:
        try:
            import httpx

            image_b64 = base64.b64encode(image_bytes).decode()
            gemini_url = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"gemini-2.0-flash:generateContent?key={api_key}"
            )
            prompt = (
                "You are an agricultural damage assessor. "
                "Analyze this crop image for insurance claim purposes. "
                "Return ONLY valid JSON with keys: "
                "severity (Low/Medium/High), confidenceScore (0-100), "
                "disease (damage type description), treatment (recovery advice). "
                f"Crop type: {crop_type}. "
                "Focus on visible damage extent, discoloration, and crop loss indicators."
            )
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt},
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": image_b64,
                                }
                            },
                        ]
                    }
                ]
            }
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(gemini_url, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                text = (
                    data.get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "")
                )
                if text:
                    import re
                    match = re.search(r"\{.*\}", text, re.DOTALL)
                    if match:
                        return json.loads(match.group(0))
        except Exception as exc:
            logger.warning("Gemini damage analysis failed, using local fallback: %s", exc)

    return _analyse_image_locally(image_bytes)


def _generate_claim_pdf(claim: dict) -> bytes:
    """Generate a structured PDF claim report using reportlab."""
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # ── Header ──────────────────────────────────────────────────────────────
    pdf.setFillColor(colors.HexColor("#1a7340"))
    pdf.rect(0, height - 1.4 * inch, width, 1.4 * inch, fill=1, stroke=0)

    pdf.setFont("Helvetica-Bold", 22)
    pdf.setFillColor(colors.white)
    pdf.drawCentredString(width / 2, height - 0.65 * inch, "FASAL SAATHI")
    pdf.setFont("Helvetica", 12)
    pdf.drawCentredString(width / 2, height - 0.95 * inch, "Crop Insurance Claim Report")

    pdf.setFont("Helvetica", 9)
    pdf.setFillColor(colors.HexColor("#d1fae5"))
    pdf.drawRightString(
        width - 0.5 * inch,
        height - 1.25 * inch,
        f"Claim ID: {claim['claim_id']}",
    )

    # ── Claim Status Badge ───────────────────────────────────────────────────
    status = claim.get("status", "SUBMITTED").upper()
    status_color = colors.HexColor("#ef4444") if status == "REJECTED" else colors.HexColor("#f59e0b")
    pdf.setFillColor(status_color)
    pdf.roundRect(0.5 * inch, height - 1.9 * inch, 1.4 * inch, 0.32 * inch, 4, fill=1, stroke=0)
    pdf.setFont("Helvetica-Bold", 9)
    pdf.setFillColor(colors.white)
    pdf.drawCentredString(
        0.5 * inch + 0.7 * inch,
        height - 1.77 * inch,
        status,
    )

    pdf.setFont("Helvetica", 9)
    pdf.setFillColor(colors.HexColor("#6b7280"))
    pdf.drawRightString(
        width - 0.5 * inch,
        height - 1.77 * inch,
        f"Submitted: {claim.get('submitted_at', '')}",
    )

    y = height - 2.4 * inch

    def section_header(title, y_pos):
        pdf.setFillColor(colors.HexColor("#f0fdf4"))
        pdf.rect(0.5 * inch, y_pos - 0.05 * inch, width - 1 * inch, 0.28 * inch, fill=1, stroke=0)
        pdf.setFont("Helvetica-Bold", 11)
        pdf.setFillColor(colors.HexColor("#166534"))
        pdf.drawString(0.6 * inch, y_pos + 0.07 * inch, title)
        return y_pos - 0.45 * inch

    def field_row(label, value, y_pos):
        pdf.setFont("Helvetica-Bold", 10)
        pdf.setFillColor(colors.HexColor("#374151"))
        pdf.drawString(0.6 * inch, y_pos, label)
        pdf.setFont("Helvetica", 10)
        pdf.setFillColor(colors.HexColor("#111827"))
        pdf.drawString(2.8 * inch, y_pos, str(value))
        return y_pos - 0.28 * inch

    # ── Farmer Details ───────────────────────────────────────────────────────
    y = section_header("👤  Farmer Details", y)
    y = field_row("Farmer Name:", claim.get("farmer_name", "—"), y)
    y = field_row("Location:", claim.get("location", "—"), y)
    y -= 0.1 * inch

    # ── Crop Details ─────────────────────────────────────────────────────────
    y = section_header("🌾  Crop Details", y)
    y = field_row("Crop Type:", claim.get("crop_type", "—"), y)
    y = field_row("Season:", claim.get("season", "—"), y)
    y = field_row("Farm Area:", claim.get("farm_area", "—"), y)
    y -= 0.1 * inch

    # ── Damage Assessment ────────────────────────────────────────────────────
    y = section_header("⚠️  Damage Assessment", y)
    y = field_row("Cause of Damage:", claim.get("damage_cause", "—"), y)
    y = field_row("Damage Description:", claim.get("damage_description", "—"), y)

    severity = claim.get("damage_severity", "Medium")
    sev_color = {
        "Low": colors.HexColor("#22c55e"),
        "Medium": colors.HexColor("#f59e0b"),
        "High": colors.HexColor("#ef4444"),
    }.get(severity, colors.HexColor("#f59e0b"))

    pdf.setFont("Helvetica-Bold", 10)
    pdf.setFillColor(colors.HexColor("#374151"))
    pdf.drawString(0.6 * inch, y, "Damage Severity:")
    pdf.setFillColor(sev_color)
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(2.8 * inch, y, severity)
    y -= 0.28 * inch

    y = field_row(
        "Estimated Loss:",
        f"{claim.get('estimated_loss_pct', 0):.1f}%",
        y,
    )
    y = field_row(
        "AI Confidence Score:",
        f"{claim.get('confidence_score', 0):.1f}%",
        y,
    )
    y = field_row(
        "Evidence Photos:",
        f"{claim.get('image_count', 0)} image(s) uploaded",
        y,
    )
    y -= 0.1 * inch

    # ── Recovery Notes ───────────────────────────────────────────────────────
    if claim.get("treatment_hint") and claim.get("status") != "Rejected":
        y = section_header("💡  Recovery Guidance", y)
        pdf.setFont("Helvetica", 9)
        pdf.setFillColor(colors.HexColor("#374151"))
        # Word-wrap the treatment text
        words = claim["treatment_hint"].split()
        line, lines = [], []
        for word in words:
            if len(" ".join(line + [word])) <= 90:
                line.append(word)
            else:
                lines.append(" ".join(line))
                line = [word]
        if line:
            lines.append(" ".join(line))
        for text_line in lines[:4]:
            pdf.drawString(0.6 * inch, y, text_line)
            y -= 0.22 * inch
        y -= 0.05 * inch

    # ── Rejection Details ────────────────────────────────────────────────────
    if claim.get("status") == "Rejected" and claim.get("rejection_details"):
        y = section_header("❌  Rejection Details", y)
        for detail in claim.get("rejection_details", []):
            pdf.setFont("Helvetica-Bold", 10)
            pdf.setFillColor(colors.HexColor("#ef4444"))
            pdf.drawString(0.6 * inch, y, f"Reason: {detail.get('reason', '')}")
            y -= 0.20 * inch
            pdf.setFont("Helvetica", 9)
            pdf.setFillColor(colors.HexColor("#16a34a"))
            pdf.drawString(0.8 * inch, y, f"Recommended Action: {detail.get('recommended_action', '')}")
            y -= 0.26 * inch
        y -= 0.1 * inch

    # ── Footer ───────────────────────────────────────────────────────────────
    pdf.setStrokeColor(colors.HexColor("#d1d5db"))
    pdf.line(0.5 * inch, 0.8 * inch, width - 0.5 * inch, 0.8 * inch)
    pdf.setFont("Helvetica", 8)
    pdf.setFillColor(colors.HexColor("#9ca3af"))
    pdf.drawCentredString(
        width / 2,
        0.55 * inch,
        "This report is generated by Fasal Saathi. "
        "Submit to your insurer with supporting evidence.",
    )
    pdf.drawCentredString(
        width / 2,
        0.38 * inch,
        f"Report generated on {datetime.now().strftime('%d %B %Y, %I:%M %p')}",
    )

    pdf.showPage()
    pdf.save()
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/insurance/claim")
async def submit_insurance_claim(
    request: Request,
    farmer_name: str = Form(..., max_length=100),
    crop_type: str = Form(..., max_length=50),
    season: str = Form(..., max_length=50),
    location: str = Form(..., max_length=150),
    farm_area: str = Form(..., max_length=50),
    damage_cause: str = Form(..., max_length=100),
    images: List[UploadFile] = File(...),
    simulate_rejection: Optional[str] = Form(None),
):
    """
    Submit a crop insurance claim with damage photos.

    Accepts up to 5 images. Each image is analyzed for damage severity.
    The worst-case (highest severity) analysis is used for the claim.
    Returns a structured claim object with AI damage assessment.
    """
    if verify_role_fn is None:
        raise HTTPException(status_code=500, detail="Auth service not initialized")

    token_data = await verify_role_fn(request)
    uid = token_data.get("uid")
    if not uid:
        raise HTTPException(status_code=401, detail="Invalid authentication token")

    # Validate image count
    if len(images) == 0:
        raise HTTPException(status_code=422, detail="At least one damage photo is required")
    if len(images) > 5:
        raise HTTPException(status_code=422, detail="Maximum 5 images allowed per claim")

    # Analyse all images, pick worst severity
    severity_rank = {"Low": 0, "Medium": 1, "High": 2}
    best_analysis = None

    for img_file in images:
        content_type = img_file.content_type or ""
        if not content_type.startswith("image/"):
            raise HTTPException(
                status_code=422,
                detail=f"File '{img_file.filename}' is not a valid image",
            )

        raw = await img_file.read()
        if len(raw) > 10 * 1024 * 1024:  # 10 MB cap
            raise HTTPException(status_code=422, detail="Each image must be under 10 MB")

        try:
            analysis = await _analyse_damage_image(raw, crop_type)
        except Exception as exc:
            logger.warning("Image analysis failed for %s: %s", img_file.filename, exc)
            analysis = {"severity": "Medium", "confidenceScore": 50.0, "disease": "Analysis failed"}

        if best_analysis is None or (
            severity_rank.get(analysis.get("severity", "Medium"), 1)
            > severity_rank.get(best_analysis.get("severity", "Medium"), 1)
        ):
            best_analysis = analysis

    damage = _estimate_damage_from_analysis(best_analysis or {})

    claim_id = str(uuid.uuid4())[:8].upper()
    is_rejected = simulate_rejection == "true"
    
    claim = {
        "claim_id": claim_id,
        "owner_uid": uid,
        "farmer_name": farmer_name,
        "crop_type": crop_type,
        "season": season,
        "location": location,
        "farm_area": farm_area,
        "damage_cause": damage_cause,
        "damage_severity": damage["damage_severity"],
        "estimated_loss_pct": damage["estimated_loss_pct"],
        "confidence_score": damage["confidence_score"],
        "damage_description": damage["damage_description"],
        "treatment_hint": damage["treatment_hint"],
        "image_count": len(images),
        "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "status": "Rejected" if is_rejected else "Submitted",
        "rejection_details": [
            {
                "reason": "Missing land ownership proof",
                "recommended_action": "Upload Khasra document (Land record details)",
            },
            {
                "reason": "Unclear damage photographs",
                "recommended_action": "Retake and upload high-resolution photos under clear daylight",
            },
            {
                "reason": "Acreage mismatch with registered area",
                "recommended_action": "Provide correct farm dimensions or update land registration profile",
            }
        ] if is_rejected else None,
    }
    _claims[claim_id] = claim

    return {
        "success": True,
        "claim": claim,
        "applicable_schemes": _INSURANCE_SCHEMES,
    }


@router.get("/insurance/claim/{claim_id}")
async def get_claim(request: Request, claim_id: str):
    """Retrieve a previously submitted claim by its ID. Owner verification required."""
    if verify_role_fn is None:
        raise HTTPException(status_code=500, detail="Auth service not initialized")

    token_data = await verify_role_fn(request)
    uid = token_data.get("uid")
    if not uid:
        raise HTTPException(status_code=401, detail="Invalid authentication token")

    claim = _claims.get(claim_id.upper())
    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found")

    # Verify ownership
    if claim.get("owner_uid") != uid:
        raise HTTPException(status_code=403, detail="You do not have permission to access this claim")

    return {"success": True, "claim": claim, "applicable_schemes": _INSURANCE_SCHEMES}


@router.get("/insurance/claims")
async def list_claims(request: Request):
    """Retrieve all claims submitted by the current user."""
    if verify_role_fn is None:
        raise HTTPException(status_code=500, detail="Auth service not initialized")

    token_data = await verify_role_fn(request)
    uid = token_data.get("uid")
    if not uid:
        raise HTTPException(status_code=401, detail="Invalid authentication token")

    user_claims = [c for c in _claims.values() if c.get("owner_uid") == uid]
    return {"success": True, "claims": user_claims}


@router.get("/insurance/claim/{claim_id}/export")
async def export_claim_pdf(request: Request, claim_id: str):
    """Export a claim as a downloadable PDF report. Owner verification required."""
    if verify_role_fn is None:
        raise HTTPException(status_code=500, detail="Auth service not initialized")

    token_data = await verify_role_fn(request)
    uid = token_data.get("uid")
    if not uid:
        raise HTTPException(status_code=401, detail="Invalid authentication token")

    claim = _claims.get(claim_id.upper())
    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found")

    # Verify ownership
    if claim.get("owner_uid") != uid:
        raise HTTPException(status_code=403, detail="You do not have permission to access this claim")

    try:
        pdf_bytes = _generate_claim_pdf(claim)
    except Exception as exc:
        logger.error("PDF generation failed for claim %s: %s", claim_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate claim report")

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=InsuranceClaim_{claim_id}.pdf"
        },
    )


@router.get("/insurance/schemes")
async def list_insurance_schemes():
    """Return all supported government crop insurance schemes (public endpoint)."""
    return {"success": True, "schemes": _INSURANCE_SCHEMES}
