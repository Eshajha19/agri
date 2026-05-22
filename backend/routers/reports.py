"""Reports & Logging Router"""
import base64
import hashlib
import io
import json
import logging
from datetime import datetime

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from typing import Optional

router = APIRouter()
logger = logging.getLogger(__name__)


class ClientErrorReport(BaseModel):
    message: str = Field(..., min_length=1, max_length=500)
    source: Optional[str] = Field(default=None, max_length=200)
    stack: Optional[str] = Field(default=None, max_length=2000)
    level: str = Field(default="error", max_length=20)


class ReportRequest(BaseModel):
    name: str = Field(..., max_length=100)
    crop: str = Field(..., max_length=50)
    area: str = Field(..., max_length=50)
    profit: str = Field(..., max_length=50)
    season: str = Field(..., max_length=50)

    @validator("name", "crop", "area", "profit", "season", pre=True)
    def reject_pipe_characters(cls, v):
        if isinstance(v, str) and "|" in v:
            raise ValueError("Field value must not contain the '|' character.")
        return v


verify_role_fn = None
get_signing_keys_fn = None
sanitise_log_field_fn = None
logger_instance = None


def init_reports(vr_fn, gsk_fn, slf_fn, log_inst):
    global verify_role_fn, get_signing_keys_fn, sanitise_log_field_fn, logger_instance
    verify_role_fn = vr_fn
    get_signing_keys_fn = gsk_fn
    sanitise_log_field_fn = slf_fn
    logger_instance = log_inst


# ---------------------------------------------------------------------------
# PDF generation helpers
# ---------------------------------------------------------------------------

def _build_pdf(data: ReportRequest, signature_hex: str, cert_id: str) -> bytes:
    """Render a bank-ready financial report as a PDF and return the raw bytes."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter

    # ── Header bar ──────────────────────────────────────────────────────────
    c.setFillColor(colors.HexColor("#2E7D32"))
    c.rect(0, height - 80, width, 80, fill=True, stroke=False)

    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 22)
    c.drawString(inch, height - 45, "Fasal Saathi")
    c.setFont("Helvetica", 12)
    c.drawString(inch, height - 65, "Certified Agricultural Financial Report")

    # ── Certificate ID & date ────────────────────────────────────────────────
    c.setFillColor(colors.HexColor("#1B5E20"))
    c.setFont("Helvetica-Bold", 10)
    c.drawRightString(width - inch, height - 95, f"Certificate ID: {cert_id}")
    c.drawRightString(width - inch, height - 110, f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

    # ── Section: Farmer Details ──────────────────────────────────────────────
    y = height - 150
    c.setFillColor(colors.HexColor("#2E7D32"))
    c.setFont("Helvetica-Bold", 14)
    c.drawString(inch, y, "Farmer Details")
    c.setStrokeColor(colors.HexColor("#2E7D32"))
    c.line(inch, y - 4, width - inch, y - 4)

    fields = [
        ("Farmer Name", data.name),
        ("Crop Type", data.crop),
        ("Farm Area", data.area),
        ("Season", data.season),
        ("Estimated Profit (₹)", data.profit),
    ]

    y -= 24
    c.setFont("Helvetica", 11)
    for label, value in fields:
        c.setFillColor(colors.HexColor("#555555"))
        c.drawString(inch, y, f"{label}:")
        c.setFillColor(colors.black)
        c.drawString(3 * inch, y, str(value))
        y -= 20

    # ── Section: Cryptographic Signature ────────────────────────────────────
    y -= 20
    c.setFillColor(colors.HexColor("#2E7D32"))
    c.setFont("Helvetica-Bold", 14)
    c.drawString(inch, y, "Cryptographic Signature (Ed25519)")
    c.line(inch, y - 4, width - inch, y - 4)

    y -= 24
    c.setFont("Helvetica", 8)
    c.setFillColor(colors.HexColor("#333333"))

    # Wrap the hex signature across multiple lines (64 chars each)
    chunk_size = 64
    sig_lines = [signature_hex[i:i + chunk_size] for i in range(0, len(signature_hex), chunk_size)]
    for line in sig_lines:
        c.drawString(inch, y, line)
        y -= 12

    # ── Footer ───────────────────────────────────────────────────────────────
    c.setFillColor(colors.HexColor("#EEEEEE"))
    c.rect(0, 0, width, 50, fill=True, stroke=False)
    c.setFillColor(colors.HexColor("#555555"))
    c.setFont("Helvetica", 8)
    c.drawCentredString(
        width / 2,
        30,
        "This document is cryptographically signed and cannot be altered after generation.",
    )
    c.drawCentredString(
        width / 2,
        18,
        "Verify authenticity at: https://fasalsaathi.in/verify",
    )

    c.save()
    return buf.getvalue()


def _sign_report(private_key: Ed25519PrivateKey, data: ReportRequest, cert_id: str) -> str:
    """Return a hex-encoded Ed25519 signature over the canonical report payload."""
    payload = json.dumps(
        {
            "cert_id": cert_id,
            "name": data.name,
            "crop": data.crop,
            "area": data.area,
            "profit": data.profit,
            "season": data.season,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    signature_bytes = private_key.sign(payload)
    return signature_bytes.hex()


def _make_cert_id(data: ReportRequest) -> str:
    """Derive a short, deterministic certificate ID from the report fields."""
    raw = f"{data.name}|{data.crop}|{data.season}|{datetime.utcnow().strftime('%Y%m%d')}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10].upper()
    return f"CERT-{digest}"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/reports/generate")
async def generate_signed_report(request: Request, data: ReportRequest):
    """Generate a cryptographically signed PDF bank report.

    The endpoint:
    1. Verifies the caller's Firebase auth token.
    2. Loads the Ed25519 signing key via get_signing_keys_fn.
    3. Signs a canonical JSON payload of the report fields.
    4. Renders a PDF with ReportLab embedding the signature.
    5. Returns the PDF as an application/pdf streaming response so the
       frontend's responseType:'blob' download works correctly.
    """
    if not all([verify_role_fn, get_signing_keys_fn]):
        raise HTTPException(status_code=500, detail="Not initialized")

    try:
        await verify_role_fn(request)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Auth error: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed")

    try:
        private_key = get_signing_keys_fn()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Key retrieval error: {e}")
        raise HTTPException(status_code=500, detail="Failed to load signing key")

    try:
        cert_id = _make_cert_id(data)
        signature_hex = _sign_report(private_key, data, cert_id)
        pdf_bytes = _build_pdf(data, signature_hex, cert_id)
    except Exception as e:
        logger.error(f"PDF generation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate report")

    filename = f"FasalSaathi_BankReport_{cert_id}.pdf"
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(pdf_bytes)),
        },
    )


@router.post("/log-error")
async def log_error(request: Request, body: ClientErrorReport):
    if sanitise_log_field_fn is None or logger_instance is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    try:
        message = sanitise_log_field_fn(body.message)
        source = sanitise_log_field_fn(body.source or "")
        level = sanitise_log_field_fn(body.level).upper()
        logger_instance.info(f"Client [{level}] from {source}: {message}")
        return {"success": True, "message": "Error logged"}
    except Exception as e:
        logger.error(f"Log error: {e}")
        raise HTTPException(status_code=500, detail="Failed to log error")
