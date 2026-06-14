"""Reports & Logging Router"""
import re
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Optional
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

# ---------------------------------------------------------------------------
# Validation bounds — intentionally generous to accommodate large commercial
# farms while still rejecting obviously fabricated figures.
# ---------------------------------------------------------------------------
_PROFIT_MAX_INR = 50_000_000   # ₹5 crore per season
_AREA_MAX_ACRES = 10_000       # 10,000 acres

# Regex that matches a valid Indian-locale number string produced by the
# frontend (e.g. "50,000" or "1,00,000") or a plain integer string.
_NUMERIC_RE = re.compile(r"^[\d,]+(\.\d+)?$")


def _parse_inr(value: str) -> float:
    """Strip commas and parse as float. Raises ValueError on invalid input."""
    cleaned = value.replace(",", "").strip()
    if not cleaned:
        raise ValueError("Value is empty")
    return float(cleaned)


def _parse_acres(value: str) -> float:
    """Extract the numeric part from strings like '5 Acres' or '5.5'."""
    cleaned = value.lower().replace("acres", "").replace("acre", "").strip()
    cleaned = cleaned.replace(",", "")
    if not cleaned:
        raise ValueError("Value is empty")
    return float(cleaned)


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

    @validator("profit")
    def validate_profit(cls, v):
        try:
            amount = _parse_inr(v)
        except (ValueError, TypeError):
            raise ValueError("Profit must be a valid number (e.g. 50000 or 50,000).")
        if amount < 0:
            raise ValueError("Profit cannot be negative.")
        if amount > _PROFIT_MAX_INR:
            raise ValueError(
                f"Profit cannot exceed ₹{_PROFIT_MAX_INR:,} per season. "
                "If your farm genuinely exceeds this, contact support."
            )
        return v

    @validator("area")
    def validate_area(cls, v):
        try:
            acres = _parse_acres(v)
        except (ValueError, TypeError):
            raise ValueError("Farm area must be a valid number of acres (e.g. '5 Acres' or '5.5').")
        if acres <= 0:
            raise ValueError("Farm area must be greater than zero.")
        if acres > _AREA_MAX_ACRES:
            raise ValueError(
                f"Farm area cannot exceed {_AREA_MAX_ACRES:,} acres. "
                "If your farm genuinely exceeds this, contact support."
            )
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

def _build_pdf(data: ReportRequest, signature_hex: str, cert_id: str, generated_at: str) -> bytes:
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
    c.drawRightString(width - inch, height - 110, f"Generated: {generated_at}")

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


def _sign_report(private_key: Ed25519PrivateKey, data: ReportRequest, cert_id: str, generated_at: str) -> str:
    """Return a hex-encoded Ed25519 signature over the canonical report payload."""
    payload = json.dumps(
        {
            "cert_id": cert_id,
            "generated_at": generated_at,
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
        generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        signature_hex = _sign_report(private_key, data, cert_id, generated_at)
        pdf_bytes = _build_pdf(data, signature_hex, cert_id, generated_at)
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


# ---------------------------------------------------------------------------
# Admin: role assignment with custom-claim sync
# ---------------------------------------------------------------------------

class AssignRoleRequest(BaseModel):
    target_uid: str = Field(..., min_length=1, max_length=128)
    role: str = Field(..., pattern=r"^(admin|expert|farmer|vendor|system|guest)$")


@router.post("/admin/assign-role")
async def assign_role(request: Request, body: AssignRoleRequest):
    """
    Assign a role to a user and sync the Firebase custom claim.

    Admin only.  Updates both the Firestore users/{uid}.role field and the
    Firebase Auth custom claim so that Firestore security rules
    (request.auth.token.role) stay consistent with the stored role.

    The target user's next token refresh will include the updated claim.
    Existing tokens remain valid until they expire (≤ 1 hour).
    """
    if verify_role_fn is None:
        raise HTTPException(status_code=500, detail="Not initialized")

    await verify_role_fn(request, required_roles=["admin"])

    import firebase_admin
    from firebase_admin import firestore as _fs
    from role_sync import sync_role_claim, VALID_ROLES

    if not firebase_admin._apps:
        raise HTTPException(status_code=503, detail="Authorization service unavailable")

    try:
        db = _fs.client()
        user_ref = db.collection("users").document(body.target_uid)
        snap = user_ref.get()
        if not snap.exists:
            raise HTTPException(status_code=404, detail="User profile not found")

        user_ref.update({"role": body.role})
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("assign_role: Firestore update failed uid=%s: %s", body.target_uid, exc)
        raise HTTPException(status_code=503, detail="Authorization service unavailable")

    try:
        await sync_role_claim(body.target_uid, body.role)
    except Exception as exc:
        # Firestore write succeeded; log the claim-sync failure but don't
        # roll back — the backend verify_role still reads from Firestore,
        # so access control is not broken.  The claim will be corrected on
        # the next assign-role call or backfill run.
        logger.error(
            "assign_role: custom claim sync failed uid=%s role=%s: %s",
            body.target_uid, body.role, exc,
        )

    return {
        "success": True,
        "target_uid": body.target_uid,
        "role": body.role,
        "message": "Role updated. The user's next token refresh will include the new claim.",
    }


@router.post("/admin/backfill-role-claims")
async def backfill_role_claims_endpoint(request: Request):
    """
    One-time backfill: set the 'role' custom claim for every user in Firestore.

    Admin only.  Safe to call multiple times — idempotent.  Run this once
    after deploying the Firestore rules change that switched from get()-based
    role checks to custom-claim-based checks.
    """
    if verify_role_fn is None:
        raise HTTPException(status_code=500, detail="Not initialized")

    await verify_role_fn(request, required_roles=["admin"])

    import firebase_admin
    from firebase_admin import firestore as _fs
    from role_sync import backfill_role_claims

    if not firebase_admin._apps:
        raise HTTPException(status_code=503, detail="Authorization service unavailable")

    try:
        db = _fs.client()
        summary = backfill_role_claims(db)
    except Exception as exc:
        logger.error("backfill_role_claims: failed: %s", exc)
        raise HTTPException(status_code=500, detail="Backfill failed — check server logs")

    return {"success": True, **summary}
