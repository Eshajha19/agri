"""Reports & Logging Router"""
import re
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Optional
import base64
import hashlib
import io
import json
import logging
from datetime import datetime, timezone
import time, uuid
from fastapi import APIRouter, HTTPException

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

from backend.core.logging_config import setup_logging

logger = setup_logging(__name__)

router = APIRouter()

# Simple in-memory nonce store (replace with Redis/Firestore in production)
used_nonces = set()
SIGNATURE_TTL = 300  # 5 minutes

@router.post("/submit-report")
def submit_report(payload: dict):
    nonce = payload.get("nonce")
    timestamp = payload.get("timestamp")
    signature = payload.get("signature")

    # ✅ Nonce check
    if not nonce or nonce in used_nonces:
        raise HTTPException(status_code=400, detail="Invalid or replayed nonce")
    used_nonces.add(nonce)

    # ✅ Timestamp check
    now = int(time.time())
    if not timestamp or abs(now - int(timestamp)) > SIGNATURE_TTL:
        raise HTTPException(status_code=400, detail="Signature expired")

    # ✅ Signature verification (pseudo-code)
    if not verify_signature(payload, signature):
        raise HTTPException(status_code=400, detail="Invalid signature")

    return {"status": "accepted"}

# ---------------------------------------------------------------------------
# Validation bounds — intentionally generous to accommodate large commercial
# farms while still rejecting obviously fabricated figures.
# ---------------------------------------------------------------------------
_PROFIT_MAX_INR = 50_000_000   # ₹5 crore per season
_AREA_MAX_ACRES = 10_000       # 10,000 acres
MAX_EXPORT_RECORDS = 5000      # Future safeguard for large dataset exports
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


from typing import Callable, Optional

def init_reports(
    vr_fn: Callable,
    gsk_fn: Callable,
    slf_fn: Optional[Callable] = None,
) -> None:
    """
    Initialize report router dependencies.

    Args:
        vr_fn: Authentication/authorization verifier.
        gsk_fn: Signing key provider.
        slf_fn: Optional log field sanitization function.

    Raises:
        ValueError: If required dependencies are missing.
    """
    global verify_role_fn, get_signing_keys_fn, sanitise_log_field_fn

    if vr_fn is None:
        raise ValueError("verify_role_fn cannot be None")

    if gsk_fn is None:
        raise ValueError("get_signing_keys_fn cannot be None")

    verify_role_fn = vr_fn
    get_signing_keys_fn = gsk_fn
    sanitise_log_field_fn = slf_fn

    logger.info(
        "reports.router.initialized "
        "auth_provider=%s "
        "key_provider=%s "
        "sanitizer_enabled=%s",
        getattr(vr_fn, "__name__", type(vr_fn).__name__),
        getattr(gsk_fn, "__name__", type(gsk_fn).__name__),
        slf_fn is not None,
    )


# ---------------------------------------------------------------------------
# PDF generation helpers
# ---------------------------------------------------------------------------

def _validate_report_integrity(
    data: ReportRequest,
    cert_id: str,
    signature_hex: str,
    pdf_bytes: bytes,
):
    validation = {
        "valid": True,
        "checks": [],
        "warnings": [],
    }

    required_fields = {
        "name": data.name,
        "crop": data.crop,
        "area": data.area,
        "profit": data.profit,
        "season": data.season,
    }

    missing_fields = [
        field
        for field, value in required_fields.items()
        if not str(value).strip()
    ]

    if missing_fields:
        validation["valid"] = False
        validation["warnings"].append(
            f"missing_fields:{','.join(missing_fields)}"
        )

    if not cert_id:
        validation["valid"] = False
        validation["warnings"].append(
            "missing_certificate_id"
        )

    else:
        validation["checks"].append(
            "certificate_id_present"
        )

    if not signature_hex:
        validation["valid"] = False
        validation["warnings"].append(
            "missing_signature"
        )

    else:
        validation["checks"].append(
            "signature_present"
        )

    if not pdf_bytes:
        validation["valid"] = False
        validation["warnings"].append(
            "empty_pdf"
        )

    if pdf_bytes:
        validation["checks"].append(
            "pdf_generated"
        )

        if len(pdf_bytes) < 1000:
            validation["valid"] = False
            validation["warnings"].append(
                "pdf_content_suspiciously_small"
            )

        if not pdf_bytes.startswith(b"%PDF"):
            validation["valid"] = False
            validation["warnings"].append(
                "invalid_pdf_header"
            )

    validation["checks"].append(
        "required_field_validation"
    )

    return validation

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
    c.drawRightString(width - inch, height - 110, f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

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
    """Generate a unique certificate ID for each report request.

    A random 8-byte nonce is mixed into the hash input so that two requests
    with identical field values (same farmer name, crop, and season on the
    same day) always produce different IDs. Without the nonce, the ID is fully
    deterministic and collides silently on repeated submissions, causing the
    second PDF to overwrite or be confused with the first in any downstream
    system that indexes by certificate ID.
    """
    import secrets
    nonce = secrets.token_hex(8)
    raw = f"{data.name}|{data.crop}|{data.season}|{datetime.now(timezone.utc).strftime('%Y%m%d')}|{nonce}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10].upper()
    return f"CERT-{digest}"


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_signed_report(
    name: str,
    crop: str,
    area: str,
    profit: str,
    season: str,
    cert_id: str,
    signature_hex: str,
    public_key_pem: Optional[str] = None,
) -> dict:
    """Verify an Ed25519-signed report payload.

    Parameters
    ----------
    name, crop, area, profit, season : str
        Report field values (must match what was signed).
    cert_id : str
        Certificate ID embedded during signing.
    signature_hex : str
        Hex-encoded Ed25519 signature.
    public_key_pem : str, optional
        PEM-encoded public key.  When *not* provided the function attempts
        to load ``keys/report_signing.pub`` from the configured path.

    Returns
    -------
    dict with keys:
        ``verified`` (bool), ``cert_id`` (str), ``key_fingerprint`` (str),
        and on failure ``error`` (str).
    """
    # 1. Rebuild the canonical payload exactly as _sign_report did.
    canonical = json.dumps(
        {
            "cert_id": cert_id,
            "name": name,
            "crop": crop,
            "area": area,
            "profit": profit,
            "season": season,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    # 2. Decode the signature.
    try:
        signature = bytes.fromhex(signature_hex)
    except ValueError:
        return {"verified": False, "cert_id": cert_id, "error": "Invalid signature hex encoding"}

    # 3. Load the public key.
    if public_key_pem:
        try:
            public_key = serialization.load_pem_public_key(public_key_pem.encode("utf-8"))
        except Exception:
            return {"verified": False, "cert_id": cert_id, "error": "Invalid public key PEM"}
    else:
        try:
            from main import PUBLIC_KEY_PATH as _pk_path
            with open(_pk_path, "rb") as f:
                public_key = serialization.load_pem_public_key(f.read())
        except Exception:
            return {"verified": False, "cert_id": cert_id, "error": "Unable to load public key"}

    if not isinstance(public_key, Ed25519PublicKey):
        return {"verified": False, "cert_id": cert_id, "error": "Key is not an Ed25519 public key"}

    # 4. Compute fingerprint of the public key (SHA-256 of DER).
    try:
        der = public_key.public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        key_fingerprint = hashlib.sha256(der).hexdigest()[:16]
    except Exception:
        key_fingerprint = "unknown"

    # 5. Verify.
    try:
        public_key.verify(signature, canonical)
        return {"verified": True, "cert_id": cert_id, "key_fingerprint": key_fingerprint}
    except InvalidSignature:
        return {"verified": False, "cert_id": cert_id, "key_fingerprint": key_fingerprint, "error": "Signature does not match"}
    except Exception as e:
        return {"verified": False, "cert_id": cert_id, "key_fingerprint": key_fingerprint, "error": str(e)}


class VerifyReportRequest(BaseModel):
    name: str = Field(..., max_length=100)
    crop: str = Field(..., max_length=50)
    area: str = Field(..., max_length=50)
    profit: str = Field(..., max_length=50)
    season: str = Field(..., max_length=50)
    cert_id: str = Field(..., min_length=1, max_length=100)
    signature: str = Field(..., min_length=1, max_length=256)
    public_key_pem: Optional[str] = Field(default=None, max_length=2000)


@router.post("/reports/verify")
async def verify_report_endpoint(body: VerifyReportRequest):
    """Verify an Ed25519-signed report payload.

    Accepts the same report fields that were submitted to ``/reports/generate``
    plus the ``cert_id`` and hex ``signature`` embedded in the PDF.

    Returns ``{"verified": true/false, "cert_id": "...", "key_fingerprint": "..."}``.
    """
    result = verify_signed_report(
        name=body.name,
        crop=body.crop,
        area=body.area,
        profit=body.profit,
        season=body.season,
        cert_id=body.cert_id,
        signature_hex=body.signature,
        public_key_pem=body.public_key_pem,
    )
    status_code = 200 if result["verified"] else 400
    return JSONResponse(content=result, status_code=status_code)


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
        await verify_role_fn(request, required_roles=["admin"])
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

        validation = _validate_report_integrity(
            data,
            cert_id,
            signature_hex,
            pdf_bytes,
        )

        if not validation["valid"]:
            logger.error(
                "Report integrity validation failed: %s",
                validation,
            )

            raise HTTPException(
                status_code=500,
                detail="Generated report failed integrity validation",
            )

        logger.info(
            "[REPORT_VALIDATION] cert_id=%s checks=%s warnings=%s",
            cert_id,
            validation["checks"],
            validation["warnings"],
        )

        MAX_PDF_SIZE_MB = 10

        if len(pdf_bytes) > MAX_PDF_SIZE_MB * 1024 * 1024:
            logger.error(
                "Export validation failed: PDF exceeds size limit (%s bytes)",
                len(pdf_bytes),
            )
            raise HTTPException(
                status_code=500,
                detail="Generated report exceeds supported export size",
            )

        if not pdf_bytes.startswith(b"%PDF"):
            logger.error(
                "Export validation failed: Invalid PDF structure for certificate %s",
                cert_id,
            )
            raise HTTPException(
                status_code=500,
                detail="Generated report failed integrity validation",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF generation error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate report",
        )

    filename = f"FasalSaathi_BankReport_{cert_id}.pdf"

    logger.info(
        "[EXPORT_AUDIT] cert_id=%s size_bytes=%s farmer=%s crop=%s",
        cert_id,
        len(pdf_bytes),
        data.name,
        data.crop,
    )

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(pdf_bytes)),
        },
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

# Admin: role assignment with custom-claim sync
# ---------------------------------------------------------------------------

class AssignRoleRequest(BaseModel):
    target_uid: str = Field(..., min_length=1, max_length=128)
    role: str = Field(..., pattern=r"^(admin|expert|farmer|vendor|system|guest)$")


@router.post("/assign-role")
async def assign_role(request: Request, body: AssignRoleRequest):
    """
    Assign a role to a user and sync the Firebase custom claim.

    Admin only.  Syncs the Firebase Auth custom claim FIRST, then updates
    the Firestore users/{uid}.role field.  If the claim sync fails the
    Firestore update is skipped entirely, keeping both stores consistent.

    Refresh tokens are revoked so the target user must sign in again with the
    updated claim (no stale-role window).
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

        # Sync the Auth custom claim FIRST so that if it fails we never touch
        # Firestore, keeping both stores consistent.
        await sync_role_claim(body.target_uid, body.role)

        # Only update Firestore once the claim is confirmed.
        user_ref.update({"role": body.role})
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("assign_role: update failed uid=%s: %s", body.target_uid, exc)
        raise HTTPException(status_code=503, detail="Authorization service unavailable")

    return {
        "success": True,
        "target_uid": body.target_uid,
        "role": body.role,
        "message": "Role updated. The user must sign in again to apply the new credentials.",
    }


@router.post("/backfill-role-claims")
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
