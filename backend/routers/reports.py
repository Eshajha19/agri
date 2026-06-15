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
import secrets
from datetime import datetime

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

# Replay protection state
used_nonces = {}

SIGNATURE_TTL = 300  # 5 minutes
MAX_NONCES = 10000

cleanup_thread_started = False

nonce_metrics = {
    "active": 0,
    "expired_removed": 0,
    "evicted": 0,
}

replay_metrics = {
    "accepted": 0,
    "rejected": 0,
    "duplicates": 0,
    "expired": 0,
}

nonce_store_provider = None

def cleanup_expired_nonces():
    now = time.time()

    expired = []

    for nonce, meta in list(used_nonces.items()):
        if now - meta["timestamp"] > SIGNATURE_TTL:
            expired.append(nonce)

    for nonce in expired:
        del used_nonces[nonce]

    replay_metrics["expired"] += len(expired)

    if expired:
        logger.info(
            "reports.router.initialized "
            "auth_provider=%s "
            "key_provider=%s "
            "sanitizer_enabled=%s "
            "persistent_nonce_provider=%s",
            getattr(vr_fn, "__name__", type(vr_fn).__name__),
            getattr(gsk_fn, "__name__", type(gsk_fn).__name__),
            slf_fn is not None,
            nonce_provider is not None,
        )

    return len(expired)


@router.post("/submit-report")
def submit_report(payload: dict):

    cleanup_expired_nonces()

    nonce = payload.get("nonce")
    timestamp = payload.get("timestamp")
    signature = payload.get("signature")

    if not nonce:

        replay_metrics["rejected"] += 1

        raise HTTPException(
            status_code=400,
            detail="Missing nonce",
        )

    if nonce in used_nonces:

        replay_metrics["duplicates"] += 1
        replay_metrics["rejected"] += 1

        logger.warning(
            "[REPLAY_BLOCKED] nonce=%s first_seen=%s",
            nonce,
            used_nonces[nonce]["timestamp"],
        )

        raise HTTPException(
            status_code=400,
            detail="Invalid or replayed nonce",
        )

    now = int(time.time())

    if not timestamp or abs(now - int(timestamp)) > SIGNATURE_TTL:

        replay_metrics["rejected"] += 1

        raise HTTPException(
            status_code=400,
            detail="Signature expired",
        )

    if not verify_signature(payload, signature):

        replay_metrics["rejected"] += 1

        raise HTTPException(
            status_code=400,
            detail="Invalid signature",
        )

    entry = {
        "timestamp": time.time(),
        "request_id": str(uuid.uuid4()),
        "status": "accepted",
    }

    used_nonces[nonce] = entry

    if nonce_store_provider:
        try:
            nonce_store_provider.store_nonce(
                nonce,
                entry,
            )
        except Exception:
            logger.exception(
                "persistent_nonce_storage_failed"
            )

    replay_metrics["accepted"] += 1

    logger.info(
        "[REPLAY_ACCEPTED] nonce=%s request_id=%s",
        nonce,
        used_nonces[nonce]["request_id"],
    )

    return {
        "status": "accepted"
    }

def cleanup_expired_nonces():
    now = time.time()

    with nonce_lock:
        expired = [
            nonce
            for nonce, created_at in used_nonces.items()
            if now - created_at > SIGNATURE_TTL
        ]

        for nonce in expired:
            del used_nonces[nonce]

        nonce_metrics["expired_removed"] += len(expired)
        nonce_metrics["active"] = len(used_nonces)

        return len(expired)


def enforce_nonce_limit():
    with nonce_lock:

        if len(used_nonces) <= MAX_NONCES:
            return

        overflow = len(used_nonces) - MAX_NONCES

        oldest = sorted(
            used_nonces.items(),
            key=lambda x: x[1]
        )[:overflow]

        for nonce, _ in oldest:
            del used_nonces[nonce]

        nonce_metrics["evicted"] += overflow
        nonce_metrics["active"] = len(used_nonces)



def nonce_cleanup_worker():
    while True:
        try:
            removed = cleanup_expired_nonces()

            if removed:
                logger.info(
                    "nonce_cleanup removed=%s active=%s",
                    removed,
                    len(used_nonces),
                )

        except Exception:
            logger.exception("nonce cleanup failed")

        time.sleep(60)


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
    nonce_provider=None,
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
    global verify_role_fn
    global get_signing_keys_fn
    global sanitise_log_field_fn
    global nonce_store_provider

    if vr_fn is None:
        raise ValueError("verify_role_fn cannot be None")

    if gsk_fn is None:
        raise ValueError("get_signing_keys_fn cannot be None")

    verify_role_fn = vr_fn
    get_signing_keys_fn = gsk_fn
    sanitise_log_field_fn = slf_fn
    nonce_store_provider = nonce_provider
    

    if not cleanup_thread_started:
        thread = threading.Thread(
            target=nonce_cleanup_worker,
            daemon=True,
        )
        thread.start()
        cleanup_thread_started = True

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
    """Generate a collision-resistant, unpredictable certificate ID.

    The ID combines a short deterministic prefix (farmer+crop+date) with
    128 bits of CSPRNG entropy so that IDs are both traceable and
    impossible to guess or predict.
    """
    date_str = datetime.utcnow().strftime("%Y%m%d")
    prefix_raw = f"{data.name}|{data.crop}|{date_str}"
    prefix_digest = hashlib.sha256(prefix_raw.encode("utf-8")).hexdigest()[:6].upper()
    random_part = secrets.token_hex(8).upper()  # 64 bits → 16 hex chars
    return f"CERT-{prefix_digest}-{random_part}"


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

    try:
        signature = bytes.fromhex(signature_hex)
    except ValueError:
        return {"verified": False, "cert_id": cert_id, "error": "Invalid signature hex encoding"}

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

    try:
        der = public_key.public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        key_fingerprint = hashlib.sha256(der).hexdigest()[:16]
    except Exception:
        key_fingerprint = "unknown"

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

        try:
            cert_id = _make_cert_id(data)
            signature_hex = _sign_report(private_key, data, cert_id)
            pdf_bytes = _build_pdf(data, signature_hex, cert_id)

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
                "Content-Disposition": f'attachment; filename=\"{filename}\"',
                "Content-Length": str(len(pdf_bytes)),
            },
        )
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

@router.get("/admin/replay-metrics")
async def replay_metrics_endpoint(request: Request):

    if verify_role_fn is None:
        raise HTTPException(
            status_code=500,
            detail="Not initialized",
        )

    await verify_role_fn(
        request,
        required_roles=["admin"],
    )

    return {
        "success": True,
        "active_nonces": len(used_nonces),
        "metrics": replay_metrics,
    }

