"""Alerts & Notifications Router"""
import asyncio
import base64
import hashlib
import hmac
import os
import re
import time
import urllib.parse
from datetime import datetime
from typing import Optional, Dict

from fastapi import APIRouter, Form, HTTPException, Query, Request
from pydantic import BaseModel, Field
from error_utils import safe_detail
from backend.core.logging_config import setup_logging
from backend.schemas.alerts import AlertTriggerRequest, AlertSummary

router = APIRouter()
logger = setup_logging(__name__)

# Idempotency store for processed webhook SIDs (MessageSid → timestamp).
# Prevents duplicate processing from Twilio retries or replay attacks.
# Entries older than WEBHOOK_IDEMPOTENCY_TTL seconds are pruned on access.
_processed_webhook_sids: dict = {}
WEBHOOK_IDEMPOTENCY_TTL = 86400  # 24 hours

notification_store = None
subscriber_store = None
generate_alerts_fn = None
send_whatsapp_fn = None
format_alert_fn = None
verify_role_fn = None
resolve_user_profile_fn = None
ALERT_HISTORY = {}


def init_alerts(ns, ss, ga_fn, sw_fn, fa_fn, vr_fn, rp_fn=None):
    global notification_store, subscriber_store, generate_alerts_fn
    global send_whatsapp_fn, format_alert_fn, verify_role_fn, resolve_user_profile_fn
    notification_store = ns
    subscriber_store = ss
    generate_alerts_fn = ga_fn
    send_whatsapp_fn = sw_fn
    format_alert_fn = fa_fn
    verify_role_fn = vr_fn
    resolve_user_profile_fn = rp_fn

def _calculate_alert_severity(
    frequency_score: int,
    failure_score: int,
    impact_score: int,
    history_score: int,
):
    severity_score = round(
        (
            frequency_score * 0.30
            + failure_score * 0.30
            + impact_score * 0.25
            + history_score * 0.15
        ),
        2,
    )

    if severity_score >= 80:
        severity = "critical"
    elif severity_score >= 65:
        severity = "high"
    elif severity_score >= 45:
        severity = "medium"
    else:
        severity = "low"

    return {
        "severity": severity,
        "severity_score": severity_score,
    }


def profile_regions(profile: Optional[Dict]) -> set:
    """Extract regions from user profile. Returns empty set if profile is None or missing regions."""
    if profile is None:
        return set()
    regions = profile.get("regions", []) if isinstance(profile, dict) else []
    return set(regions) if isinstance(regions, (list, set)) else set()


def notification_matches_regions(notification: Dict, user_regions: set) -> bool:
    """Check if notification's region matches user's allowed regions."""
    if not user_regions:
        return True
    notif_region = notification.get("region_id")
    if not notif_region:
        return True
    return notif_region in user_regions


def normalize_region_identifier(region_id: Optional[str]) -> Optional[str]:
    """Normalize region identifier: strip whitespace, lowercase, validate."""
    if not region_id:
        return None
    normalized = region_id.strip().lower()
    return normalized if normalized else None


def profile_can_broadcast_region(profile: Optional[Dict], region_id: str) -> bool:
    """Check if user profile has permission to broadcast to a specific region."""
    if profile is None:
        return False
    broadcast_regions = profile.get("broadcast_regions", []) if isinstance(profile, dict) else []
    return region_id in broadcast_regions


def region_matches(owned_region: str, target_region: str) -> bool:
    """Check if owned region matches target region (case-insensitive comparison)."""
    return owned_region.strip().lower() == target_region.strip().lower()


def _verify_twilio_signature(request: Request, raw_body: bytes) -> None:
    """Verify Twilio request signature using HMAC-SHA1.
    
    Raises HTTPException(403) if signature is invalid or missing.
    """
    twilio_token = os.environ.get("TWILIO_AUTH_TOKEN", "")
    if not twilio_token:
        logger.warning("TWILIO_AUTH_TOKEN not configured")
        raise HTTPException(status_code=500, detail="Twilio auth not configured")
    
    signature = request.headers.get("X-Twilio-Signature", "")
    if not signature:
        logger.warning("Missing X-Twilio-Signature header")
        raise HTTPException(status_code=403, detail="Invalid signature")
    
    request_url = str(request.url)
    expected_hash = base64.b64encode(
        hmac.new(
            twilio_token.encode(),
            (request_url + raw_body.decode()).encode(),
            hashlib.sha1
        ).digest()
    ).decode()
    
    if not hmac.compare_digest(signature, expected_hash):
        logger.warning("Invalid Twilio signature")
        raise HTTPException(status_code=403, detail="Invalid signature")


def _validate_whatsapp_number(whatsapp_number: str) -> str:
    """Extract and validate WhatsApp phone number from 'whatsapp:+1234567890' format.
    
    Returns the E.164 formatted phone number.
    Raises ValueError if format is invalid.
    """
    number = whatsapp_number
    if number.startswith("whatsapp:"):
        number = number[9:]
    
    if not _PHONE_E164_RE.match(number):
        raise ValueError(f"Invalid E.164 phone number: {number}")
    
    return number




@router.get("")
async def get_notifications(
    request: Request,
    crop: str = Query(None),
    irrigation_count: int = Query(None, ge=0),
    water_coverage: int = Query(None, ge=0, le=100),
    season: str = Query(None),
):
    if notification_store is None or generate_alerts_fn is None or verify_role_fn is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    token_data = await verify_role_fn(request)
    uid = token_data.get("sub") or token_data.get("uid")
    user_regions = profile_regions(resolve_user_profile_fn(uid)) if resolve_user_profile_fn is not None else set()
    dynamic_alerts = generate_alerts_fn(
        crop=crop,
        irrigation_count=irrigation_count,
        water_coverage=water_coverage,
        season=season,
    )
    stored = [
        notification
        for notification in notification_store.get_recent_for_user(uid)
        if notification_matches_regions(notification, user_regions)
    ]
    
    # Validate and serialize all alerts through AlertSummary schema
    all_alerts = stored + dynamic_alerts
    validated_alerts = []
    for alert in all_alerts:
        try:
            validated = AlertSummary.model_validate(alert)
            validated_alerts.append(validated.model_dump())
        except Exception as e:
            logger.warning(f"Alert validation failed for uid={uid}: {e}")
            # Skip invalid alerts rather than breaking the entire response
            continue
    
    return {"success": True, "data": validated_alerts}


# E.164 phone number: optional leading '+', then 7-15 digits with a
# non-zero leading digit. Rejects empty strings, letters, and numbers
# that are too short or too long to be valid phone numbers.
_PHONE_E164_RE = re.compile(r"^\+?[1-9]\d{6,14}$")


@router.post("/whatsapp/subscribe")
async def subscribe_whatsapp(
    request: Request,
    phone_number: str = Form(..., max_length=16),
    name: str = Form(..., min_length=1, max_length=100),
    region_id: Optional[str] = Form(None, max_length=100),
):
    if not all([subscriber_store, send_whatsapp_fn, verify_role_fn]):
        raise HTTPException(status_code=500, detail="Not initialized")

    # Validate phone_number format before passing it to Twilio.
    # Without this check an oversized or malformed value is forwarded
    # directly to the Twilio API, potentially causing unexpected billing
    # events or injection into Twilio's URL parameters.
    if not _PHONE_E164_RE.match(phone_number):
        raise HTTPException(
            status_code=422,
            detail="phone_number must be a valid E.164 number (e.g. +919876543210).",
        )

    # Strip control characters and leading/trailing whitespace from name
    # before embedding it into the WhatsApp welcome message.
    clean_name = re.sub(r"[\x00-\x1f\x7f]", "", name).strip()
    if not clean_name:
        raise HTTPException(status_code=422, detail="name must not be empty after sanitisation.")

    try:
        token_data = await verify_role_fn(request)
        uid = token_data.get("sub") or token_data.get("uid")
        subscriber = {
            "phone_number": phone_number,
            "name": clean_name,
            "subscribed_at": datetime.now().isoformat(),
            "region_id": normalize_region_identifier(region_id) or None,
        }
        subscriber_store.upsert(uid, subscriber)
        welcome_msg = f"Namaste {name}! 🙏\nWelcome to *Fasal Saathi WhatsApp Alerts*."
        await asyncio.to_thread(send_whatsapp_fn, phone_number, welcome_msg)
        return {"success": True, "message": "Successfully subscribed"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=safe_detail(e, 500))


@router.post("/whatsapp/trigger-alert")
async def trigger_whatsapp_alert(request: Request, data: AlertTriggerRequest):
    if not all([subscriber_store, send_whatsapp_fn, format_alert_fn, notification_store, verify_role_fn]):
        raise HTTPException(status_code=500, detail="Not initialized")
    try:
        token_data = await verify_role_fn(request)
        uid = token_data.get("sub") or token_data.get("uid")
        role = str(token_data.get("role", "")).strip().lower()
        region_id = normalize_region_identifier(data.region_id) if data.region_id else ""

        if region_id:
            if role not in {"admin", "expert"}:
                if resolve_user_profile_fn is None or not profile_can_broadcast_region(resolve_user_profile_fn(uid), region_id):
                    raise HTTPException(status_code=403, detail="Access denied: insufficient regional authority")
        elif role not in {"admin", "expert"}:
            raise HTTPException(status_code=403, detail="Access denied: insufficient permissions")

        subscribers = subscriber_store.get_all()
        results = []
        formatted_msg = format_alert_fn(data.alert_type, data.message)
        history = ALERT_HISTORY.get(data.alert_type, 0) + 1
        ALERT_HISTORY[data.alert_type] = history
        if region_id:
            subscribers = {
                user_id: info
                for user_id, info in subscribers.items()
                if any(region_matches(owned_region, region_id) for owned_region in profile_regions(info))
            }
        for user_id, info in subscribers.items():
            res = await asyncio.to_thread(send_whatsapp_fn, info["phone_number"], formatted_msg)
            results.append({
                "user_id": user_id,
                "success": res.get("success", False),
                "status": res.get("status", "error"),
            })
        delivered = sum(
            1 for r in results
            if r["success"]
        )

        failed_deliveries = len(results) - delivered

        frequency_score = min(
            history * 10,
            100,
        )

        failure_score = min(
            int(
                (
                    failed_deliveries
                    / max(len(results), 1)
                ) * 100
            ),
            100,
        )

        impact_score = min(
            delivered * 5,
            100,
        )

        history_score = min(
            history * 8,
            100,
        )

        severity_data = _calculate_alert_severity(
            frequency_score=frequency_score,
            failure_score=failure_score,
            impact_score=impact_score,
            history_score=history_score,
        )

        notification_store.append(
            alert_type=data.alert_type,
            message=data.message,
            region_id=region_id or None,
            severity=severity_data["severity"],
            severity_score=severity_data["severity_score"],
            occurrence_count=history,
            delivery_success_count=delivered,
            delivery_failure_count=failed_deliveries,
        )
        return {
            "success": True,
            "results": results,
            "delivered": delivered,
            "total": len(results),
            "alert_context": {
                "severity": severity_data["severity"],
                "severity_score": severity_data["severity_score"],
                "occurrence_count": history,
                "delivery_success_count": delivered,
                "delivery_failure_count": failed_deliveries,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=safe_detail(e, 500))



MAX_WEBHOOK_BODY_SIZE = 10 * 1024

@router.post("/whatsapp/webhook")
async def whatsapp_webhook(
    request: Request,
    Body: str = Form(...),
    From: str = Form(...),
    MessageSid: str = Form(...),
):
    """Receive inbound WhatsApp messages from Twilio.

    Security controls applied:
    1. Idempotency — duplicate MessageSid values are silently acknowledged
       so Twilio retries and replay attacks cannot trigger repeated effects.
    2. Twilio signature verification — every request is validated with
       HMAC-SHA1 against TWILIO_AUTH_TOKEN before any processing.
       Requests with a missing or invalid X-Twilio-Signature are
       rejected with HTTP 403.
    3. Sender number validation — the From field is checked against a
       basic E.164 pattern after stripping the 'whatsapp:' prefix so
       malformed values cannot propagate further.
    """
    if len(Body) > MAX_WEBHOOK_BODY_SIZE:
        raise HTTPException(status_code=413, detail="Request body too large")

    if send_whatsapp_fn is None:
        raise HTTPException(status_code=500, detail="Not initialized")

    # Read the raw body for signature verification (body is cached by Starlette
    # after Form parsing, so this is safe to call after FastAPI parses params).
    raw_body = await request.body()
    _verify_twilio_signature(request, raw_body)

    # Idempotency check: skip already-processed MessageSid.
    # Runs after signature verification so an attacker cannot probe
    # which SIDs have been seen without a valid signature.
    now = time.monotonic()
    cutoff = now - WEBHOOK_IDEMPOTENCY_TTL
    already_seen = False
    stale_keys = []
    for sid, ts in _processed_webhook_sids.items():
        if ts < cutoff:
            stale_keys.append(sid)
        elif sid == MessageSid:
            already_seen = True
    for sid in stale_keys:
        _processed_webhook_sids.pop(sid, None)
    if already_seen:
        return {"status": "duplicate", "message": "Already processed"}

    incoming_msg = Body.lower().strip()
    sender_number = _validate_whatsapp_number(From)

    responses = {
        "weather": "🌡️ *Weather Update*\n\n28°C, Clear skies.",
        "pest": "🐛 *Pest Assistant*\n\nPlease use the tool in-app.",
        "hi": "🙏 *Namaste!*\n\nI am your AI Assistant.",
        "hello": "🙏 *Namaste!*\n\nI am your AI Assistant.",
    }
    response = next(
        (v for k, v in responses.items() if k in incoming_msg),
        "Try 'Weather' or 'Pest' 🌱",
    )
    send_whatsapp_fn(sender_number, response)
    _processed_webhook_sids[MessageSid] = time.monotonic()
    return {"status": "success"}
