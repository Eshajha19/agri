"""Alerts & Notifications Router"""
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Form, HTTPException, Query, Request
from twilio_webhook_security import handle_inbound_whatsapp_webhook
from pydantic import BaseModel, Field

from geo_alerts import notification_matches_regions, profile_can_broadcast_region, profile_regions, region_matches, normalize_region_identifier

router = APIRouter()


class AlertTriggerRequest(BaseModel):
    alert_type: str = Field(..., pattern=r'^(weather|pest|advisory)$')
    message: str = Field(..., min_length=1, max_length=500)
    region_id: Optional[str] = Field(default=None, max_length=100)


notification_store = None
subscriber_store = None
generate_alerts_fn = None
send_whatsapp_fn = None
format_alert_fn = None
verify_role_fn = None
resolve_user_profile_fn = None


def _normalise_region_set(value: Any) -> set[str]:
    """Return a validated set of normalized region tokens."""
    if not value:
        return set()

    candidates = value if isinstance(value, (set, frozenset, list, tuple)) else [value]
    normalized: set[str] = set()
    for candidate in candidates:
        region = normalize_region_identifier(candidate)
        if region:
            normalized.add(region)
    return normalized


def _profile_regions_from_store(profile: Any) -> set[str]:
    """Safely extract normalized regions from store-backed profile data."""
    if not isinstance(profile, dict):
        return set()
    return _normalise_region_set(profile_regions(profile))


def _notification_matches_requested_regions(notification: Any, requested_regions: set[str]) -> bool:
    """Return True only for well-formed notifications matching requested regions."""
    if not isinstance(notification, dict):
        return False
    return notification_matches_regions(notification, requested_regions)


def _subscriber_matches_region(info: Any, region_id: str) -> bool:
    """Return True for subscribers whose stored profile matches the region."""
    if not isinstance(info, dict):
        return False

    return any(region_matches(owned_region, region_id) for owned_region in _profile_regions_from_store(info))


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


@router.get("/notifications")
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
    uid = token_data["uid"]
    user_regions = _profile_regions_from_store(resolve_user_profile_fn(uid)) if resolve_user_profile_fn is not None else set()
    dynamic_alerts = generate_alerts_fn(
        crop=crop,
        irrigation_count=irrigation_count,
        water_coverage=water_coverage,
        season=season,
    )
    stored = [
        notification
        for notification in notification_store.get_recent_for_user(uid)
        if _notification_matches_requested_regions(notification, user_regions)
    ]
    return {"success": True, "data": stored + dynamic_alerts}


@router.post("/whatsapp/subscribe")
async def subscribe_whatsapp(
    request: Request,
    phone_number: str = Form(...),
    name: str = Form(...),
    region_id: Optional[str] = Form(None),
):
    if not all([subscriber_store, send_whatsapp_fn, verify_role_fn]):
        raise HTTPException(status_code=500, detail="Not initialized")
    try:
        token_data = await verify_role_fn(request)
        uid = token_data.get("uid")
        subscriber = {
            "phone_number": phone_number,
            "name": name,
            "subscribed_at": datetime.now().isoformat(),
            "region_id": normalize_region_identifier(region_id) or None,
        }
        subscriber_store.upsert(uid, subscriber)
        welcome_msg = f"Namaste {name}! 🙏\nWelcome to *Fasal Saathi WhatsApp Alerts*."
        send_whatsapp_fn(phone_number, welcome_msg)
        return {"success": True, "message": "Successfully subscribed"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/whatsapp/trigger-alert")
async def trigger_whatsapp_alert(request: Request, data: AlertTriggerRequest):
    if not all([subscriber_store, send_whatsapp_fn, format_alert_fn, notification_store, verify_role_fn]):
        raise HTTPException(status_code=500, detail="Not initialized")
    try:
        token_data = await verify_role_fn(request)
        uid = token_data["uid"]
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
        if region_id:
            subscribers = {
                user_id: info
                for user_id, info in subscribers.items()
                if _subscriber_matches_region(info, region_id)
            }
        for user_id, info in subscribers.items():
            res = send_whatsapp_fn(info["phone_number"], formatted_msg)
            results.append({
                "user_id": user_id,
                "success": res.get("success", False),
                "status": res.get("status", "error"),
            })
        notification_store.append(alert_type=data.alert_type, message=data.message, region_id=region_id or None)
        delivered = sum(1 for r in results if r["success"])
        return {"success": True, "results": results, "delivered": delivered, "total": len(results)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/whatsapp/webhook")
async def whatsapp_webhook(request: Request):
    """Receive inbound WhatsApp messages from Twilio (delegates to shared handler)."""
    return await handle_inbound_whatsapp_webhook(request)
