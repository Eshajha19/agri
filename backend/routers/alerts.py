"""Alerts & Notifications Router"""
import asyncio
import base64
import hashlib
import hmac
import os
import re
import urllib.parse
from datetime import datetime

from fastapi import APIRouter, Form, HTTPException, Query, Request
from pydantic import BaseModel, Field

router = APIRouter()


class AlertTriggerRequest(BaseModel):
    alert_type: str = Field(..., pattern=r'^(weather|pest|advisory)$')
    message: str = Field(..., min_length=1, max_length=500)


notification_store = None
subscriber_store = None
generate_alerts_fn = None
send_whatsapp_fn = None
format_alert_fn = None
verify_role_fn = None


def init_alerts(ns, ss, ga_fn, sw_fn, fa_fn, vr_fn):
    global notification_store, subscriber_store, generate_alerts_fn
    global send_whatsapp_fn, format_alert_fn, verify_role_fn
    notification_store = ns
    subscriber_store = ss
    generate_alerts_fn = ga_fn
    send_whatsapp_fn = sw_fn
    format_alert_fn = fa_fn
    verify_role_fn = vr_fn


def _verify_twilio_signature(request: Request, body: bytes) -> None:
    """Validate the X-Twilio-Signature header using HMAC-SHA1.

    Twilio signs every webhook request with:
        HMAC-SHA1(auth_token, url + sorted_params)

    If the signature is absent or does not match we raise HTTP 403 so
    forged requests are rejected before any processing occurs.

    Reference: https://www.twilio.com/docs/usage/webhooks/webhooks-security
    """
    auth_token = os.getenv("TWILIO_AUTH_TOKEN", "")
    if not auth_token:
        # If the token is not configured we cannot verify — fail closed.
        raise HTTPException(
            status_code=500,
            detail="Webhook signature verification is not configured",
        )

    twilio_signature = request.headers.get("X-Twilio-Signature", "")
    if not twilio_signature:
        raise HTTPException(status_code=403, detail="Missing Twilio signature")

    # Reconstruct the full URL exactly as Twilio sees it.
    url = str(request.url)

    # Parse the form-encoded body and sort parameters alphabetically.
    try:
        params = urllib.parse.parse_qsl(body.decode("utf-8"), keep_blank_values=True)
    except Exception:
        params = []
    sorted_params = sorted(params, key=lambda kv: kv[0])

    # Build the string Twilio signs: URL + key1value1key2value2...
    signing_string = url + "".join(k + v for k, v in sorted_params)

    expected = hmac.new(
        auth_token.encode("utf-8"),
        signing_string.encode("utf-8"),
        hashlib.sha1,
    ).digest()

    expected_b64 = base64.b64encode(expected).decode("utf-8")

    if not hmac.compare_digest(expected_b64, twilio_signature):
        raise HTTPException(status_code=403, detail="Invalid Twilio signature")


def _validate_whatsapp_number(raw: str) -> str:
    """Strip the 'whatsapp:' prefix and do basic E.164 sanity check.

    Raises HTTP 400 for values that don't look like a phone number so
    the raw From field cannot be used to inject arbitrary strings.
    """
    number = raw.replace("whatsapp:", "").strip()
    # E.164: optional leading +, then 7–15 digits
    if not re.fullmatch(r"\+?\d{7,15}", number):
        raise HTTPException(status_code=400, detail="Invalid sender number")
    return number


@router.get("/notifications")
async def get_notifications(
    request: Request,
    crop: str = Query(None),
    irrigation_count: int = Query(None, ge=0),
    water_coverage: int = Query(None, ge=0, le=100),
    season: str = Query(None),
):
    if notification_store is None or generate_alerts_fn is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    dynamic_alerts = generate_alerts_fn(
        crop=crop,
        irrigation_count=irrigation_count,
        water_coverage=water_coverage,
        season=season,
    )
    return {"success": True, "data": notification_store.get_recent() + dynamic_alerts}


@router.post("/whatsapp/subscribe")
async def subscribe_whatsapp(
    request: Request,
    phone_number: str = Form(...),
    name: str = Form(...),
):
    if not all([subscriber_store, send_whatsapp_fn, verify_role_fn]):
        raise HTTPException(status_code=500, detail="Not initialized")
    try:
        token_data = await verify_role_fn(request)
        uid = token_data["uid"]
        subscriber = {
            "phone_number": phone_number,
            "name": name,
            "subscribed_at": datetime.now().isoformat(),
        }
        subscriber_store.upsert(uid, subscriber)
        welcome_msg = f"Namaste {name}! 🙏\nWelcome to *Fasal Saathi WhatsApp Alerts*."
        await asyncio.to_thread(send_whatsapp_fn, phone_number, welcome_msg)
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
        await verify_role_fn(request, required_roles=["admin", "expert"])
        subscribers = subscriber_store.get_all()
        results = []
        formatted_msg = format_alert_fn(data.alert_type, data.message)
        for user_id, info in subscribers.items():
            res = send_whatsapp_fn(info["phone_number"], formatted_msg)
            results.append({
                "user_id": user_id,
                "success": res.get("success", False),
                "status": res.get("status", "error"),
            })
        notification_store.append(alert_type=data.alert_type, message=data.message)
        delivered = sum(1 for r in results if r["success"])
        return {"success": True, "results": results, "delivered": delivered, "total": len(results)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/whatsapp/webhook")
async def whatsapp_webhook(
    request: Request,
    Body: str = Form(...),
    From: str = Form(...),
):
    """Receive inbound WhatsApp messages from Twilio.

    Security controls applied:
    1. Twilio signature verification — every request is validated with
       HMAC-SHA1 against TWILIO_AUTH_TOKEN before any processing.
       Requests with a missing or invalid X-Twilio-Signature are
       rejected with HTTP 403.
    2. Sender number validation — the From field is checked against a
       basic E.164 pattern after stripping the 'whatsapp:' prefix so
       malformed values cannot propagate further.
    """
    if send_whatsapp_fn is None:
        raise HTTPException(status_code=500, detail="Not initialized")

    # Read the raw body for signature verification before FastAPI
    # consumes it via Form parameters.
    raw_body = await request.body()
    _verify_twilio_signature(request, raw_body)

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
    return {"status": "success"}
