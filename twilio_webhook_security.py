"""
Twilio WhatsApp webhook signature verification and inbound request handling.

All WhatsApp webhook routes must call :func:`handle_inbound_whatsapp_webhook` so
X-Twilio-Signature validation stays consistent across the application.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import urllib.parse
from typing import Tuple

from fastapi import HTTPException, Request


def verify_twilio_signature(request: Request, body: bytes) -> None:
    """Validate the X-Twilio-Signature header using HMAC-SHA1.

    Twilio signs every webhook request with:
        HMAC-SHA1(auth_token, url + sorted_params)

    Reference: https://www.twilio.com/docs/usage/webhooks/webhooks-security
    """
    auth_token = os.getenv("TWILIO_AUTH_TOKEN", "")
    if not auth_token:
        raise HTTPException(
            status_code=500,
            detail="Webhook signature verification is not configured",
        )

    twilio_signature = request.headers.get("X-Twilio-Signature", "")
    if not twilio_signature:
        raise HTTPException(status_code=403, detail="Missing Twilio signature")

    url = str(request.url)

    try:
        params = urllib.parse.parse_qsl(body.decode("utf-8"), keep_blank_values=True)
    except Exception:
        params = []
    sorted_params = sorted(params, key=lambda kv: kv[0])
    signing_string = url + "".join(k + v for k, v in sorted_params)

    expected = hmac.new(
        auth_token.encode("utf-8"),
        signing_string.encode("utf-8"),
        hashlib.sha1,
    ).digest()
    expected_b64 = base64.b64encode(expected).decode("utf-8")

    if not hmac.compare_digest(expected_b64, twilio_signature):
        raise HTTPException(status_code=403, detail="Invalid Twilio signature")


def validate_whatsapp_number(raw: str) -> str:
    """Strip the whatsapp: prefix and validate E.164-ish phone numbers."""
    number = raw.replace("whatsapp:", "").strip()
    if not re.fullmatch(r"\+?\d{7,15}", number):
        raise HTTPException(status_code=400, detail="Invalid sender number")
    return number


def parse_whatsapp_form(body: bytes) -> Tuple[str, str]:
    """Extract Body and From fields from a Twilio form-encoded webhook payload."""
    params = dict(urllib.parse.parse_qsl(body.decode("utf-8"), keep_blank_values=True))
    return params.get("Body", ""), params.get("From", "")


def enqueue_whatsapp_webhook_processing(message_body: str, sender_number: str) -> None:
    """Enqueue inbound WhatsApp webhook work on the Celery worker."""
    from celery_worker import process_whatsapp_webhook_task

    process_whatsapp_webhook_task.delay(message_body, sender_number)


async def handle_inbound_whatsapp_webhook(request: Request) -> dict:
    """
    Verify Twilio signature, validate sender, and enqueue async processing.

    Returns immediately so Twilio does not time out under burst traffic.
    """
    raw_body = await request.body()
    verify_twilio_signature(request, raw_body)

    message_body, from_raw = parse_whatsapp_form(raw_body)
    sender_number = validate_whatsapp_number(from_raw)

    enqueue_whatsapp_webhook_processing(message_body, sender_number)
    return {"status": "success"}
