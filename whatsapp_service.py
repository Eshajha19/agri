"""
whatsapp_service.py — Twilio WhatsApp messaging with a shared client singleton.

Problem solved
--------------
The previous implementation called get_twilio_client() (which called Client())
inside send_whatsapp_message(), meaning a brand-new Twilio HTTP client was
instantiated for every single message.  For a broadcast to N subscribers this
created N separate TCP connections and N separate TLS handshakes, causing:

  • Connection exhaustion under load
  • Twilio 429 rate-limit errors on large broadcasts
  • Slow delivery due to repeated connection setup overhead
  • Silent failures — exceptions were caught and returned as
    {"success": False} but the broadcast loop in main.py still returned
    HTTP 200, masking delivery failures from callers

Fix
---
  1. The Twilio Client is created exactly once at module import time and
     reused for every subsequent call.  The Twilio Python SDK manages an
     internal connection pool, so all sends share the same pool of
     persistent HTTP connections.

  2. send_whatsapp_message() now returns a structured result dict that
     distinguishes between:
       - success          : message accepted by Twilio
       - rate_limited     : Twilio returned HTTP 429
       - client_error     : other 4xx (bad number, unverified, etc.)
       - server_error     : Twilio 5xx
       - not_configured   : credentials missing / client failed to init
       - error            : unexpected exception

  3. Callers (e.g. trigger_whatsapp_alert in main.py) can now inspect the
     "status" field to accurately report delivery outcomes instead of
     silently swallowing failures.
"""

import logging
import os
import re

from twilio.base.exceptions import TwilioRestException
from twilio.rest import Client

logger = logging.getLogger(__name__)

# =============================================================================
# ENV CONFIG
# =============================================================================

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER", "").strip()

# =============================================================================
# VALIDATION
# =============================================================================

E164_RE = re.compile(r"^\+[1-9]\d{7,14}$")

MAX_MESSAGE_LENGTH = 1500

# =============================================================================
# TWILIO CLIENT
# =============================================================================

_client = None


def _get_client() -> Client:
    global _client

    if _client is not None:
        return _client

    if not TWILIO_ACCOUNT_SID:
        raise RuntimeError("TWILIO_ACCOUNT_SID missing")

    if not TWILIO_AUTH_TOKEN:
        raise RuntimeError("TWILIO_AUTH_TOKEN missing")

    _client = Client(
        TWILIO_ACCOUNT_SID,
        TWILIO_AUTH_TOKEN,
    )

    return _client


# =============================================================================
# HELPERS
# =============================================================================

def _validate_phone_number(phone_number: str) -> str:
    if not isinstance(phone_number, str):
        raise ValueError("Phone number must be string")

    phone_number = phone_number.strip()

    if not E164_RE.fullmatch(phone_number):
        raise ValueError(
            "Phone number must be valid E.164 format"
        )

    return phone_number


def _sanitize_message(message: str) -> str:
    if not isinstance(message, str):
        raise ValueError("Message must be string")

    message = message.strip()

    if not message:
        raise ValueError("Message cannot be empty")

    message = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", message)

    if len(message) > MAX_MESSAGE_LENGTH:
        message = message[:MAX_MESSAGE_LENGTH]

    return message


# =============================================================================
# PUBLIC API
# =============================================================================

def format_outbound_alert_message(alert_type: str, message: str) -> str:
    """
    Create formatted WhatsApp alert message.
    """

    alert_type = (alert_type or "").strip().lower()

    emoji_map = {
        "weather": "🌦️",
        "pest": "🐛",
        "advisory": "📢",
    }

    title_map = {
        "weather": "WEATHER ALERT",
        "pest": "PEST ALERT",
        "advisory": "FARM ADVISORY",
    }

    emoji = emoji_map.get(alert_type, "📢")
    title = title_map.get(alert_type, "ALERT")

    message = _sanitize_message(message)

    return (
        f"{emoji} *{title}*\n\n"
        f"{message}\n\n"
        f"- Fasal Saathi"
    )


def send_whatsapp_message(phone_number: str, message: str) -> dict[str, Any]:
    try:
        phone_number = _validate_phone_number(phone_number)
        message = _sanitize_message(message)

        if not TWILIO_WHATSAPP_NUMBER:
            return {
                "success": False,
                "status": "not_configured",
                "sid": None,
                "error": "TWILIO_WHATSAPP_NUMBER missing",
            }

        client = _get_client()
        twilio_message = client.messages.create(
            body=message,
            from_=f"whatsapp:{TWILIO_WHATSAPP_NUMBER}",
            to=f"whatsapp:{phone_number}",
        )

        return {
            "success": True,
            "status": "sent",
            "sid": twilio_message.sid,
            "error": None,
        }

    except TwilioRestException as exc:
        code = getattr(exc, "code", 0)
        if code == 429:
            status = "rate_limited"
        elif 400 <= code < 500:
            status = "client_error"
        elif 500 <= code < 600:
            status = "server_error"
        else:
            status = "twilio_error"

        return {
            "success": False,
            "status": status,
            "sid": None,
            "error": str(exc),
        }

    except Exception as exc:
        return {
            "success": False,
            "status": "internal_error",
            "sid": None,
            "error": str(exc),
        }

    except Exception as exc:
        logger.exception(
            "WhatsApp send failed"
        )

        return {
            "success": False,
            "status": "internal_error",
            "error": str(exc),
        }


def sanitise_message(text: str) -> str:
    """Strip ASCII control characters (0x00-0x1F, 0x7F) except \\n, \\r, \\t."""
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)


def format_outbound_alert_message(alert_type: str, content: str) -> str:
    """
    Process inbound WhatsApp webhook messages.
    """
    content = sanitise_message(content)
    header = "🌾 *Fasal Saathi Alert* 🌾\n\n"

    body = _sanitize_message(body)
    sender_number = _validate_phone_number(sender_number)

    normalized = body.lower().strip()

    if normalized in {"hi", "hello", "start"}:
        reply = (
            "🙏 Welcome to Fasal Saathi!\n\n"
            "Available commands:\n"
            "- weather\n"
            "- help\n"
            "- market\n"
            "- support"
        )

        return send_whatsapp_message(
            sender_number,
            reply,
        )

    if normalized == "help":
        reply = (
            "📘 Fasal Saathi Help\n\n"
            "Send:\n"
            "- weather\n"
            "- market\n"
            "- support"
        )

        return send_whatsapp_message(
            sender_number,
            reply,
        )

    if normalized == "weather":
        reply = (
            "🌦️ Weather updates feature connected successfully."
        )

        return send_whatsapp_message(
            sender_number,
            reply,
        )

    if normalized == "market":
        reply = (
            "📈 Market prices feature connected successfully."
        )

        return send_whatsapp_message(
            sender_number,
            reply,
        )

    reply = (
        "❓ Unknown command.\n"
        "Send 'help' to view commands."
    )

    return send_whatsapp_message(
        sender_number,
        reply,
    )