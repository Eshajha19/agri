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

from dotenv import load_dotenv
from twilio.base.exceptions import TwilioRestException
from twilio.rest import Client

load_dotenv()

logger = logging.getLogger(__name__)

# ── Twilio configuration ──────────────────────────────────────────────────────
TWILIO_ACCOUNT_SID    = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN     = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER", "+14155238886")

# ── Shared client singleton ───────────────────────────────────────────────────
# Initialised once at module import time.  The Twilio SDK maintains an internal
# connection pool, so all send operations reuse the same pool of persistent
# HTTP connections — no per-message TCP/TLS overhead.
_twilio_client: Client | None = None

def _init_client() -> Client | None:
    """Create and return the Twilio Client, or None if credentials are missing."""
    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
        logger.warning(
            "Twilio credentials not configured — "
            "TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN must be set in .env"
        )
        return None
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        logger.info("Twilio client initialised (account: %s…)", TWILIO_ACCOUNT_SID[:8])
        return client
    except Exception as exc:
        logger.error("Failed to initialise Twilio client: %s", exc)
        return None

_twilio_client = _init_client()


def get_twilio_client() -> Client | None:
    """
    Return the shared Twilio client singleton.

    Returns None when credentials are missing or initialisation failed.
    Callers should treat None as a non-retryable configuration error.
    """
    return _twilio_client


def send_whatsapp_message(to_number: str, message_body: str) -> dict:
    """
    Send a WhatsApp message via the shared Twilio client.

    Parameters
    ----------
    to_number : str
        Recipient phone number with country code (e.g. +911234567890).
        The ``whatsapp:`` prefix is added automatically if absent.
    message_body : str
        Text content of the message.

    Returns
    -------
    dict
        Always returns a dict with at least:
          - ``success`` (bool)
          - ``status``  (str) — one of:
              "success" | "not_configured" | "rate_limited" |
              "client_error" | "server_error" | "error"
        On success also includes:
          - ``sid`` (str) — Twilio message SID
        On failure also includes:
          - ``error`` (str) — human-readable description
          - ``code``  (int, optional) — Twilio error code when available
    """
    client = get_twilio_client()
    if client is None:
        return {
            "success": False,
            "status": "not_configured",
            "error": "Twilio client is not initialised. Check credentials.",
        }

    # Normalise the recipient number to the whatsapp: URI scheme.
    if not to_number.startswith("whatsapp:"):
        to_number = f"whatsapp:{to_number}"

    try:
        message = client.messages.create(
            from_=f"whatsapp:{TWILIO_WHATSAPP_NUMBER}",
            body=message_body,
            to=to_number,
        )
        logger.debug("WhatsApp message sent to %s — SID: %s", to_number, message.sid)
        return {"success": True, "status": "success", "sid": message.sid}

    except TwilioRestException as exc:
        # Distinguish rate-limit errors from other Twilio API errors so
        # callers can implement back-off or skip retries appropriately.
        if exc.status == 429:
            logger.warning(
                "Twilio rate limit hit sending to %s (code=%s): %s",
                to_number, exc.code, exc.msg,
            )
            return {
                "success": False,
                "status": "rate_limited",
                "error": "Twilio rate limit exceeded. Retry after a delay.",
                "code": exc.code,
            }
        if 400 <= exc.status < 500:
            logger.error(
                "Twilio client error sending to %s (HTTP %s, code=%s): %s",
                to_number, exc.status, exc.code, exc.msg,
            )
            return {
                "success": False,
                "status": "client_error",
                "error": exc.msg,
                "code": exc.code,
            }
        # 5xx — Twilio server-side error
        logger.error(
            "Twilio server error sending to %s (HTTP %s, code=%s): %s",
            to_number, exc.status, exc.code, exc.msg,
        )
        return {
            "success": False,
            "status": "server_error",
            "error": exc.msg,
            "code": exc.code,
        }

    except Exception as exc:
        logger.exception("Unexpected error sending WhatsApp message to %s", to_number)
        return {"success": False, "status": "error", "error": str(exc)}


def format_alert_message(alert_type: str, content: str) -> str:
    """
    Format an alert payload into a WhatsApp-friendly message string.

    Parameters
    ----------
    alert_type : str
        One of "weather", "pest", "advisory", or any other string.
    content : str
        The body text of the alert.

    Returns
    -------
    str
        Formatted WhatsApp message with emoji header and footer.
    """
    header = "🌾 *Fasal Saathi Alert* 🌾\n\n"

    icons = {
        "weather":  ("⛈️",  "*Weather Warning*"),
        "pest":     ("🐛",  "*Pest Outbreak Alert*"),
        "advisory": ("📝",  "*Farming Advisory*"),
    }
    icon, title = icons.get(alert_type, ("📢", "*Notification*"))

    return (
        f"{header}{icon} {title}\n\n"
        f"{content}\n\n"
        "_Stay safe and stay informed with Fasal Saathi._"
    )
