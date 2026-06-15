"""
webhook_validator.py

Provides schema validation and field sanitization for incoming
WhatsApp webhook payloads before they are processed by the Celery task.
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ── Strict size limits ────────────────────────────────────────────────────────
MAX_BODY_BYTES: int = 65_536          # 64 KB raw body ceiling
MAX_MESSAGE_TEXT_LEN: int = 4_096     # WhatsApp max message length
MAX_SENDER_NUMBER_LEN: int = 20       # E.164 numbers are ≤ 15 digits + country code
MAX_MEDIA_ID_LEN: int = 256
MAX_CAPTION_LEN: int = 1_024
MAX_BUTTON_PAYLOAD_LEN: int = 256

# Allowlist for sender phone numbers – E.164 format (digits only after optional +)
_PHONE_RE = re.compile(r"^\+?[1-9]\d{6,14}$")


class WebhookValidationError(ValueError):
    """Raised when a webhook payload fails validation."""


@dataclass
class NormalizedMessage:
    """Sanitized, validated representation of a single inbound WhatsApp message."""

    sender_number: str
    message_type: str          # text | image | audio | video | document | button | ...
    text: Optional[str]
    media_id: Optional[str]
    caption: Optional[str]
    button_payload: Optional[str]
    raw_message_id: str


# ── Public API ────────────────────────────────────────────────────────────────

def validate_and_parse(raw_body: str, sender_number: str) -> NormalizedMessage:
    """
    Entry point called by the Celery task.

    1. Enforces raw body size limit.
    2. Parses JSON safely.
    3. Validates top-level schema shape.
    4. Extracts and sanitizes the first inbound message.

    Raises ``WebhookValidationError`` on any problem so the caller
    can discard the payload without sending any outbound message.
    """
    _check_body_size(raw_body)
    sender_number = _sanitize_sender(sender_number)
    payload = _parse_json(raw_body)
    _validate_top_level_schema(payload)
    message = _extract_message(payload)
    return _build_normalized_message(message, sender_number)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _check_body_size(raw_body: str) -> None:
    size = len(raw_body.encode("utf-8"))
    if size > MAX_BODY_BYTES:
        raise WebhookValidationError(
            f"Payload too large: {size} bytes (max {MAX_BODY_BYTES})"
        )


def _parse_json(raw_body: str) -> dict:
    if not isinstance(raw_body, str):
        raise WebhookValidationError(
            f"raw_body must be a str, got {type(raw_body).__name__}"
        )
    try:
        data = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise WebhookValidationError(f"Invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise WebhookValidationError("Payload root must be a JSON object")
    return data


def _validate_top_level_schema(payload: dict) -> None:
    """
    Minimal structural check mirroring the WhatsApp Cloud API envelope:
    {
      "object": "whatsapp_business_account",
      "entry": [ { "changes": [ { "value": { "messages": [...] } } ] } ]
    }
    """
    if payload.get("object") != "whatsapp_business_account":
        raise WebhookValidationError(
            f"Unexpected object type: {payload.get('object')!r}"
        )
    entry = payload.get("entry")
    if not isinstance(entry, list) or not entry:
        raise WebhookValidationError("'entry' must be a non-empty list")
    changes = entry[0].get("changes")
    if not isinstance(changes, list) or not changes:
        raise WebhookValidationError("'entry[0].changes' must be a non-empty list")
    value = changes[0].get("value")
    if not isinstance(value, dict):
        raise WebhookValidationError("'entry[0].changes[0].value' must be an object")
    messages = value.get("messages")
    if not isinstance(messages, list) or not messages:
        raise WebhookValidationError("No messages found in payload")


def _extract_message(payload: dict) -> dict:
    try:
        msg = payload["entry"][0]["changes"][0]["value"]["messages"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise WebhookValidationError(f"Could not extract message: {exc}") from exc
    if not isinstance(msg, dict):
        raise WebhookValidationError("Message must be a JSON object")
    return msg


def _build_normalized_message(msg: dict, sender_number: str) -> NormalizedMessage:
    msg_type = _require_string(msg, "type", max_len=32)
    msg_id = _require_string(msg, "id", max_len=128)

    text: Optional[str] = None
    media_id: Optional[str] = None
    caption: Optional[str] = None
    button_payload: Optional[str] = None

    if msg_type == "text":
        text_obj = msg.get("text")
        if not isinstance(text_obj, dict):
            raise WebhookValidationError("'text' field must be an object for type=text")
        text = _sanitize_string(
            text_obj.get("body", ""), max_len=MAX_MESSAGE_TEXT_LEN, field="text.body"
        )

    elif msg_type in ("image", "audio", "video", "document", "sticker"):
        media_obj = msg.get(msg_type)
        if not isinstance(media_obj, dict):
            raise WebhookValidationError(
                f"'{msg_type}' field must be an object for type={msg_type}"
            )
        media_id = _sanitize_string(
            media_obj.get("id", ""), max_len=MAX_MEDIA_ID_LEN, field=f"{msg_type}.id"
        )
        raw_caption = media_obj.get("caption")
        if raw_caption is not None:
            caption = _sanitize_string(
                raw_caption, max_len=MAX_CAPTION_LEN, field="caption"
            )

    elif msg_type == "button":
        button_obj = msg.get("button")
        if not isinstance(button_obj, dict):
            raise WebhookValidationError("'button' field must be an object for type=button")
        button_payload = _sanitize_string(
            button_obj.get("payload", ""),
            max_len=MAX_BUTTON_PAYLOAD_LEN,
            field="button.payload",
        )
        text = _sanitize_string(
            button_obj.get("text", ""), max_len=MAX_MESSAGE_TEXT_LEN, field="button.text"
        )

    # Unknown types are allowed through but carry no text/media — callers
    # should handle them gracefully or log-and-discard.

    return NormalizedMessage(
        sender_number=sender_number,
        message_type=msg_type,
        text=text,
        media_id=media_id,
        caption=caption,
        button_payload=button_payload,
        raw_message_id=msg_id,
    )


# ── Field helpers ─────────────────────────────────────────────────────────────

def _sanitize_sender(sender_number: str) -> str:
    if not isinstance(sender_number, str):
        raise WebhookValidationError(
            f"sender_number must be a str, got {type(sender_number).__name__}"
        )
    stripped = sender_number.strip()
    if len(stripped) > MAX_SENDER_NUMBER_LEN:
        raise WebhookValidationError(
            f"sender_number too long: {len(stripped)} chars (max {MAX_SENDER_NUMBER_LEN})"
        )
    if not _PHONE_RE.match(stripped):
        raise WebhookValidationError(
            f"sender_number failed E.164 validation: {stripped!r}"
        )
    return stripped


def _require_string(obj: dict, key: str, max_len: int) -> str:
    val = obj.get(key)
    if not isinstance(val, str):
        raise WebhookValidationError(f"'{key}' must be a string, got {type(val).__name__}")
    if len(val) > max_len:
        raise WebhookValidationError(
            f"'{key}' too long: {len(val)} chars (max {max_len})"
        )
    return val.strip()


def _sanitize_string(value: str, max_len: int, field: str) -> str:
    if not isinstance(value, str):
        raise WebhookValidationError(f"'{field}' must be a string")
    if len(value) > max_len:
        raise WebhookValidationError(
            f"'{field}' too long: {len(value)} chars (max {max_len})"
        )
    # Strip leading/trailing whitespace; no further HTML-encoding needed here
    # because downstream callers are responsible for context-specific escaping.
    return value.strip()