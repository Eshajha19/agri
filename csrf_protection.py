"""CSRF Protection for browser-submitted Form endpoints.

Provides:
- Origin / Referer header validation against a trusted-origin allowlist.
- Stateless CSRF token generation and validation (HMAC-SHA256, TTL-bound).
"""

import hmac
import hashlib
import secrets
import time
from typing import List, Optional
from urllib.parse import urlparse

from fastapi import HTTPException, Request

_SECRET = secrets.token_hex(32)
_TRUSTED_ORIGINS: List[str] = []


def configure(trusted_origins: List[str]) -> None:
    global _TRUSTED_ORIGINS
    _TRUSTED_ORIGINS = list(trusted_origins)


def _extract_origin(request: Request) -> Optional[str]:
    origin = request.headers.get("Origin")
    if origin:
        return origin.rstrip("/")
    referer = request.headers.get("Referer")
    if referer:
        parsed = urlparse(referer)
        return f"{parsed.scheme}://{parsed.netloc}"
    return None


def reject_cross_origin(request: Request) -> None:
    """Raise HTTP 403 when the Origin / Referer is present but not trusted.

    Server-to-server callers that omit both headers are allowed through
    (e.g. Twilio webhooks), while browser-originated cross-origin POSTs
    are rejected.
    """
    client_origin = _extract_origin(request)
    if client_origin is None:
        return
    if client_origin not in _TRUSTED_ORIGINS:
        raise HTTPException(
            status_code=403,
            detail="Cross-origin request rejected",
        )


def generate_token(uid: str) -> str:
    """Return a stateless HMAC-signed token tied to *uid*, valid 1 hour."""
    expiry = int(time.time()) + 3600
    payload = f"{uid}|{expiry}"
    sig = hmac.new(_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}|{sig}"


def validate_token(token: str, uid: str) -> bool:
    """Return True iff *token* is a valid, non-expired token for *uid*."""
    try:
        parts = token.split("|")
        if len(parts) != 3:
            return False
        payload = f"{parts[0]}|{parts[1]}"
        expected = hmac.new(
            _SECRET.encode(), payload.encode(), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(expected, parts[2]):
            return False
        if parts[0] != uid:
            return False
        if int(parts[1]) < time.time():
            return False
        return True
    except (ValueError, IndexError):
        return False
