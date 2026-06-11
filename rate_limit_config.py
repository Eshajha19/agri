"""
Shared rate limiting configuration.

Ensures both API apps use a consistent client key strategy and a
structured 429 response format.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Optional

from fastapi import Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded


def _client_fingerprint(request: Request) -> str:
    """Derive a stable, client-specific fallback key when no IP is
    available by fingerprinting the request headers that typically
    vary per client."""
    raw = "|".join([
        request.headers.get("user-agent") or "",
        request.headers.get("accept-language") or "",
        request.headers.get("accept-encoding") or "",
        request.headers.get("sec-ch-ua") or "",
        str(getattr(getattr(request, "state", None), "request_id", "")),
    ])
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def extract_client_ip(request: Request) -> str:
    """Resolve the best-effort client IP behind proxies/CDNs."""
    # Prefer CDN and reverse proxy headers when present.
    cf_ip = request.headers.get("cf-connecting-ip")
    if cf_ip:
        return cf_ip.strip()

    xff = request.headers.get("x-forwarded-for")
    if xff:
        # First IP in the chain is the originating client.
        first = xff.split(",")[0].strip()
        if first:
            return first

    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()

    if request.client and request.client.host:
        return request.client.host

    return _client_fingerprint(request)


def build_limiter(default_limits: Optional[list[str]] = None) -> Limiter:
    """Create a limiter with shared defaults and consistent headers."""
    return Limiter(
        key_func=extract_client_ip,
        default_limits=default_limits or ["120/minute"],
        headers_enabled=False,
    )


async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    """Return a consistent JSON payload for 429 responses."""
    retry_after = None
    if hasattr(exc, "headers") and isinstance(exc.headers, dict):
        retry_after = exc.headers.get("Retry-After")

    request_id = getattr(getattr(request, "state", None), "request_id", None)

    return JSONResponse(
        status_code=429,
        content={
            "success": False,
            "request_id": request_id,
            "error": {
                "code": "rate_limit_exceeded",
                "message": "Too many requests. Please retry later.",
                "detail": str(getattr(exc, "detail", "Rate limit exceeded")),
                "retry_after": retry_after,
            },
            "path": str(request.url.path),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        headers={"Retry-After": str(retry_after)} if retry_after else None,
    )
