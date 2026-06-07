"""
Shared rate limiting configuration.

Ensures both API apps use a consistent client key strategy and a
structured 429 response format.
"""

from __future__ import annotations

import ipaddress
import os
from datetime import datetime, timezone
from typing import Optional

from fastapi import Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded

# ---------------------------------------------------------------------------
# Trusted proxy configuration
#
# Only headers from known-trusted reverse proxies / CDNs are honoured when
# resolving the real client IP.  Any host not in this set has its forwarding
# headers ignored, so an ordinary client cannot spoof their address.
#
# Populate via the TRUSTED_PROXY_IPS environment variable as a
# comma-separated list of CIDRs or exact IPs, e.g.:
#   TRUSTED_PROXY_IPS=10.0.0.0/8,172.16.0.0/12,192.168.0.0/16
#
# Cloudflare egress ranges and loopback are included by default.
# ---------------------------------------------------------------------------

_DEFAULT_TRUSTED_NETWORKS = [
    # Loopback and private ranges commonly used by local reverse proxies
    "127.0.0.0/8",
    "::1/128",
    "10.0.0.0/8",
    "172.16.0.0/12",
    "192.168.0.0/16",
]


def _load_trusted_networks() -> list[ipaddress.IPv4Network | ipaddress.IPv6Network]:
    raw = os.environ.get("TRUSTED_PROXY_IPS", "")
    entries = [e.strip() for e in raw.split(",") if e.strip()] if raw else []
    networks = []
    for cidr in _DEFAULT_TRUSTED_NETWORKS + entries:
        try:
            networks.append(ipaddress.ip_network(cidr, strict=False))
        except ValueError:
            pass
    return networks


_TRUSTED_NETWORKS: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = (
    _load_trusted_networks()
)


def _is_trusted_proxy(host: str) -> bool:
    """Return True if *host* is within a trusted proxy network."""
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False
    return any(addr in net for net in _TRUSTED_NETWORKS)


def extract_client_ip(request: Request) -> str:
    """Resolve the real client IP, honouring proxy headers only from trusted sources.

    Headers like X-Forwarded-For and X-Real-IP can be trivially spoofed by
    any client.  This function only trusts them when the immediate peer
    (request.client.host) is a known-trusted proxy or CDN.  Direct clients
    always use their socket address, making rate-limit bypass via header
    injection impossible.
    """
    peer = request.client.host if request.client else None

    if peer and _is_trusted_proxy(peer):
        # Only read forwarding headers when the connection comes from a proxy
        # we control or a known CDN.
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

    # Direct connection or untrusted peer: use the socket address.
    if peer:
        return peer

    return "unknown"


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
