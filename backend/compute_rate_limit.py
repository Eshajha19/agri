"""Shared rate limiting for expensive authenticated endpoints.

Backend selection
-----------------
When the REDIS_URL environment variable is set, rate-limit counters are
stored in Redis so all Gunicorn / Uvicorn workers share the same bucket.
Without Redis, each worker process has its own independent in-process dict
and a user can exceed their quota by distributing requests across workers.

Set REDIS_URL to any redis:// or rediss:// URL to enable the shared
backend, e.g.:

    REDIS_URL=redis://localhost:6379/0

The in-process fallback is retained for local development and single-worker
deployments where Redis is not available.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from math import ceil
from ipaddress import ip_address
from threading import Lock
from time import monotonic
from typing import Optional

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Redis client (optional)
# ---------------------------------------------------------------------------

_redis_client = None

_REDIS_URL = os.getenv("REDIS_URL", "")
if _REDIS_URL:
    try:
        import redis as _redis_lib  # type: ignore[import]
        _redis_client = _redis_lib.from_url(_REDIS_URL, decode_responses=True)
        # Validate connectivity at import time so misconfigured URLs fail early.
        _redis_client.ping()
        logger.info("compute_rate_limit: using Redis backend (%s)", _REDIS_URL.split("@")[-1])
    except Exception as _redis_err:  # noqa: BLE001
        logger.warning(
            "compute_rate_limit: Redis unavailable (%s); falling back to in-process store. "
            "Rate limits will not be shared across workers.",
            _redis_err,
        )
        _redis_client = None

# ---------------------------------------------------------------------------
# In-process store (used when Redis is not configured or unavailable)
# ---------------------------------------------------------------------------

_compute_rate_limit_store: dict[str, tuple[int, float, int]] = {}
_compute_rate_limit_lock = Lock()
_last_compute_rate_limit_prune = 0.0
_PRUNE_INTERVAL_SECONDS = 60


def reset_compute_rate_limit_state() -> None:
    """Clear limiter state for tests."""
    global _last_compute_rate_limit_prune

    with _compute_rate_limit_lock:
        _compute_rate_limit_store.clear()
        _last_compute_rate_limit_prune = 0.0


def _extract_client_ip(request: Request) -> Optional[str]:
    forwarded_for = request.headers.get("x-forwarded-for")

    if forwarded_for:
        for candidate in forwarded_for.split(","):
            candidate = candidate.strip()

            try:
                ip_address(candidate)
                return candidate.lower()
            except ValueError:
                continue

    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        try:
            ip_address(real_ip.strip())
            return real_ip.strip().lower()
        except ValueError:
            pass

    if request.client and request.client.host:
        try:
            ip_address(request.client.host.strip())
            return request.client.host.strip().lower()
        except ValueError:
            pass

    return None

def _request_actor_key(request: Request, uid: Optional[str]) -> str:
    if uid:
        return f"uid:{uid.strip().lower()}"

    client_ip = _extract_client_ip(request)

    if client_ip:
        return f"ip:{client_ip}"

    return "ip:unknown"


def _prune_expired_entries(now: float) -> None:
    global _last_compute_rate_limit_prune

    if now - _last_compute_rate_limit_prune < _PRUNE_INTERVAL_SECONDS:
        return

    expired_keys = [
        key
        for key, (_, window_start, window_seconds) in _compute_rate_limit_store.items()
        if now - window_start >= window_seconds
    ]
    for key in expired_keys:
        _compute_rate_limit_store.pop(key, None)

    _last_compute_rate_limit_prune = now


def build_compute_rate_limit_response(
    request: Request,
    *,
    scope: str,
    retry_after_seconds: int,
) -> JSONResponse:
    request_id = getattr(getattr(request, "state", None), "request_id", None)
    retry_after = max(1, int(retry_after_seconds))
    return JSONResponse(
        status_code=429,
        content={
            "success": False,
            "request_id": request_id,
            "error": {
                "code": "rate_limit_exceeded",
                "message": "Too many requests. Please retry later.",
                "detail": f"{scope} rate limit exceeded",
                "retry_after": retry_after,
            },
            "path": str(request.url.path),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        headers={"Retry-After": str(retry_after)},
    )


def _enforce_redis(
    key: str,
    *,
    scope: str,
    limit: int,
    window_seconds: int,
    request: Request,
) -> Optional[JSONResponse]:
    """Redis-backed fixed-window rate limit using an atomic Lua script.

    The Lua script increments the counter and conditionally sets the TTL in
    a single atomic operation.  This avoids the INCR-then-EXPIRE race where a
    process crash between the two pipeline commands left the key without a TTL,
    permanently locking the affected user or IP until manual key deletion.

    Lua guarantees:
    - If the key did not exist, INCR creates it with count=1 and EXPIRE sets
      the window TTL atomically.
    - If the key already exists (window still active), only INCR is executed;
      the existing TTL is preserved.
    """
    _LUA_INCR_WITH_EXPIRE = """
local current = redis.call("INCR", KEYS[1])
if current == 1 then
    redis.call("EXPIRE", KEYS[1], ARGV[1])
end
return current
"""
    try:
        count = _redis_client.eval(
            _LUA_INCR_WITH_EXPIRE, 1, key, window_seconds
        )
        if count > limit:
            ttl = _redis_client.ttl(key)
            retry_after = max(1, ttl if ttl > 0 else window_seconds)
            return build_compute_rate_limit_response(
                request, scope=scope, retry_after_seconds=retry_after
            )
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Redis rate-limit check failed (%s); allowing request.", exc)
        # Fail open: prefer a missed rate limit over a false 429 that blocks
        # legitimate users during a Redis outage.
        return None


def enforce_compute_rate_limit(
    request: Request,
    *,
    scope: str,
    uid: Optional[str],
    limit: int,
    window_seconds: int,
) -> Optional[JSONResponse]:
    """Return a structured 429 response when the caller's bucket is exhausted.

    Uses Redis when REDIS_URL is configured so the quota is shared across all
    Gunicorn workers.  Falls back to the in-process dict when Redis is not
    available, with the caveat that each worker tracks its own counters.
    """
    actor_key = _request_actor_key(request, uid)
    full_key = f"compute_rl:{scope}:{actor_key}"

    if _redis_client is not None:
        return _enforce_redis(
            full_key,
            scope=scope,
            limit=limit,
            window_seconds=window_seconds,
            request=request,
        )

    # In-process fallback.
    now = monotonic()
    with _compute_rate_limit_lock:
        _prune_expired_entries(now)

        current = _compute_rate_limit_store.get(full_key)
        if current is None:
            _compute_rate_limit_store[full_key] = (1, now, window_seconds)
            return None

        count, window_start, stored_window_seconds = current
        if now - window_start >= stored_window_seconds:
            _compute_rate_limit_store[full_key] = (1, now, window_seconds)
            return None

        if count >= limit:
            retry_after = ceil(stored_window_seconds - (now - window_start))
            return build_compute_rate_limit_response(
                request,
                scope=scope,
                retry_after_seconds=retry_after,
            )

        _compute_rate_limit_store[full_key] = (count + 1, window_start, stored_window_seconds)
        return None
