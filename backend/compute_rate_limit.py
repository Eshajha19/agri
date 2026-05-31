"""Shared in-process rate limiting for expensive authenticated endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from math import ceil
from threading import Lock
from time import monotonic
from typing import Optional

from fastapi import Request
from fastapi.responses import JSONResponse


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


def _request_actor_key(request: Request, uid: Optional[str]) -> str:
    if uid:
        return f"uid:{uid.strip().lower()}"

    if request.client and request.client.host:
        return f"ip:{request.client.host.strip().lower()}"

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


def enforce_compute_rate_limit(
    request: Request,
    *,
    scope: str,
    uid: Optional[str],
    limit: int,
    window_seconds: int,
) -> Optional[JSONResponse]:
    """Return a structured 429 response when the bucket is exhausted."""
    now = monotonic()
    key = f"{scope}:{_request_actor_key(request, uid)}"

    with _compute_rate_limit_lock:
        _prune_expired_entries(now)

        current = _compute_rate_limit_store.get(key)
        if current is None:
            _compute_rate_limit_store[key] = (1, now, window_seconds)
            return None

        count, window_start, stored_window_seconds = current
        if now - window_start >= stored_window_seconds:
            _compute_rate_limit_store[key] = (1, now, window_seconds)
            return None

        if count >= limit:
            retry_after = ceil(stored_window_seconds - (now - window_start))
            return build_compute_rate_limit_response(
                request,
                scope=scope,
                retry_after_seconds=retry_after,
            )

        _compute_rate_limit_store[key] = (count + 1, window_start, stored_window_seconds)
        return None
