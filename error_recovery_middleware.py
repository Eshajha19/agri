"""
Error Recovery Middleware for FastAPI
Provides structured error handling and recovery for async operations
"""

import logging
import random
import time
import uuid
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.exceptions import HTTPException
import traceback
from typing import Dict
from middleware_utils import ensure_body_available
import collections
from urllib.parse import urlparse

import re
import base64
import hashlib
import threading

BINARY_MAGIC = [b"\x1F\x8B", b"PK\x03\x04"]  # gzip, zip signatures


def looks_like_binary(payload: bytes) -> bool:
    """Return True if the payload appears to be binary/compressed data."""
    for magic in BINARY_MAGIC:
        if payload.startswith(magic):
            return True
    # Heuristic: if >30% of bytes are non-printable, treat as binary
    non_printable = sum(1 for b in payload if b < 9 or (13 < b < 32))
    return non_printable / max(len(payload), 1) > 0.3


def looks_like_base64(text: str) -> bool:
    """
    Return True only if the text is long, matches the base64 alphabet,
    AND decodes to content that is NOT valid UTF-8 (i.e. actual binary data).

    Pure-text payloads that happen to be base64-encodable (UUID lists,
    alphanumeric crop codes, etc.) are NOT flagged — they decode to
    readable UTF-8, so there is nothing suspicious about them.
    """
    if len(text) < 100:
        return False
    if not re.fullmatch(r"[A-Za-z0-9+/=]+", text):
        return False
    try:
        decoded = base64.b64decode(text, validate=True)
    except Exception:
        return False
    # If it decodes to valid UTF-8, it is ordinary text — not suspicious.
    try:
        decoded.decode("utf-8")
        return False
    except UnicodeDecodeError:
        return True  # Binary content hidden in base64


logger = logging.getLogger(__name__)

SUSPICIOUS_PATTERNS = [
    r"<script[\s>]",
    r"javascript\s*:",
    r"onerror\s*=",
    r"onload\s*=",
]

SQL_PATTERNS = [
    re.compile(r"\b(DROP|TRUNCATE)\s+TABLE\s+\w+\s*;", re.IGNORECASE),
    re.compile(r"\bUNION\s+(ALL\s+)?SELECT\b.+\bFROM\b", re.IGNORECASE | re.DOTALL),
    re.compile(r"'\s*(OR|AND)\s+'?\d+'?\s*=\s*'?\d+", re.IGNORECASE),
    re.compile(r"(--|#|/\*)\s*(DROP|SELECT|INSERT|UPDATE|DELETE|UNION)", re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# DoS mitigation: scan-result cache + per-path token-bucket budget (#2366)
# ---------------------------------------------------------------------------
_SCAN_CACHE_TTL: float = 5.0
_SCAN_RPS_LIMIT: float = 10.0
_SCAN_BURST: int = 3


class _ScanCache:
    """Thread-safe cache keyed on (path, body_blake2b) with TTL expiry."""

    def __init__(self, ttl: float = _SCAN_CACHE_TTL, maxsize: int = 4096) -> None:
        self._ttl = ttl
        self._maxsize = maxsize
        self._store: Dict[tuple, tuple] = {}
        self._lock = threading.Lock()

    def _key(self, path: str, body: bytes) -> tuple:
        digest = hashlib.blake2b(body, digest_size=16).hexdigest()
        return (path, digest)

    def get(self, path: str, body: bytes):
        key = self._key(path, body)
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            result, expires_at = entry
            if time.monotonic() > expires_at:
                del self._store[key]
                return None
            return result

    def set(self, path: str, body: bytes, result: bool) -> None:
        key = self._key(path, body)
        with self._lock:
            if len(self._store) >= self._maxsize:
                now = time.monotonic()
                expired = [k for k, (_, exp) in self._store.items() if exp <= now]
                for k in expired:
                    del self._store[k]
                if len(self._store) >= self._maxsize:
                    for k in list(self._store)[:self._maxsize // 4]:
                        del self._store[k]
            self._store[key] = (result, time.monotonic() + self._ttl)


class _ScanBudget:
    """Per-path token-bucket: consume() returns True when a scan may proceed."""

    def __init__(self, rate: float = _SCAN_RPS_LIMIT, burst: int = _SCAN_BURST) -> None:
        self._rate = rate
        self._burst = burst
        self._buckets: Dict[str, tuple] = {}
        self._lock = threading.Lock()

    def consume(self, path: str) -> bool:
        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.get(path, (float(self._burst), now))
            tokens = min(float(self._burst), tokens + (now - last) * self._rate)
            if tokens >= 1.0:
                self._buckets[path] = (tokens - 1.0, now)
                return True
            self._buckets[path] = (tokens, now)
            return False


# Module-level singletons
_scan_cache = _ScanCache()
_scan_budget = _ScanBudget()

# ---------------------------------------------------------------------------


class CircuitBreakerState:
    def __init__(self, max_entries=1000):
        self._state = collections.OrderedDict()
        self._max_entries = max_entries

    def _normalize_key(self, method: str, path: str) -> str:
        # Strip query params
        parsed = urlparse(path)
        return f"{method.upper()} {parsed.path}"

    def get(self, method: str, path: str):
        key = self._normalize_key(method, path)
        return self._state.get(key)          # FIX: was self._state.get(method, path)

    def set(self, method: str, path: str, value: dict):
        key = self._normalize_key(method, path)
        if key in self._state:
            self._state.move_to_end(key)
        self._state[key] = value            # FIX: was self._state.set(method, path, value) = value
        # Prune oldest if over cap
        if len(self._state) > self._max_entries:
            self._state.popitem(last=False)


class ErrorRecoveryMiddleware(BaseHTTPMiddleware):
    """
    Middleware for handling errors with structured recovery

    Features:
    - Automatic error tracking
    - Structured error responses
    - Request/response logging
    - Circuit breaker integration
    - Rate-limited log emission per endpoint per window
    """

    _FAILURE_THRESHOLD = 5
    _RESET_TIMEOUT = 60  # seconds
    _JITTER_MAX = 5

    _CLOSED = "closed"
    _OPEN = "open"
    _HALF_OPEN = "half_open"

    def __init__(self, app, log_cooldown: float = 5.0):
        super().__init__(app)
        self.error_counts: Dict[str, int] = {}
        self.error_timestamps: Dict[str, float] = {}
        self._log_cooldown = log_cooldown
        self._log_last_emitted: Dict[str, float] = {}
        self._log_suppressed: Dict[str, int] = {}
        self._circuit_state: Dict[str, str] = {}
        self._failure_timestamps: Dict[str, list] = {}
        self._circuit_open_since: Dict[str, float] = {}

    def _contains_suspicious_content(self, content: str) -> bool:
        """
        Check XSS patterns and tighter SQL patterns.
        Called once per request after binary/base64 screening.
        """
        if not content:
            return False
        lower = content.lower()
        if any(re.search(pattern, lower) for pattern in SUSPICIOUS_PATTERNS):
            return True
        return any(p.search(content) for p in SQL_PATTERNS)

    async def dispatch(self, request: Request, call_next) -> Response:
        """Handle request with error recovery"""

        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        start_time = time.time()
        endpoint = f"{request.method} {request.url.path}"

        # ---- Read body ONCE and cache it on request.state ----
        body: bytes = b""
        if request.method in {"POST", "PUT", "PATCH"}:
            body = await ensure_body_available(request)

        # ---- Single, consolidated payload scan (with DoS mitigation) ----
        if body:
            if looks_like_binary(body):
                logger.debug("[%s] Binary payload detected — skipping text scan", request_id)
            else:
                payload_text = body.decode("utf-8", errors="ignore")

                if looks_like_base64(payload_text):
                    logger.debug("[%s] Base64-binary payload — skipping text scan", request_id)
                else:
                    # --- DoS fix: cache + budget before running regex (#2366) ---
                    path = request.url.path
                    cached = _scan_cache.get(path, body)
                    if cached is not None:
                        suspicious = cached
                    elif _scan_budget.consume(path):
                        suspicious = self._contains_suspicious_content(payload_text)
                        _scan_cache.set(path, body, suspicious)
                    else:
                        # Budget exhausted, no cache hit — assume safe under load
                        suspicious = False

                    if suspicious:
                        logger.warning(
                            "[%s] Suspicious payload detected on %s",
                            request_id,
                            endpoint,
                        )
                        return JSONResponse(
                            status_code=400,
                            content={
                                "success": False,
                                "request_id": request_id,
                                "error": {
                                    "message": "Suspicious payload detected",
                                    "status_code": 400,
                                    "category": "validation",
                                },
                            },
                        )

        # ---- Circuit breaker: pre-request check ----
        state = self._circuit_state.get(endpoint)
        if state == self._OPEN:
            opened_at = self._circuit_open_since.get(endpoint, 0.0)
            jitter = self._circuit_open_since.get(f"{endpoint}.__jitter__", 0.0)
            elapsed = time.time() - opened_at
            if elapsed >= self._RESET_TIMEOUT + jitter:
                self._circuit_state[endpoint] = self._HALF_OPEN
                logger.info("Circuit breaker half-open for %s — allowing probe", endpoint)
            else:
                retry_after = int(self._RESET_TIMEOUT + jitter - elapsed) + 1
                return JSONResponse(
                    status_code=503,
                    headers={"Retry-After": str(retry_after)},
                    content={
                        "success": False,
                        "request_id": request_id,
                        "error": {
                            "message": "Service temporarily unavailable",
                            "status_code": 503,
                            "category": "service_error",
                            "recoverable": True,
                            "retry_after_seconds": retry_after,
                        },
                    },
                )

        try:
            response = await call_next(request)

            duration = time.time() - start_time
            self._rate_log(
                f"info:{endpoint}", logging.INFO,
                "[%s] %s - Status: %s - Duration: %.2fs",
                request_id, endpoint, response.status_code, duration,
            )

            if self._circuit_state.get(endpoint) == self._HALF_OPEN:
                logger.info("Circuit breaker closed for %s — probe succeeded", endpoint)

            self._circuit_state.pop(endpoint, None)
            self._failure_timestamps.pop(endpoint, None)
            self._circuit_open_since.pop(endpoint, None)
            self._circuit_open_since.pop(f"{endpoint}.__jitter__", None)

            response.headers["X-Request-ID"] = request_id
            return response

        except HTTPException as http_exc:
            duration = time.time() - start_time
            self._rate_log(
                f"http_error:{endpoint}", logging.WARNING,
                "[%s] %s - HTTP Error: %s - Detail: %s - Duration: %.2fs",
                request_id, endpoint, http_exc.status_code, http_exc.detail, duration,
            )
            return JSONResponse(
                status_code=http_exc.status_code,
                content={
                    "success": False,
                    "request_id": request_id,
                    "error": {
                        "message": http_exc.detail,
                        "status_code": http_exc.status_code,
                        "category": self._categorize_error(http_exc.status_code),
                    },
                },
            )

        except ValueError as val_exc:
            duration = time.time() - start_time
            self._rate_log(
                f"validation_error:{endpoint}", logging.WARNING,
                "[%s] %s - Validation Error: %s - Duration: %.2fs",
                request_id, endpoint, val_exc, duration,
            )
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "request_id": request_id,
                    "error": {
                        "message": str(val_exc),
                        "status_code": 400,
                        "category": "validation",
                    },
                },
            )

        except TimeoutError as timeout_exc:
            duration = time.time() - start_time
            self._rate_log(
                f"timeout:{endpoint}", logging.ERROR,
                "[%s] %s - Timeout Error - Duration: %.2fs",
                request_id, endpoint, duration,
            )
            self._record_failure(endpoint)
            return JSONResponse(
                status_code=504,
                content={
                    "success": False,
                    "request_id": request_id,
                    "error": {
                        "message": "Request timeout - please try again",
                        "status_code": 504,
                        "category": "network",
                        "recoverable": True,
                    },
                },
            )

        except Exception as exc:
            duration = time.time() - start_time
            error_id = str(uuid.uuid4())
            self._rate_log(
                f"unexpected:{endpoint}", logging.ERROR,
                "[%s] %s - Unexpected Error [%s]: %s - Duration: %.2fs\n%s",
                request_id, endpoint, error_id, exc, duration, traceback.format_exc(),
            )
            self._record_failure(endpoint)

            if self._circuit_state.get(endpoint) == self._OPEN:
                return JSONResponse(
                    status_code=503,
                    content={
                        "success": False,
                        "request_id": request_id,
                        "error": {
                            "message": "Service temporarily unavailable",
                            "status_code": 503,
                            "category": "service_error",
                            "recoverable": True,
                            "error_id": error_id,
                        },
                    },
                )

            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "request_id": request_id,
                    "error": {
                        "message": "An unexpected error occurred",
                        "status_code": 500,
                        "category": "unknown",
                        "error_id": error_id,
                        "recoverable": True,
                    },
                },
            )

    # ------------------------------------------------------------------
    # Circuit breaker helpers
    # ------------------------------------------------------------------

    def _record_failure(self, endpoint: str) -> None:
        now = time.time()
        ts_list = self._failure_timestamps.setdefault(endpoint, [])
        self._failure_timestamps[endpoint] = [t for t in ts_list if now - t < self._RESET_TIMEOUT]
        self._failure_timestamps[endpoint].append(now)

        if self._circuit_state.get(endpoint) == self._HALF_OPEN:
            self._circuit_state[endpoint] = self._OPEN
            self._circuit_open_since[endpoint] = now
            logger.warning("Circuit breaker re-opened for %s — probe failed", endpoint)
            return

        if len(self._failure_timestamps[endpoint]) >= self._FAILURE_THRESHOLD:
            if self._circuit_state.get(endpoint) != self._OPEN:
                self._circuit_state[endpoint] = self._OPEN
                self._circuit_open_since[endpoint] = now
                jitter = random.uniform(0, self._JITTER_MAX)
                self._circuit_open_since[f"{endpoint}.__jitter__"] = jitter
                logger.warning(
                    "Circuit breaker opened for %s: %d failures in rolling %.0fs window "
                    "(recovery in ~%.0fs + %.1fs jitter)",
                    endpoint, self._FAILURE_THRESHOLD, self._RESET_TIMEOUT,
                    self._RESET_TIMEOUT, jitter,
                )
        else:
            if not self._failure_timestamps[endpoint]:
                self._failure_timestamps.pop(endpoint, None)

    def reset_circuit(self, endpoint: str) -> bool:
        previous = self._circuit_state.get(endpoint)
        was_open = previous in (self._OPEN, self._HALF_OPEN)
        self._circuit_state[endpoint] = self._CLOSED
        self._failure_timestamps.pop(endpoint, None)
        self._circuit_open_since.pop(endpoint, None)
        self._circuit_open_since.pop(f"{endpoint}.__jitter__", None)
        if was_open:
            logger.info("Circuit breaker manually reset for %s (was %s)", endpoint, previous)
        return was_open

    def _categorize_error(self, status_code: int) -> str:
        if 400 <= status_code < 500:
            if status_code == 401:
                return "authentication"
            elif status_code == 403:
                return "authorization"
            elif status_code == 404:
                return "not_found"
            else:
                return "client_error"
        elif status_code >= 500:
            return "server_error"
        return "unknown"

    def _check_circuit_breaker(self, endpoint: str) -> bool:
        error_count = self.error_counts.get(endpoint, 0)
        error_time = self.error_timestamps.get(endpoint, time.time())
        self.error_timestamps[endpoint] = time.time()
        time_since_error = time.time() - error_time
        if error_count >= 5 and time_since_error < 60:
            self._rate_log(
                f"circuit_breaker:{endpoint}", logging.WARNING,
                "Circuit breaker opened for %s: %d errors in %.0fs",
                endpoint, error_count, time_since_error,
            )
            return True
        if time_since_error >= 60:
            self.error_counts[endpoint] = 0
        return False

    def _rate_log(self, key: str, level: int, msg: str, *args):
        now = time.time()
        last = self._log_last_emitted.get(key, 0.0)
        if now - last >= self._log_cooldown:
            self._log_last_emitted[key] = now
            suppressed = self._log_suppressed.pop(key, 0)
            if suppressed:
                logger.log(level, "%s — (%d similar messages suppressed in the last %.0fs)",
                           msg, suppressed, self._log_cooldown, *args)
            else:
                logger.log(level, msg, *args)
        else:
            self._log_suppressed[key] = self._log_suppressed.get(key, 0) + 1

    def get_error_stats(self) -> dict:
        now = time.time()
        pruned = {
            ep: [t for t in ts if now - t < self._RESET_TIMEOUT]
            for ep, ts in self._failure_timestamps.items()
        }
        circuit_detail: dict = {}
        for ep, state in self._circuit_state.items():
            if ep.endswith(".__jitter__"):
                continue
            detail: dict = {"state": state}
            if state in (self._OPEN, self._HALF_OPEN):
                opened_at = self._circuit_open_since.get(ep, 0.0)
                jitter = self._circuit_open_since.get(f"{ep}.__jitter__", 0.0)
                elapsed = now - opened_at
                detail["open_since"] = opened_at
                detail["elapsed_seconds"] = round(elapsed, 2)
                detail["time_until_retry_seconds"] = round(
                    max(0.0, self._RESET_TIMEOUT + jitter - elapsed), 2
                )
            circuit_detail[ep] = detail

        return {
            "endpoints_with_errors": len(self.error_counts),
            "error_counts": self.error_counts,
            "timestamps": dict(self.error_timestamps),
            "log_suppressed": dict(self._log_suppressed),
        }


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


async def error_logger_callback(error_context):
    logger.log(
        logging.WARNING if error_context.severity.value == "medium" else logging.ERROR,
        f"Error [{error_context.error_id}] in {error_context.source}: "
        f"{error_context.message}"
    )