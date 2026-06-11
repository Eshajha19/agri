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
import collections
from urllib.parse import urlparse

import re
import base64

BINARY_MAGIC = [b"\x1F\x8B", b"PK\x03\x04"]  # gzip, zip signatures

def looks_like_binary(payload: bytes) -> bool:
    # Check magic headers
    for magic in BINARY_MAGIC:
        if payload.startswith(magic):
            return True
    # Heuristic: if >30% of bytes are non‑printable, treat as binary
    non_printable = sum(1 for b in payload if b < 9 or (13 < b < 32))
    return non_printable / max(len(payload), 1) > 0.3

def looks_like_base64(text: str) -> bool:
    # Simple heuristic: long alphanumeric string with padding
    if len(text) > 100 and re.fullmatch(r"[A-Za-z0-9+/=]+", text):
        try:
            base64.b64decode(text, validate=True)
            return True
        except Exception:
            return False
    return False

logger = logging.getLogger(__name__)
SUSPICIOUS_PATTERNS = [
    r"<script.*?>",
    r"javascript:",
    r"onerror\s*=",
    r"onload\s*=",
    r"union\s+select",
    r"drop\s+table",
]

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
        return self._state.get(method, path)

    def set(self, method: str, path: str, value: dict):
        key = self._normalize_key(method, path)
        if key in self._state:
            self._state.move_to_end(key)
        self._state.set(method, path, value) = value
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
    # Maximum random jitter (seconds) added to the recovery timeout to
    # stagger half-open probes when many circuits recover at the same time.
    _JITTER_MAX = 5

    # Circuit states
    _CLOSED = "closed"
    _OPEN = "open"
    _HALF_OPEN = "half_open"

    def _contains_suspicious_content(self, content: str) -> bool:
        if not content:
            return False

        content = content.lower()

        return any(
            re.search(pattern, content)
            for pattern in SUSPICIOUS_PATTERNS
        )
    
    def __init__(self, app, log_cooldown: float = 5.0):
        super().__init__(app)
        self.error_counts = {}  # endpoint -> count
        self.error_timestamps = {}  # endpoint -> timestamp
        self._log_cooldown = log_cooldown  # seconds between same-category logs per endpoint
        self._log_last_emitted: Dict[str, float] = {}  # key -> last log time
        self._log_suppressed: Dict[str, int] = {}  # key -> count of suppressed logs
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Handle request with error recovery"""

        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Track timing
        start_time = time.time()
        endpoint = f"{request.method} {request.url.path}"

        if request.method in {"POST", "PUT", "PATCH"}:
            body = await request.body()

            if body:
                payload = body.decode("utf-8", errors="ignore")

                if self._contains_suspicious_content(payload):
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

        # ---- Pre-request: check if a probe (half-open) should be allowed ----
        state = self._circuit_state.get(endpoint)
        if state == self._OPEN:
            opened_at = self._circuit_open_since.get(endpoint, 0.0)
            # Apply a small random jitter so that multiple circuits whose
            # base timeout expired simultaneously don't all fire probes at
            # exactly the same instant (thundering-herd prevention).
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
            # --- NEW: inspect body before scanning ---
            body = await request.body()
            if isinstance(body, bytes) and looks_like_binary(body):
                # Skip regex scanning for binary/compressed payloads
                return await call_next(request)

            body_text = body.decode(errors="ignore")
            if looks_like_base64(body_text):
                # Skip regex scanning for base64-like payloads
                return await call_next(request)

            # Apply suspicious regex only for text/JSON
            suspicious_regex = re.compile(r"(DROP\s+TABLE|<script>|UNION\s+SELECT)", re.IGNORECASE)
            if suspicious_regex.search(body_text):
                logger.warning(f"[{request_id}] Suspicious payload detected at {endpoint}")
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "request_id": request_id,
                        "error": {
                            "message": "Suspicious payload detected",
                            "status_code": 400,
                            "category": "security"
                    }
                }
            )
            # Call the endpoint
            response = await call_next(request)

            # Log successful request
            duration = time.time() - start_time
            self._rate_log(
                f"info:{endpoint}", logging.INFO,
                "[%s] %s - Status: %s - Duration: %.2fs",
                request_id, endpoint, response.status_code, duration,
            )

            # Half-open → closed on success; otherwise retain failure
            # timestamps so the rolling window can track instability.
            if self._circuit_state.get(endpoint) == self._HALF_OPEN:
                logger.info("Circuit breaker closed for %s — probe succeeded", endpoint)

            # Remove all state for this endpoint when it transitions to CLOSED.
            # Keeping a CLOSED entry in _circuit_state causes the dict to grow
            # without bound — one entry per distinct endpoint ever seen.
            # Absent from the dict is semantically identical to CLOSED, so we
            # pop the key instead of writing "closed" to it.  The failure
            # timestamps and open-since entries are also pruned so the dicts
            # stay bounded to only currently-open or recently-failing endpoints.
            self._circuit_state.pop(endpoint, None)
            self._failure_timestamps.pop(endpoint, None)
            self._circuit_open_since.pop(endpoint, None)
            self._circuit_open_since.pop(f"{endpoint}.__jitter__", None)

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id

            return response

        except HTTPException as http_exc:
            # Handle HTTP exceptions
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
            # Handle validation errors
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
            # Handle timeout errors
            duration = time.time() - start_time
            
            self._rate_log(
                f"timeout:{endpoint}", logging.ERROR,
                "[%s] %s - Timeout Error - Duration: %.2fs",
                request_id, endpoint, duration,
            )

            # Record failure and check circuit breaker
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
            # Handle unexpected errors
            duration = time.time() - start_time
            error_id = str(uuid.uuid4())
            
            self._rate_log(
                f"unexpected:{endpoint}", logging.ERROR,
                "[%s] %s - Unexpected Error [%s]: %s - Duration: %.2fs\n%s",
                request_id, endpoint, error_id, exc, duration, traceback.format_exc(),
            )

            # Record failure and check circuit breaker
            self._record_failure(endpoint)

            # Check if circuit breaker should open
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

            # Return generic error
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
        """Record a failure timestamp and transition to OPEN if threshold met."""
        now = time.time()

        # Prune timestamps outside the rolling 60 s window
        ts_list = self._failure_timestamps.setdefault(endpoint, [])
        self._failure_timestamps[endpoint] = [t for t in ts_list if now - t < self._RESET_TIMEOUT]

        # Append this failure
        self._failure_timestamps[endpoint].append(now)

        # Half-open → open on the first probe failure (one strike, not five)
        if self._circuit_state.get(endpoint) == self._HALF_OPEN:
            self._circuit_state[endpoint] = self._OPEN
            self._circuit_open_since[endpoint] = now
            logger.warning(
                "Circuit breaker re-opened for %s — probe failed",
                endpoint,
            )
            return

        # Open the circuit if threshold reached within the rolling window
        if len(self._failure_timestamps[endpoint]) >= self._FAILURE_THRESHOLD:
            if self._circuit_state.get(endpoint) != self._OPEN:
                self._circuit_state[endpoint] = self._OPEN
                self._circuit_open_since[endpoint] = now
                # Store per-endpoint jitter so the recovery offset is stable
                # for this open period (not re-rolled on every request).
                jitter = random.uniform(0, self._JITTER_MAX)
                self._circuit_open_since[f"{endpoint}.__jitter__"] = jitter
                logger.warning(
                    "Circuit breaker opened for %s: %d failures in rolling %.0fs window "
                    "(recovery in ~%.0fs + %.1fs jitter)",
                    endpoint,
                    self._FAILURE_THRESHOLD,
                    self._RESET_TIMEOUT,
                    self._RESET_TIMEOUT,
                    jitter,
                )
        else:
            # Failure recorded but threshold not yet reached — prune the
            # failure_timestamps list if it is now empty so the dict does
            # not accumulate entries for endpoints that recovered naturally.
            if not self._failure_timestamps[endpoint]:
                self._failure_timestamps.pop(endpoint, None)

    def reset_circuit(self, endpoint: str) -> bool:
        """Manually reset the circuit breaker for *endpoint* to CLOSED.

        Intended for admin or health-check endpoints that need to force
        recovery without waiting for the full timeout.  Returns ``True``
        when the circuit was previously OPEN or HALF_OPEN, ``False`` when
        it was already CLOSED (or unknown).
        """
        previous = self._circuit_state.get(endpoint)
        was_open = previous in (self._OPEN, self._HALF_OPEN)
        self._circuit_state[endpoint] = self._CLOSED
        self._failure_timestamps.pop(endpoint, None)
        self._circuit_open_since.pop(endpoint, None)
        self._circuit_open_since.pop(f"{endpoint}.__jitter__", None)
        if was_open:
            logger.info(
                "Circuit breaker manually reset for %s (was %s)",
                endpoint,
                previous,
            )
        return was_open

    def _categorize_error(self, status_code: int) -> str:
        """Categorize HTTP error"""
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
        """Check if circuit breaker should open"""
        
        # Get error count and last error time
        error_count = self.error_counts.get(endpoint, 0)
        error_time = self.error_timestamps.get(endpoint, time.time())
        
        # Update timestamp
        self.error_timestamps[endpoint] = time.time()
        
        # Open circuit if more than 5 errors in 60 seconds
        time_since_error = time.time() - error_time
        
        if error_count >= 5 and time_since_error < 60:
            self._rate_log(
                f"circuit_breaker:{endpoint}", logging.WARNING,
                "Circuit breaker opened for %s: %d errors in %.0fs",
                endpoint, error_count, time_since_error,
            )
            return True
        
        # Reset if 60 seconds have passed
        if time_since_error >= 60:
            self.error_counts[endpoint] = 0
        
        return False
    
    def _rate_log(self, key: str, level: int, msg: str, *args):
        """Emit log at most once per ``_log_cooldown`` seconds per key.
        
        Suppressed calls increment a counter; periodic summary lines are
        emitted so that suppressed events remain measurable.
        """
        now = time.time()
        last = self._log_last_emitted.get(key, 0.0)
        if now - last >= self._log_cooldown:
            self._log_last_emitted[key] = now
            suppressed = self._log_suppressed.pop(key, 0)
            if suppressed:
                logger.log(level, "%s — (%d similar messages suppressed in the last %.0fs)", msg, suppressed, self._log_cooldown, *args)
            else:
                logger.log(level, msg, *args)
        else:
            self._log_suppressed[key] = self._log_suppressed.get(key, 0) + 1

    def get_error_stats(self) -> dict:
        """Get error statistics including per-endpoint circuit details."""
        now = time.time()
        pruned = {
            ep: [t for t in ts if now - t < self._RESET_TIMEOUT]
            for ep, ts in self._failure_timestamps.items()
        }

        # Build per-endpoint circuit detail (skip internal jitter keys)
        circuit_detail: dict[str, dict] = {}
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
                time_remaining = max(0.0, self._RESET_TIMEOUT + jitter - elapsed)
                detail["time_until_retry_seconds"] = round(time_remaining, 2)
            circuit_detail[ep] = detail

        return {
            "endpoints_with_errors": len(self.error_counts),
            "error_counts": self.error_counts,
            "timestamps": {
                k: v for k, v in self.error_timestamps.items()
            },
            "log_suppressed": dict(self._log_suppressed),
        }


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to add request IDs to all requests"""
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Add request ID to all responses"""
        
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        
        return response


async def error_logger_callback(error_context):
    """Callback for logging errors from async error handler"""
    logger.log(
        logging.WARNING if error_context.severity.value == "medium" else logging.ERROR,
        f"Error [{error_context.error_id}] in {error_context.source}: "
        f"{error_context.message}"
    )
