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
from async_error_handler import CircuitBreakerAsync

logger = logging.getLogger(__name__)

# Patterns tightened to require executable SQL/XSS syntax rather than
# matching the keywords anywhere in prose.
#
# Removed from SUSPICIOUS_PATTERNS (handled in the try-block scan below):
#   drop\s+table, union\s+select  — prone to prose false positives.
# Kept here only the unambiguous XSS signals.
SUSPICIOUS_PATTERNS = [
    r"<script[\s>]",          # <script> or <script ...> — requires the tag
    r"javascript\s*:",         # javascript: URI scheme
    r"onerror\s*=",
    r"onload\s*=",
]

# Tighter SQL patterns used in the single consolidated scan.
# Each requires surrounding syntax that distinguishes real SQL from prose:
#   - DROP TABLE / TRUNCATE TABLE must be followed by a semicolon
#   - UNION SELECT must be followed by a FROM clause
#   - Classic OR tautology: ' OR '1'='1
SQL_PATTERNS = [
    re.compile(r"\b(DROP|TRUNCATE)\s+TABLE\s+\w+\s*;", re.IGNORECASE),
    re.compile(r"\bUNION\s+(ALL\s+)?SELECT\b.+\bFROM\b", re.IGNORECASE | re.DOTALL),
    re.compile(r"'\s*(OR|AND)\s+'?\d+'?\s*=\s*'?\d+", re.IGNORECASE),
    re.compile(r"(--|#|/\*)\s*(DROP|SELECT|INSERT|UPDATE|DELETE|UNION)", re.IGNORECASE),
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
    
    def __init__(self, app, circuit_breaker=None):
        super().__init__(app)
        self.circuit_breaker = circuit_breaker or CircuitBreakerAsync(
            failure_threshold=5,
            recovery_timeout=60,
            name="middleware"
        )
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Handle request with error recovery"""

        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        start_time = time.time()
        endpoint = f"{request.method} {request.url.path}"

        # ---- Read body ONCE and cache it on request.state ----
        # BaseHTTPMiddleware does not automatically replay the body stream,
        # so we read it here and store it so both the security scan and the
        # downstream handler can access the same bytes without a second read.
        body: bytes = b""
        if request.method in {"POST", "PUT", "PATCH"}:
            body = await ensure_body_available(request)

        # ---- Single, consolidated payload scan ----
        if body:
            # 1. Binary payloads: skip regex scanning entirely.
            if looks_like_binary(body):
                logger.debug("[%s] Binary payload detected — skipping text scan", request_id)
            else:
                payload_text = body.decode("utf-8", errors="ignore")

                # 2. Base64 that decodes to binary: skip regex scanning.
                if looks_like_base64(payload_text):
                    logger.debug("[%s] Base64-binary payload — skipping text scan", request_id)
                else:
                    # 3. Plain text / JSON: run the consolidated suspicious-content check.
                    if self._contains_suspicious_content(payload_text):
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
            
            # Reset circuit breaker on 2xx responses
            if 200 <= response.status_code < 300:
                self.circuit_breaker.record_success()
            
            # Add request ID to response headers
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
            
            # Track failure with circuit breaker
            is_broken = self.circuit_breaker.record_failure()
            
            if is_broken:
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
    
    def get_error_stats(self) -> dict:
        """Get error statistics"""
        status = self.circuit_breaker.get_status()
        return {
            "circuit_breaker": status,
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