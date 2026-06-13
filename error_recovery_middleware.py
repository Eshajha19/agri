"""
Error Recovery Middleware for FastAPI
Provides structured error handling and recovery for async operations
"""

import base64
import collections
import logging
import random
import re
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from urllib.parse import urlparse
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
        return self._state.get(key)

    def set(self, method: str, path: str, value: dict):
        key = self._normalize_key(method, path)
        if key in self._state:
            self._state.move_to_end(key)
        self._state[key] = value
        # Prune oldest if over cap
        if len(self._state) > self._max_entries:
            self._state.popitem(last=False)


# ── Suspicious content scanner (encoding-aware) ────────────────────────


class ConfidenceLevel(Enum):
    HIGH = "high"          # clean UTF-8 decode, no replacement chars
    MEDIUM = "medium"      # partial UTF-8 or Latin-1, few artifacts
    LOW = "low"            # binary-heavy or garbled decode
    SKIP = "skip"          # confidence below threshold – do not scan


@dataclass
class ScanResult:
    """Result of a suspicious-content scan."""
    suspicious: bool
    confidence: ConfidenceLevel
    threat_type: Optional[str] = None
    decoded_text: Optional[str] = None


class SuspiciousContentScanner:
    """Encoding-aware scanner that only runs regex on high-confidence decodes.

    Design rationale
    ────────────────
    Raw request bodies may arrive in mixed encodings (UTF-8 embedded in
    binary segments, base64 fragments, Latin-1 fallback, …).  A naive
    decode-then-scan pipeline will produce false positives when the
    decoder emits garbled text that happens to match a regex pattern.

    This scanner solves that by:

    1. Trying UTF-8 first (strict).  If it succeeds with zero replacement
       characters → HIGH confidence → safe to scan.
    2. Falling back to Latin-1 (all 256 bytes are valid).  A Latin-1
       decode always succeeds but may contain binary garbage →
       MEDIUM confidence → still scanned but logged with lower severity.
    3. Explicitly detecting base64-encoded or binary-heavy payloads
       before attempting any scan → LOW confidence → skipped.
    """

    # Suspicious regex patterns (focused on actual attack payloads).
    SUSPICIOUS_PATTERNS: list[re.Pattern] = [
        re.compile(r"<\s*script[^>]*>", re.IGNORECASE),               # XSS
        re.compile(r"'\s*OR\s*'?\d*'\s*=\s*'?\d*'?", re.IGNORECASE), # SQLi tautology
        re.compile(r"'\s*--\s*", re.IGNORECASE),                      # SQLi comment
        re.compile(r"'\s*;\s*DROP\s+TABLE", re.IGNORECASE),           # SQLi drop
        re.compile(r"'\s*;\s*SELECT\s+.*\s+FROM", re.IGNORECASE),     # SQLi union
        re.compile(r"\{\{.*\}\}"),                                     # SSTI (Jinja2)
        re.compile(r"#\{.*\}"),                                        # SSTI (Ruby)
        re.compile(r"\$\(.*\)"),                                       # Shell injection
        re.compile(r"`[^`]+`"),                                        # Shell backtick
        re.compile(r"\|.*sh\b", re.IGNORECASE),                        # Pipe to shell
        re.compile(r"__import__|__subclasses__|__globals__", re.I),    # Python SSTI
    ]

    # ── Public API ──────────────────────────────────────────────────────

    def __init__(self, confidence_threshold: ConfidenceLevel = ConfidenceLevel.MEDIUM):
        self._threshold = confidence_threshold

    def scan(self, raw_bytes: bytes) -> ScanResult:
        """Analyse *raw_bytes* for suspicious content.

        Returns a ``ScanResult``.  If the decoding confidence is below
        the configured threshold, ``suspicious`` is *always* ``False``
        and ``confidence`` is ``SKIP``.
        """
        if not raw_bytes:
            return ScanResult(suspicious=False, confidence=ConfidenceLevel.HIGH)

        decoded, confidence = self._decode_with_confidence(raw_bytes)

        if self._skip_scan(confidence):
            return ScanResult(suspicious=False, confidence=ConfidenceLevel.SKIP,
                              decoded_text=decoded if confidence != ConfidenceLevel.SKIP else None)

        # Only scan when we have meaningful text.
        for pattern in self.SUSPICIOUS_PATTERNS:
            if pattern.search(decoded):
                return ScanResult(suspicious=True, confidence=confidence,
                                  threat_type=pattern.pattern, decoded_text=decoded)

        return ScanResult(suspicious=False, confidence=confidence, decoded_text=decoded)

    # ── Decoding pipeline ───────────────────────────────────────────────

    @staticmethod
    def _decode_with_confidence(raw: bytes) -> tuple[str, ConfidenceLevel]:
        """Return ``(decoded_text, confidence)``."""
        # 1) Detect base64-packed payloads early.
        if SuspiciousContentScanner._is_base64_like(raw):
            try:
                decoded_b64 = base64.b64decode(raw, validate=True)
                # Recurse once so we scan the *decoded* content.
                return SuspiciousContentScanner._decode_with_confidence(decoded_b64)
            except (base64.binascii.Error, ValueError):
                pass  # fall through to normal decode

        # 2) Detect binary payloads early.
        if SuspiciousContentScanner._is_binary(raw):
            # Best-effort Latin-1 so we can at least log the shape.
            return raw.decode("latin-1"), ConfidenceLevel.LOW

        # 3) Strict UTF-8.
        try:
            text = raw.decode("utf-8")
            # Check for Unicode replacement characters.
            if "\ufffd" in text:
                return text, ConfidenceLevel.MEDIUM
            return text, ConfidenceLevel.HIGH
        except UnicodeDecodeError:
            pass

        # 4) Latin-1 fallback (always succeeds).
        text = raw.decode("latin-1")
        return text, ConfidenceLevel.MEDIUM

    # ── Classification helpers ──────────────────────────────────────────

    @staticmethod
    def _is_binary(raw: bytes) -> bool:
        """Heuristic: > 5 % non-printable / non-whitespace bytes → binary."""
        if not raw:
            return False
        control = sum(
            1 for b in raw if b < 32 and b not in (9, 10, 13)  # tab, nl, cr
        )
        return (control / len(raw)) > 0.05

    @staticmethod
    def _is_base64_like(raw: bytes) -> bool:
        """Heuristic: looks like a base64-encoded payload."""
        if len(raw) < 8:
            return False
        # Base64 only uses A-Z a-z 0-9 + / = and optional whitespace.
        cleaned = raw.translate(None, b"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                     b"abcdefghijklmnopqrstuvwxyz"
                                     b"0123456789+/=\n\r\t ")
        # More than 10 % non-base64 chars → not base64.
        return len(cleaned) / len(raw) < 0.10

    # ── Internal ────────────────────────────────────────────────────────

    def _skip_scan(self, level: ConfidenceLevel) -> bool:
        """Return True when *level* is below the configured threshold."""
        order = [ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM, ConfidenceLevel.LOW,
                 ConfidenceLevel.SKIP]
        return order.index(level) > order.index(self._threshold)


# ── Helper functions ──────────────────────────────────────────────────────


def looks_like_binary(data: bytes) -> bool:
    """Heuristic: > 5% non-printable / non-whitespace bytes → binary."""
    if not data:
        return False
    control = sum(1 for b in data if b < 32 and b not in (9, 10, 13))
    return (control / len(data)) > 0.05


def looks_like_base64(text: str) -> bool:
    """Heuristic: looks like a base64-encoded payload."""
    if len(text) < 8:
        return False
    cleaned = re.sub(r'[A-Za-z0-9+/=\n\r\t ]', '', text)
    return len(cleaned) / len(text) < 0.10 if text else False


async def ensure_body_available(request: Request) -> bytes:
    """Read request body once and cache it on request.state."""
    if hasattr(request.state, '_body_cache'):
        return request.state._body_cache
    body = await request.body()
    request.state._body_cache = body
    return body


# ── Error-recovery middleware ──────────────────────────────────────────


class ErrorRecoveryMiddleware(BaseHTTPMiddleware):
    """
    Middleware for handling errors with structured recovery

    Features:
    - Automatic error tracking
    - Structured error responses
    - Request/response logging
    - Circuit breaker integration
    - Suspicious content scanning (encoding-aware)
    """
    
    # Circuit breaker state constants
    _OPEN = "open"
    _CLOSED = "closed"
    _HALF_OPEN = "half-open"
    
    # Circuit breaker configuration
    _FAILURE_THRESHOLD = 5
    _RESET_TIMEOUT = 60.0
    _JITTER_MAX = 30.0
    
    def __init__(self, app, scan_request_body: bool = True,
                 confidence_threshold: ConfidenceLevel = ConfidenceLevel.MEDIUM):
        super().__init__(app)
        self.error_counts = {}
        self.error_timestamps = {}
        self._scanner = SuspiciousContentScanner(confidence_threshold)
        self._scan_request_body = scan_request_body
        
        # Initialize circuit breaker state
        self._circuit_state = {}
        self._circuit_open_since = {}
        self._failure_timestamps = {}
        self.circuit_breaker = CircuitBreakerAsync()
    
    def _contains_suspicious_content(self, text: str) -> bool:
        """Check if text contains suspicious patterns."""
        for pattern in SQL_PATTERNS + [re.compile(p) for p in SUSPICIOUS_PATTERNS]:
            if pattern.search(text):
                return True
        return False
    
    def _rate_log(self, key: str, level: int, message: str, *args) -> None:
        """Rate-limited logging to prevent log spam."""
        now = time.time()
        last_logged = self.error_timestamps.get(key, 0)
        count = self.error_counts.get(key, 0)
        
        # Log at most once every 5 seconds per key
        if now - last_logged >= 5.0:
            self.error_counts[key] = 1
            self.error_timestamps[key] = now
            logger.log(level, message, *args)
        else:
            self.error_counts[key] = count + 1
    
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
            # Suspicious content scan on request body (encoding-aware)
            if self._scan_request_body and request.method in ("POST", "PUT", "PATCH"):
                body_bytes = await request.body()
                scan_result = self._scanner.scan(body_bytes)
                if scan_result.suspicious:
                    logger.warning(
                        "[%s] Suspicious content detected (confidence=%s, threat=%s): %s",
                        request_id, scan_result.confidence.value, scan_result.threat_type,
                        endpoint,
                    )
                    return JSONResponse(
                        status_code=400,
                        content={
                            "success": False,
                            "request_id": request_id,
                            "error": {
                                "message": "Request content was flagged as suspicious",
                                "status_code": 400,
                                "category": "validation",
                                "recoverable": False,
                            },
                        },
                    )

            # Call the endpoint
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
                        "message": "Invalid request",
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