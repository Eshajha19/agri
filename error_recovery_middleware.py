"""
Error Recovery Middleware for FastAPI
Provides structured error handling and recovery for async operations
"""

import base64
import logging
import re
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.exceptions import HTTPException
import traceback

logger = logging.getLogger(__name__)


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
    
    def __init__(self, app, scan_request_body: bool = True,
                 confidence_threshold: ConfidenceLevel = ConfidenceLevel.MEDIUM):
        super().__init__(app)
        self.error_counts = {}
        self.error_timestamps = {}
        self._scanner = SuspiciousContentScanner(confidence_threshold)
        self._scan_request_body = scan_request_body
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Handle request with error recovery"""
        
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Track timing
        start_time = time.time()
        endpoint = f"{request.method} {request.url.path}"
        
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
            
            # Log successful request
            duration = time.time() - start_time
            logger.info(
                f"[{request_id}] {endpoint} - Status: {response.status_code} - "
                f"Duration: {duration:.2f}s"
            )
            
            # Reset error count on success
            if endpoint in self.error_counts:
                self.error_counts[endpoint] = 0
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
        
        except HTTPException as http_exc:
            """Handle HTTP exceptions"""
            duration = time.time() - start_time
            
            logger.warning(
                f"[{request_id}] {endpoint} - HTTP Error: {http_exc.status_code} - "
                f"Detail: {http_exc.detail} - Duration: {duration:.2f}s"
            )
            
            return JSONResponse(
                status_code=http_exc.status_code,
                content={
                    "success": False,
                    "request_id": request_id,
                    "error": {
                        "message": http_exc.detail,
                        "status_code": http_exc.status_code,
                        "category": self._categorize_error(http_exc.status_code)
                    }
                }
            )
        
        except ValueError as val_exc:
            """Handle validation errors"""
            duration = time.time() - start_time
            
            logger.warning(
                f"[{request_id}] {endpoint} - Validation Error: {val_exc} - "
                f"Duration: {duration:.2f}s"
            )
            
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "request_id": request_id,
                    "error": {
                        "message": str(val_exc),
                        "status_code": 400,
                        "category": "validation"
                    }
                }
            )
        
        except TimeoutError as timeout_exc:
            """Handle timeout errors"""
            duration = time.time() - start_time
            
            logger.error(
                f"[{request_id}] {endpoint} - Timeout Error - Duration: {duration:.2f}s"
            )
            
            # Increment error count
            self.error_counts[endpoint] = self.error_counts.get(endpoint, 0) + 1
            
            return JSONResponse(
                status_code=504,
                content={
                    "success": False,
                    "request_id": request_id,
                    "error": {
                        "message": "Request timeout - please try again",
                        "status_code": 504,
                        "category": "network",
                        "recoverable": True
                    }
                }
            )
        
        except Exception as exc:
            """Handle unexpected errors"""
            duration = time.time() - start_time
            error_id = str(uuid.uuid4())
            
            logger.error(
                f"[{request_id}] {endpoint} - Unexpected Error [{error_id}]: {exc} - "
                f"Duration: {duration:.2f}s\n{traceback.format_exc()}"
            )
            
            # Increment error count
            self.error_counts[endpoint] = self.error_counts.get(endpoint, 0) + 1
            
            # Check circuit breaker
            is_broken = self._check_circuit_breaker(endpoint)
            
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
                            "error_id": error_id
                        }
                    }
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
                        "recoverable": True
                    }
                }
            )
    
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
            logger.warning(
                f"Circuit breaker opened for {endpoint}: "
                f"{error_count} errors in {time_since_error:.0f}s"
            )
            return True
        
        # Reset if 60 seconds have passed
        if time_since_error >= 60:
            self.error_counts[endpoint] = 0
        
        return False
    
    def get_error_stats(self) -> dict:
        """Get error statistics"""
        return {
            "endpoints_with_errors": len(self.error_counts),
            "error_counts": self.error_counts,
            "timestamps": {
                k: v for k, v in self.error_timestamps.items()
            }
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
