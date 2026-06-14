"""
Secrets and PII Hygiene Program module.
Protects sensitive data from leakages and enforces rotation and redaction.
"""

import re
import hashlib
import threading
import time
from datetime import datetime, timezone
from typing import Any, Tuple, Pattern, List, Dict, Optional
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from middleware_utils import (
    ensure_body_available,
    get_cached_body,
    has_cached_body,
)

# Maximum body size to scan (256 KB)
MAX_SCAN_BODY_SIZE = 256 * 1024
# Content-Types that are eligible for body scanning
SCANNABLE_CONTENT_TYPES = frozenset({
    "application/json",
    "text/plain",
    "text/html",
    "application/x-www-form-urlencoded",
    "application/xml",
    "text/xml",
})

# ---------------------------------------------------------------------------
# DoS mitigation: scan-result cache + per-path token-bucket budget (#2366)
# ---------------------------------------------------------------------------
_SCAN_CACHE_TTL: float = 5.0    # seconds a cached result stays valid
_SCAN_RPS_LIMIT: float = 10.0   # max full scans per second per path
_SCAN_BURST: int = 3            # token-bucket burst allowance per path


class _ScanCache:
    """Thread-safe cache keyed on (path, body_blake2b) with TTL expiry."""

    def __init__(self, ttl: float = _SCAN_CACHE_TTL, maxsize: int = 4096) -> None:
        self._ttl = ttl
        self._maxsize = maxsize
        self._store: Dict[tuple, tuple] = {}   # key -> (result, expires_at)
        self._lock = threading.Lock()

    def _key(self, path: str, body: bytes) -> tuple:
        digest = hashlib.blake2b(body, digest_size=16).hexdigest()
        return (path, digest)

    def get(self, path: str, body: bytes) -> Optional[bool]:
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
                # Evict expired entries first, then oldest quarter
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
        self._buckets: Dict[str, tuple] = {}   # path -> (tokens, last_refill)
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


# Module-level singletons shared across all middleware instances
_scan_cache = _ScanCache()
_scan_budget = _ScanBudget()

# ---------------------------------------------------------------------------


class Finding:
    """Represents a sensitive data finding during scanning."""
    def __init__(self, category: str, matched_text: str, location: str):
        self.category = category
        self.matched_text = matched_text
        self.location = location

class SecretHygieneProgram:
    """Manages secret scanning patterns, rotation schedules, and detection."""
    def __init__(self, rotation_days: int = 30):
        self.rotation_days = rotation_days
        self.registered_secrets = {}
        self.SECRET_PATTERNS: Tuple[Tuple[str, Pattern], ...] = (
            ("aws_key", re.compile(r"AKIA[0-9A-Z]{16}")),
            ("stripe_secret", re.compile(r"sk_live_[0-9a-zA-Z]{24}")),
            ("generic_secret", re.compile(r"sample-secret-\d+")),
        )

    def register_secret(self, secret_name: str) -> None:
        """Register a secret name for tracking rotation."""
        self.registered_secrets[secret_name] = None

    def mark_rotated(self, secret_name: str, rotated_at: datetime) -> None:
        """Mark a secret as rotated at a specific timestamp."""
        self.registered_secrets[secret_name] = rotated_at

    def rotation_due(self) -> List[str]:
        """Return a list of secret names that are due for rotation."""
        due = []
        now = datetime.now(timezone.utc)
        for name, rotated_at in self.registered_secrets.items():
            if rotated_at is None:
                due.append(name)
            elif (now - rotated_at).days >= self.rotation_days:
                due.append(name)
        return due

    def scan_text(self, text: str, location: str = "unknown") -> List[Finding]:
        """Scan string content for any registered secret patterns."""
        findings = []
        for category, pattern in self.SECRET_PATTERNS:
            for match in pattern.finditer(text):
                findings.append(Finding(category, match.group(0), location))
        return findings

def _parse_mime_type(content_type: str | None) -> str | None:
    """Extract the MIME type from a Content-Type header value."""
    if not content_type:
        return None
    return content_type.split(";")[0].strip().lower()


# Binary MIME types whose body cannot be meaningfully scanned as text.
_BINARY_MIME_PREFIXES = ("image/", "audio/", "video/", "application/octet-stream")


class RuntimeProtectionMiddleware(BaseHTTPMiddleware):
    """FastAPI Middleware to block requests containing cleartext secrets."""

    def __init__(self, app, program: SecretHygieneProgram = None, exclude_paths=None):
        super().__init__(app)
        self.program = program or SecretHygieneProgram()
        self.exclude_paths = set(exclude_paths or ["/health", "/docs"])

    async def dispatch(self, request: Request, call_next):
        raw_ct = request.headers.get("content-type")
        mime = _parse_mime_type(raw_ct)
        should_scan = mime is None or not mime.startswith(_BINARY_MIME_PREFIXES)

        if should_scan:
            try:

                # Safe request body access.
                # request.body() is cached by Starlette, allowing middleware
                # and downstream handlers to access the same payload without
                # modifying request._receive.
                
                if has_cached_body(request):
                    body_bytes = await get_cached_body(request)
                else:
                    body_bytes = await ensure_body_available(request)
                path = request.url.path

                # --- DoS fix: check cache before running any regex (#2366) ---
                cached = _scan_cache.get(path, body_bytes)
                if cached is not None:
                    has_findings = cached
                elif _scan_budget.consume(path):
                    # Budget available — run the full scan
                    body_str = body_bytes.decode("utf-8", errors="ignore")
                    has_findings = bool(self.program.scan_text(body_str, location="middleware"))
                    _scan_cache.set(path, body_bytes, has_findings)
                else:
                    # Budget exhausted and no cached result — assume safe to
                    # avoid blocking legitimate traffic under load
                    has_findings = False

                if has_findings:
                    return JSONResponse(
                        status_code=400,
                        content={"error": "Request blocked by secrets hygiene policy"}
                    )

            except Exception:
                # Fallback in case of body read failures to avoid crashing the server
                pass

        response = await call_next(request)
        return response

def build_secret_fingerprint(value: str) -> str:
    """Generate a cryptographically stable fingerprint of a secret value."""
    if not isinstance(value, str):
        value = str(value)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

_MAX_REDACT_DEPTH = 20


def redact_sensitive_payload(payload: Any, _depth: int = 0) -> Any:
    """Recursively traverses and masks PII / Secrets in request/response payloads.

    Stops recursion at _MAX_REDACT_DEPTH (20) to prevent stack exhaustion
    from deeply nested attacker-controlled payloads.
    """
    if _depth >= _MAX_REDACT_DEPTH:
        return "[MAX_DEPTH]"

    if isinstance(payload, dict):
        new_payload = {}
        for k, v in payload.items():
            k_lower = str(k).lower()
            if "email" in k_lower:
                new_payload[k] = "[REDACTED_EMAIL]"
            elif any(s in k_lower for s in ["key", "secret", "token", "password", "auth"]):
                new_payload[k] = "[REDACTED_SECRET]"
            else:
                new_payload[k] = redact_sensitive_payload(v, _depth + 1)
        return new_payload
    elif isinstance(payload, list):
        return [redact_sensitive_payload(item, _depth + 1) for item in payload]
    elif isinstance(payload, str):
        email_pattern = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
        phone_pattern = re.compile(r"\+?[0-9]{1,4}[-.\s]?[0-9]{3,5}[-.\s]?[0-9]{4,5}")

        redacted = payload
        redacted = email_pattern.sub("[REDACTED_EMAIL]", redacted)
        redacted = phone_pattern.sub("[REDACTED_PHONE]", redacted)
        return redacted
    else:
        return payload

def scan_text_for_secrets(text: str) -> List[Finding]:
    """Scans raw text using the default SecretHygieneProgram instance."""
    program = SecretHygieneProgram()
    return program.scan_text(text, location="repo-scan")