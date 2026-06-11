"""
Secrets and PII Hygiene Program module.
Protects sensitive data from leakages and enforces rotation and redaction.
"""

import re
import hashlib
from datetime import datetime, timezone
from typing import Any, Tuple, Pattern, List
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

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

# Content-types eligible for body scanning (MIME type only, no parameters)
SCANNABLE_TYPES = frozenset({
    "application/json",
    "application/x-www-form-urlencoded",
    "application/xml",
    "text/plain",
    "text/html",
    "text/xml",
})
# Maximum body size to scan (256 KB)
MAX_SCAN_BODY_SIZE = 256 * 1024


def _parse_mime_type(content_type: str | None) -> str | None:
    """Extract the MIME type from a Content-Type header value.

    Handles ``application/json; charset=utf-8`` → ``application/json``,
    missing or empty header → ``None``, and trailing whitespace.
    """
    if not content_type or not content_type.strip():
        return None
    return content_type.split(";")[0].strip().lower()


class RuntimeProtectionMiddleware(BaseHTTPMiddleware):
    """FastAPI Middleware to block requests containing cleartext secrets."""
    def __init__(self, app, program: SecretHygieneProgram = None):
        super().__init__(app)
        self.program = program or SecretHygieneProgram()
        self.exclude_paths = exclude_paths or []

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if any(path.startswith(prefix) for prefix in self.exclude_paths):
            await self.app(scope, receive, send)
            return

        body_bytes = b""
        more_body = True
        received_messages = []

    async def dispatch(self, request: Request, call_next):
        # Only scan text/json content types with bounded size
        mime = _parse_mime_type(request.headers.get("content-type"))

        if mime in SCANNABLE_TYPES:
            try:
                body_bytes = ensure_body_available(request)
                if 0 < len(body_bytes) <= MAX_SCAN_BODY_SIZE:
                    content_type = (request.headers.get("content-type") or "").lower().split(";")[0].strip()
                    content_length_str = request.headers.get("content-length", "0")
        try:
            content_length = int(content_length_str)
        except (ValueError, TypeError):
            content_length = 0

        should_scan = (
            content_type in SCANNABLE_CONTENT_TYPES
            and 0 < content_length <= MAX_SCAN_BODY_SIZE
        )

        if should_scan:
            try:
                body_bytes = ensure_body_available(request)
                if len(body_bytes) <= MAX_SCAN_BODY_SIZE:
                    body_str = body_bytes.decode("utf-8", errors="ignore")
                    findings = self.program.scan_text(body_str, location="middleware")
                    if findings:
                        return JSONResponse(
                            status_code=400,
                            content={"error": "Request blocked by secrets hygiene policy"}
                        )

                    # Reset body read pointer so downstream handlers can consume it
                    async def receive():
                        return {"type": "http.request", "body": body_bytes, "more_body": False}
                    request._receive = receive
            except Exception:
                # Fallback in case of body read failures to avoid crashing the server
                pass

        message_idx = 0
        async def mock_receive():
            nonlocal message_idx
            if message_idx < len(received_messages):
                msg = received_messages[message_idx]
                message_idx += 1
                return msg
            return await receive()

        await self.app(scope, mock_receive, send)

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
