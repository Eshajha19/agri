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
    def __init__(self, app, program: SecretHygieneProgram = None):
        super().__init__(app)
        self.program = program or SecretHygieneProgram()

    async def dispatch(self, request: Request, call_next):
        # Determine whether the body can be meaningfully scanned.
        # Requests with a missing or unrecognised Content-Type are
        # treated as suspicious and scanned rather than silently trusted.
        raw_ct = request.headers.get("content-type")
        mime = _parse_mime_type(raw_ct)
        should_scan = mime is None or not mime.startswith(_BINARY_MIME_PREFIXES)

        if should_scan:
            try:
                body_bytes = await request.body()
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

        response = await call_next(request)
        return response

def build_secret_fingerprint(value: str) -> str:
    """Generate a cryptographically stable fingerprint of a secret value."""
    if not isinstance(value, str):
        value = str(value)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

def redact_sensitive_payload(payload: Any) -> Any:
    """Recursively traverses and masks PII / Secrets in request/response payloads."""
    if isinstance(payload, dict):
        new_payload = {}
        for k, v in payload.items():
            k_lower = str(k).lower()
            if "email" in k_lower:
                new_payload[k] = "[REDACTED_EMAIL]"
            elif any(s in k_lower for s in ["key", "secret", "token", "password", "auth"]):
                new_payload[k] = "[REDACTED_SECRET]"
            else:
                new_payload[k] = redact_sensitive_payload(v)
        return new_payload
    elif isinstance(payload, list):
        return [redact_sensitive_payload(item) for item in payload]
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
