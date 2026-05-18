from datetime import datetime, timedelta, timezone
import re

from fastapi import FastAPI
from fastapi.testclient import TestClient

from security_hygiene import (
    RuntimeProtectionMiddleware,
    SecretHygieneProgram,
    build_secret_fingerprint,
    redact_sensitive_payload,
    scan_text_for_secrets,
)


def _sample_stripe_secret() -> str:
    return "sample-secret-12345"


def _sample_aws_access_key() -> str:
    return "sample-key-12345"


def test_scan_detects_high_risk_secrets():
    program = SecretHygieneProgram()
    program.SECRET_PATTERNS = (
        ("custom_secret", re.compile(r"sample-secret-\d+")),
    )
    text = f"token={_sample_stripe_secret()} and key={_sample_aws_access_key()}"
    findings = program.scan_text(text, location="unit-test")

    categories = {finding.category for finding in findings}
    assert "custom_secret" in categories


def test_redaction_masks_pii_and_secrets():
    payload = {
        "email": "farmer@example.com",
        "api_key": _sample_aws_access_key(),
        "note": "Call +91 98765 43210",
    }

    redacted = redact_sensitive_payload(payload)
    assert redacted["email"] == "[REDACTED_EMAIL]"
    assert redacted["api_key"] == "[REDACTED_SECRET]"
    assert "[REDACTED_PHONE]" in redacted["note"]


def test_rotation_registry_marks_due_items():
    program = SecretHygieneProgram(rotation_days=30)
    program.register_secret("TWILIO_AUTH_TOKEN")
    program.mark_rotated(
        "TWILIO_AUTH_TOKEN",
        rotated_at=datetime.now(timezone.utc) - timedelta(days=45),
    )

    assert "TWILIO_AUTH_TOKEN" in program.rotation_due()


def test_runtime_middleware_blocks_secret_leakage():
    app = FastAPI()
    program = SecretHygieneProgram()
    program.SECRET_PATTERNS = (
        ("custom_secret", re.compile(r"sample-secret-\d+")),
    )
    app.add_middleware(RuntimeProtectionMiddleware, program=program)

    @app.post("/submit")
    async def submit(payload: dict):
        return {"ok": True, "payload": payload}

    client = TestClient(app)
    response = client.post("/submit", json={"token": _sample_stripe_secret()})

    assert response.status_code == 400
    assert response.json()["error"] == "Request blocked by secrets hygiene policy"


def test_runtime_middleware_allows_normal_requests():
    app = FastAPI()
    app.add_middleware(RuntimeProtectionMiddleware, program=SecretHygieneProgram())

    @app.post("/submit")
    async def submit(payload: dict):
        return {"ok": True, "payload": payload}

    client = TestClient(app)
    response = client.post("/submit", json={"name": "John Farmer", "email": "john@example.com"})

    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_repo_scan_helper_works_on_strings():
    results = scan_text_for_secrets("nothing sensitive here")
    assert results == []


def test_secret_fingerprint_is_stable():
    value = "super-secret-value"
    assert build_secret_fingerprint(value) == build_secret_fingerprint(value)
