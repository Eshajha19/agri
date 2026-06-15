from datetime import datetime, timedelta, timezone
import re

from fastapi import FastAPI
from fastapi.testclient import TestClient

from security_hygiene import (
    RuntimeProtectionMiddleware,
    SecretHygieneProgram,
    _parse_mime_type,
    SCANNABLE_TYPES,
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


# --- _parse_mime_type ---

def test_parse_mime_type_simple():
    assert _parse_mime_type("application/json") == "application/json"


def test_parse_mime_type_with_charset():
    assert _parse_mime_type("application/json; charset=utf-8") == "application/json"


def test_parse_mime_type_with_multiple_params():
    assert _parse_mime_type("text/html; charset=utf-8; boundary=abc") == "text/html"


def test_parse_mime_type_case_insensitive():
    assert _parse_mime_type("APPLICATION/JSON") == "application/json"


def test_parse_mime_type_handles_whitespace():
    assert _parse_mime_type("  application/json  ") == "application/json"


def test_parse_mime_type_none_returns_none():
    assert _parse_mime_type(None) is None


def test_parse_mime_type_empty_string_returns_none():
    assert _parse_mime_type("") is None


def test_parse_mime_type_whitespace_only_returns_none():
    assert _parse_mime_type("   ") is None


# --- RuntimeProtectionMiddleware content-type guards ---

def test_middleware_skips_multipart():
    app = FastAPI()
    app.add_middleware(RuntimeProtectionMiddleware)

    @app.post("/upload")
    async def upload(payload: dict):
        return {"ok": True}

    client = TestClient(app)
    response = client.post(
        "/upload",
        files={"file": ("test.txt", b"safe content")},
    )
    # multipart is not in SCANNABLE_TYPES, so should pass through
    assert response.status_code == 200


def test_middleware_skips_unsupported_content_type():
    app = FastAPI()
    app.add_middleware(RuntimeProtectionMiddleware)

    @app.post("/data")
    async def data(payload: dict):
        return {"ok": True}

    client = TestClient(app)
    response = client.post(
        "/data",
        content=b"some binary data",
        headers={"content-type": "application/octet-stream"},
    )
    assert response.status_code == 200


def test_middleware_scans_json_content_type():
    app = FastAPI()
    program = SecretHygieneProgram()
    program.SECRET_PATTERNS = (("test_secret", re.compile(r"SENSITIVE")),)
    app.add_middleware(RuntimeProtectionMiddleware, program=program)

    @app.post("/submit")
    async def submit(payload: dict):
        return {"ok": True}

    client = TestClient(app)
    response = client.post(
        "/submit",
        json={"data": "SENSITIVE"},
    )
    assert response.status_code == 400


def test_scannable_types_include_expected():
    assert "application/json" in SCANNABLE_TYPES
    assert "application/x-www-form-urlencoded" in SCANNABLE_TYPES
    assert "text/plain" in SCANNABLE_TYPES


def test_middleware_handles_missing_content_type():
    app = FastAPI()
    app.add_middleware(RuntimeProtectionMiddleware)

    @app.post("/data")
    async def data(payload: dict):
        return {"ok": True}

    client = TestClient(app)
    # No Content-Type header
    response = client.post("/data", content=b"{}", headers={})
    assert response.status_code == 200


# --- MAX_SCAN_BODY_SIZE protection ---

from security_hygiene import MAX_SCAN_BODY_SIZE


def test_rejects_payload_larger_than_limit():
    app = FastAPI()
    app.add_middleware(RuntimeProtectionMiddleware)

    @app.post("/submit")
    async def submit():
        return {"ok": True}

    client = TestClient(app)

    response = client.post(
        "/submit",
        content=b"x" * (MAX_SCAN_BODY_SIZE + 1),
        headers={
            "content-type": "application/json",
            "content-length": str(MAX_SCAN_BODY_SIZE + 1),
        },
    )

    assert response.status_code == 413


def test_allows_payload_at_limit():
    app = FastAPI()
    app.add_middleware(RuntimeProtectionMiddleware)

    @app.post("/submit")
    async def submit():
        return {"ok": True}

    client = TestClient(app)

    response = client.post(
        "/submit",
        content=b"x" * MAX_SCAN_BODY_SIZE,
        headers={
            "content-type": "application/json",
            "content-length": str(MAX_SCAN_BODY_SIZE),
        },
    )

    assert response.status_code == 200


def test_missing_content_length_does_not_fail():
    app = FastAPI()
    app.add_middleware(RuntimeProtectionMiddleware)

    @app.post("/submit")
    async def submit():
        return {"ok": True}

    client = TestClient(app)

    response = client.post(
        "/submit",
        content=b"{}",
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 200
