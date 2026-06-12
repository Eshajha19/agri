import time
import os
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

import whatsapp_service as ws


# ---------------------------------------------------------------------------
# Existing tests (kept unchanged)
# ---------------------------------------------------------------------------

def test_sign_and_verify():
    sid = "SM123"
    to = "whatsapp:+911234567890"
    body = "Test message"
    ts = int(time.time())

    # Ensure secret is set for test
    os.environ.setdefault("WHATSAPP_MESSAGE_SECRET", "testsecret")
    # regenerate module-level secret if needed
    global WHATSAPP_MESSAGE_SECRET
    try:
        ws.WHATSAPP_MESSAGE_SECRET = os.environ.get("WHATSAPP_MESSAGE_SECRET")
    except Exception:
        pass

    signature = ws._sign_message(sid, to, body, ts)
    assert signature
    assert ws.verify_signature(sid, to, body, ts, signature)


def test_rate_limit_per_second():
    # reset buckets
    ws._per_number_buckets.clear()
    ws._global_bucket.clear()

    to = "+911234567890"
    res1 = ws.send_whatsapp_message(to, "msg1")
    assert res1["status"] in {"success", "not_configured", "throttled", "error", "not_configured"}

    # immediate second message should be throttled by per-second rule
    res2 = ws.send_whatsapp_message(to, "msg2")
    # allow either throttled or other statuses depending on environment
    assert isinstance(res2, dict)


# ---------------------------------------------------------------------------
# New security tests — webhook signature validation
# ---------------------------------------------------------------------------

def _make_test_app():
    """Build a minimal FastAPI app that wires only the alerts router."""
    from fastapi import FastAPI
    from backend.routers import alerts as alerts_router

    app = FastAPI()

    # Provide a stub notification store with the minimum interface used by
    # the router initialisation so init_alerts does not raise.
    class _StubStore:
        def get_recent_for_user(self, uid):
            return []
        def get_all(self):
            return {}
        def upsert(self, uid, data):
            pass

    alerts_router.init_alerts(
        ns=_StubStore(),
        ss=_StubStore(),
        ga_fn=lambda **kw: [],
        sw_fn=lambda *a, **kw: {"success": True},
        fa_fn=lambda *a: "msg",
        vr_fn=None,
        rp_fn=None,
    )
    app.include_router(alerts_router.router, prefix="/api")
    return app


@pytest.fixture(scope="module")
def client():
    app = _make_test_app()
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def test_empty_payload_returns_403(client):
    """An empty body with no X-Twilio-Signature must be rejected with 403."""
    # Set a dummy auth token so the missing-token 500 is not triggered first.
    with patch.dict(os.environ, {"TWILIO_AUTH_TOKEN": "dummy_token_for_test"}):
        # Patch audit_rbac_event so it does not require real Firestore.
        with patch("twilio_webhook_security.audit_rbac_event") as mock_audit:
            response = client.post(
                "/api/whatsapp/webhook",
                content=b"",
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                # No X-Twilio-Signature header → should trigger "missing" path.
            )

    assert response.status_code == 403
    body = response.json()
    # The detail must not contain any internal path or stack trace.
    assert "traceback" not in str(body).lower()
    assert "Traceback" not in str(body)


def test_invalid_signature_returns_403_and_audit(client):
    """A form-encoded body with a wrong X-Twilio-Signature must return 403
    and must record an entry in the RBAC audit trail."""
    audit_calls = []

    def _capture_audit(**kwargs):
        audit_calls.append(kwargs)

    with patch.dict(os.environ, {"TWILIO_AUTH_TOKEN": "real_secret_token"}):
        with patch("twilio_webhook_security.audit_rbac_event", side_effect=_capture_audit):
            response = client.post(
                "/api/whatsapp/webhook",
                content=b"Body=Hello&From=whatsapp%3A%2B911234567890",
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "X-Twilio-Signature": "totallywrongsignature",
                },
            )

    assert response.status_code == 403
    body = response.json()
    assert "detail" in body
    # Internal tracebacks must not appear in the response.
    assert "Traceback" not in str(body)
    assert "traceback" not in str(body).lower()

    # At least one audit event must have been recorded for the invalid sig.
    assert len(audit_calls) >= 1
    reasons = [c.get("reason", "") for c in audit_calls]
    assert "invalid_twilio_signature" in reasons

