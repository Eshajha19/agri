"""
Tests for Twilio WhatsApp webhook signature verification.
"""

import base64
import hashlib
import hmac
import urllib.parse

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from twilio_webhook_security import (
    handle_inbound_whatsapp_webhook,
    verify_twilio_signature,
)


def _sign(url: str, body: bytes, auth_token: str) -> str:
    params = sorted(
        urllib.parse.parse_qsl(body.decode("utf-8"), keep_blank_values=True),
        key=lambda kv: kv[0],
    )
    signing_string = url + "".join(k + v for k, v in params)
    digest = hmac.new(
        auth_token.encode("utf-8"),
        signing_string.encode("utf-8"),
        hashlib.sha1,
    ).digest()
    return base64.b64encode(digest).decode("utf-8")


@pytest.fixture()
def webhook_app(monkeypatch):
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "test-auth-token")
    queued = []

    class _FakeTask:
        @staticmethod
        def delay(body, sender):
            queued.append((body, sender))

    monkeypatch.setattr(
        "twilio_webhook_security.enqueue_whatsapp_webhook_processing",
        lambda body, sender: queued.append((body, sender)),
    )

    app = FastAPI()

    @app.post("/api/whatsapp/webhook")
    async def webhook(request: Request):
        return await handle_inbound_whatsapp_webhook(request)

    client = TestClient(app)
    return client, queued


def test_missing_signature_returns_403(webhook_app):
    client, _queued = webhook_app
    body = b"Body=hello&From=whatsapp%3A%2B15551234567"
    response = client.post(
        "/api/whatsapp/webhook",
        content=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert response.status_code == 403
    assert response.json()["detail"] == "Missing Twilio signature"


def test_invalid_signature_returns_403(webhook_app):
    client, _queued = webhook_app
    body = b"Body=hello&From=whatsapp%3A%2B15551234567"
    response = client.post(
        "/api/whatsapp/webhook",
        content=body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "X-Twilio-Signature": "invalid",
        },
    )
    assert response.status_code == 403
    assert response.json()["detail"] == "Invalid Twilio signature"


def test_valid_signature_enqueues_task(webhook_app):
    client, queued = webhook_app
    body = b"Body=hello&From=whatsapp%3A%2B15551234567"
    url = "http://testserver/api/whatsapp/webhook"
    signature = _sign(url, body, "test-auth-token")

    response = client.post(
        "/api/whatsapp/webhook",
        content=body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "X-Twilio-Signature": signature,
        },
    )
    assert response.status_code == 200
    assert response.json() == {"status": "success"}
    assert queued == [("hello", "+15551234567")]


def test_verify_twilio_signature_unit(monkeypatch):
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "secret")
    auth_token = "secret"
    body = b"Body=Hi&From=whatsapp%3A%2B19998887777"
    url = "https://example.com/api/whatsapp/webhook"
    signature = _sign(url, body, auth_token)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/whatsapp/webhook",
        "headers": [
            (b"x-twilio-signature", signature.encode()),
            (b"host", b"example.com"),
        ],
        "scheme": "https",
        "server": ("example.com", 443),
    }

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request(scope, receive)
    verify_twilio_signature(request, body)
