import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_missing_signature(monkeypatch):
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "secret")
    r = client.post("/api/whatsapp/webhook", data={"From": "whatsapp:+123456"})
    assert r.status_code == 403

def test_invalid_signature(monkeypatch):
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "secret")
    r = client.post("/api/whatsapp/webhook", data={"From": "whatsapp:+123456"}, headers={"X-Twilio-Signature": "bad"})
    assert r.status_code == 403

def test_body_too_large(monkeypatch):
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "secret")
    big = "x" * (70 * 1024)
    r = client.post("/api/whatsapp/webhook", data={"From": "whatsapp:+123456"}, content=big, headers={"X-Twilio-Signature": "fake"})
    assert r.status_code == 413

def test_malformed_form(monkeypatch):
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "secret")
    r = client.post("/api/whatsapp/webhook", data="%%%notform%%%", headers={"X-Twilio-Signature": "fake"})
    assert r.status_code in (400, 403)

def test_valid_request(monkeypatch):
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "secret")
    # TODO: compute valid signature for test
    r = client.post("/api/whatsapp/webhook", data={"From": "whatsapp:+123456", "Body": "hello"}, headers={"X-Twilio-Signature": "validsig"})
    assert r.status_code == 200
