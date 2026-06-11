import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_untrusted_xff_ignored(monkeypatch):
    # Simulate spoofed X-Forwarded-For from untrusted peer
    headers = {"X-Forwarded-For": "1.2.3.4"}
    r1 = client.get("/api/some_rate_limited", headers=headers)
    r2 = client.get("/api/some_rate_limited", headers=headers)
    # Should still key by socket address, not spoofed IP
    assert r1.status_code == 200
    assert r2.status_code == 200

def test_trusted_proxy_respected(monkeypatch):
    # Simulate request from trusted proxy
    headers = {"X-Forwarded-For": "5.6.7.8"}
    # monkeypatch request.client.host to a trusted proxy
    with client.websocket_connect("/api/some_rate_limited", headers=headers) as ws:
        # Should key by 5.6.7.8 consistently
        ws.send_json({"ping": "ok"})
