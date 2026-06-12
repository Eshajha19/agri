import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_missing_origin_rejected():
    token = issue_csrf_token("user1")
    r = client.post("/finance/transfer-funds", headers={"X-CSRF-Token": token})
    assert r.status_code == 403

def test_untrusted_origin_rejected():
    token = issue_csrf_token("user1")
    r = client.post("/finance/transfer-funds", headers={
        "X-CSRF-Token": token,
        "Origin": "http://evil.com"
    })
    assert r.status_code == 403

def test_invalid_token_rejected():
    r = client.post("/finance/transfer-funds", headers={
        "X-CSRF-Token": "badtoken",
        "Origin": "https://yourdomain.com"
    })
    assert r.status_code == 403

def test_expired_token_rejected(monkeypatch):
    token = issue_csrf_token("user1")
    # monkeypatch expiry to past
    CSRF_STORE[token]["expiry"] = datetime.utcnow() - timedelta(seconds=1)
    r = client.post("/finance/transfer-funds", headers={
        "X-CSRF-Token": token,
        "Origin": "https://yourdomain.com"
    })
    assert r.status_code == 403
