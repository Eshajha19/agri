import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_rate_limit_by_uid(monkeypatch):
    # Simulate authenticated user
    class FakeUser:
        uid = "user123"
    def fake_user(request):
        request.user = FakeUser()
        return request

    monkeypatch.setattr("backend.feedback_api.rate_limit_key", fake_user)

    # First request should succeed
    r1 = client.post("/submit-feedback", json={"message": "ok"})
    assert r1.status_code == 200

    # Exceed limit
    for _ in range(5):
        client.post("/submit-feedback", json={"message": "spam"})
    r2 = client.post("/submit-feedback", json={"message": "spam"})
    assert r2.status_code == 429  # Too Many Requests
