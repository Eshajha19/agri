import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_missing_token_rejected():
    with client.websocket_connect("/api/notifications/stream") as ws:
        data = ws.receive_json()
        assert data is None  # connection closed
        assert ws.closed

def test_invalid_token_rejected(monkeypatch):
    monkeypatch.setattr("backend.main.decode_and_validate_token", lambda t: (_ for _ in ()).throw(Exception("bad")))
    with pytest.raises(Exception):
        client.websocket_connect("/api/notifications/stream", headers={"Authorization": "badtoken"})

def test_valid_token_connect(monkeypatch):
    monkeypatch.setattr("backend.main.decode_and_validate_token", lambda t: {"sub": "user1", "regions": ["north"], "crops": ["wheat"]})
    with client.websocket_connect("/api/notifications/stream", headers={"Authorization": "goodtoken"}) as ws:
        # Should connect successfully
        ws.send_json({"type": "subscribe_crops", "crops": ["wheat"]})
        msg = ws.receive_json()
        assert "error" not in msg
