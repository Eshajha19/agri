import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_predict_endpoint_error_hidden(monkeypatch):
    # Force an exception in predict endpoint
    def bad_predict(*args, **kwargs):
        raise RuntimeError("Sensitive stack trace info")
    monkeypatch.setattr("backend.predict", bad_predict)

    r = client.post("/api/predict", json={"input": "bad"})
    assert r.status_code == 500
    body = r.json()
    # Should not leak raw exception
    assert "Sensitive" not in str(body)
    assert body["error"]["code"] == "internal_error"

def test_websocket_error_hidden(monkeypatch):
    with client.websocket_connect("/api/notifications/stream") as ws:
        ws.send_json({"type": "bad_message"})
        msg = ws.receive_json()
        # Should return structured error, not raw exception
        assert "error" in msg
        assert "internal_error" in msg["error"]["code"]
