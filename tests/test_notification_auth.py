"""
Integration tests for authenticated notification endpoints.

Skipped when optional application dependencies are not installed locally.
"""

import pytest

pytest.importorskip("qrcode")

from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient

import main
from realtime_notifications import NotificationBroadcastHub


def _make_fake_firestore(roles):
    class _FakeDoc:
        def __init__(self, role):
            self.exists = True
            self._role = role

        def to_dict(self):
            return {"role": self._role}

    class _FakeUserRef:
        def __init__(self, uid):
            self._uid = uid

        def get(self):
            role = roles.get(self._uid)
            if role is None:
                return type("Missing", (), {"exists": False})()
            return _FakeDoc(role)

    class _FakeCollection:
        def document(self, uid):
            return _FakeUserRef(uid)

    class _FakeFirestore:
        def collection(self, name):
            return _FakeCollection()

    return _FakeFirestore()


@pytest.fixture()
def notification_client(monkeypatch):
    fake_db = _make_fake_firestore({"farmer-user": "farmer"})
    monkeypatch.setattr(main.auth, "verify_id_token", lambda token: {"uid": token})
    monkeypatch.setattr(main, "db_firestore", fake_db)
    monkeypatch.setattr(main.RBACManager, "get_db", lambda: fake_db)
    return TestClient(main.app)


def test_get_notifications_requires_auth(notification_client):
    response = notification_client.get("/api/notifications")
    assert response.status_code == 401


def test_get_notifications_authenticated(notification_client):
    response = notification_client.get(
        "/api/notifications",
        headers={"Authorization": "Bearer farmer-user"},
    )
    assert response.status_code == 200
    assert response.json()["success"] is True
    assert isinstance(response.json()["data"], list)


def test_websocket_rejects_missing_token():
    app = FastAPI()
    hub = NotificationBroadcastHub(history_limit=10)

    @app.websocket("/api/notifications/stream")
    async def notifications_stream(websocket: WebSocket):
        uid = await main._authenticate_notification_websocket(websocket)
        if uid is None:
            return
        await hub.connect(websocket, uid)

    client = TestClient(app)
    with pytest.raises(Exception):
        with client.websocket_connect("/api/notifications/stream"):
            pass


def test_websocket_scoped_snapshot(monkeypatch):
    monkeypatch.setattr(main.auth, "verify_id_token", lambda token: {"uid": token})
    monkeypatch.setattr(main, "db_firestore", _make_fake_firestore({"alice": "farmer", "bob": "farmer"}))

    app = FastAPI()
    hub = NotificationBroadcastHub(history_limit=10)
    hub.seed_notifications(
        [
            {"id": 1, "type": "broadcast", "message": "for all", "recipient_uid": None},
            {"id": 2, "type": "private", "message": "for alice", "recipient_uid": "alice"},
        ]
    )

    @app.websocket("/api/notifications/stream")
    async def notifications_stream(websocket: WebSocket):
        uid = await main._authenticate_notification_websocket(websocket)
        if uid is None:
            return
        await hub.connect(websocket, uid)

    client = TestClient(app)

    with client.websocket_connect("/api/notifications/stream?token=alice") as ws:
        snapshot = ws.receive_json()
        assert snapshot["type"] == "snapshot"
        assert len(snapshot["data"]) == 2

    with client.websocket_connect("/api/notifications/stream?token=bob") as ws:
        snapshot = ws.receive_json()
        assert snapshot["type"] == "snapshot"
        assert len(snapshot["data"]) == 1
        assert snapshot["data"][0]["message"] == "for all"
