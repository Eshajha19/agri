"""
Tests for the real-time notification broker and websocket fan-out.
"""

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import json

import pytest

from realtime_notifications import NotificationBroadcastHub

TEST_TOKEN = "test-valid-token"


def create_test_app():
    app = FastAPI()
    hub = NotificationBroadcastHub(history_limit=10)

    @app.websocket("/api/notifications/stream")
    async def notifications_stream(websocket: WebSocket):
        auth_header = websocket.headers.get("authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            await websocket.close(code=4001)
            return
        token = auth_header[7:].strip()
        if token != TEST_TOKEN:
            await websocket.close(code=4001)
            return
        await hub.connect(websocket)

    @app.post("/api/notifications/test-publish")
    async def publish_notification():
        await hub.publish(
            {
                "id": 101,
                "type": "weather",
                "message": "Heavy rainfall expected in your region today.",
                "time": "2026-05-20T10:00:00",
                "recipient_uid": None,
            }
        )
        return {"success": True}

    return app, hub


def test_websocket_receives_snapshot_and_live_notification():
    app, hub = create_test_app()
    hub.seed_notifications(
        [
            {
                "id": 1,
                "type": "advisory",
                "message": "Irrigate crops early in the morning.",
                "time": "2026-05-20T09:00:00",
                "recipient_uid": None,
            }
        ]
    )
    client = TestClient(app)

    with client.websocket_connect("/api/notifications/stream", headers={"Authorization": f"Bearer {TEST_TOKEN}"}) as websocket:
        snapshot = websocket.receive_json()
        assert snapshot["type"] == "snapshot"
        assert len(snapshot["data"]) == 1
        assert snapshot["data"][0]["message"] == "Irrigate crops early in the morning."

        response = client.post("/api/notifications/test-publish")
        assert response.status_code == 200
        assert response.json()["success"] is True

        event = websocket.receive_json()
        assert event["type"] == "notification"
        assert event["data"]["message"] == "Heavy rainfall expected in your region today."


def test_multiple_clients_receive_same_broadcast():
    app, hub = create_test_app()
    client = TestClient(app)

    with client.websocket_connect("/api/notifications/stream", headers={"Authorization": f"Bearer {TEST_TOKEN}"}) as ws1:
        snapshot1 = ws1.receive_json()
        assert snapshot1["type"] == "snapshot"

        with client.websocket_connect("/api/notifications/stream", headers={"Authorization": f"Bearer {TEST_TOKEN}"}) as ws2:
            snapshot2 = ws2.receive_json()
            assert snapshot2["type"] == "snapshot"

            response = client.post("/api/notifications/test-publish")
            assert response.status_code == 200

            event1 = ws1.receive_json()
            event2 = ws2.receive_json()

            assert event1["type"] == "notification"
            assert event2["type"] == "notification"
            assert event1["data"]["id"] == event2["data"]["id"] == 101


def test_websocket_rejects_missing_auth():
    app, hub = create_test_app()
    client = TestClient(app)
    try:
        with client.websocket_connect("/api/notifications/stream"):
            pass
    except WebSocketDisconnect as e:
        assert e.code == 4001


def test_websocket_rejects_bad_token():
    app, hub = create_test_app()
    client = TestClient(app)
    try:
        with client.websocket_connect(
            "/api/notifications/stream",
            headers={"Authorization": "Bearer invalid-token"},
        ):
            pass
    except WebSocketDisconnect as e:
        assert e.code == 4001
