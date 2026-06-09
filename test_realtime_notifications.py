"""
Tests for the real-time notification broker and websocket fan-out.
"""

from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient

import json

import pytest

from realtime_notifications import NotificationBroadcastHub


def create_test_app():
    app = FastAPI()
    hub = NotificationBroadcastHub(history_limit=10)

    @app.websocket("/api/notifications/stream")
    async def notifications_stream(websocket: WebSocket):
        await hub.connect(websocket)

    @app.post("/api/notifications/test-publish")
    async def publish_notification():
        await hub.publish(
            {
                "id": 101,
                "type": "weather",
                "message": "Heavy rainfall expected in your region today.",
                "time": "2026-05-20T10:00:00",
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
            }
        ]
    )
    client = TestClient(app)

    with client.websocket_connect("/api/notifications/stream") as websocket:
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

    with client.websocket_connect("/api/notifications/stream") as ws1:
        snapshot1 = ws1.receive_json()
        assert snapshot1["type"] == "snapshot"

        with client.websocket_connect("/api/notifications/stream") as ws2:
            snapshot2 = ws2.receive_json()
            assert snapshot2["type"] == "snapshot"

            response = client.post("/api/notifications/test-publish")
            assert response.status_code == 200

            event1 = ws1.receive_json()
            event2 = ws2.receive_json()

            assert event1["type"] == "notification"
            assert event2["type"] == "notification"
            assert event1["data"]["id"] == event2["data"]["id"] == 101


def test_oversized_frame_closes_connection():
    app, hub = create_test_app()
    client = TestClient(app)

    with client.websocket_connect("/api/notifications/stream") as websocket:
        snapshot = websocket.receive_json()
        assert snapshot["type"] == "snapshot"

        large_text = "x" * (64 * 1024 + 1)
        websocket.send_text(large_text)

        with pytest.raises(Exception):
            websocket.receive_json()


def test_message_rate_limit_closes_connection():
    app, hub = create_test_app()
    client = TestClient(app)

    with client.websocket_connect("/api/notifications/stream") as websocket:
        snapshot = websocket.receive_json()
        assert snapshot["type"] == "snapshot"

        # Send enough messages to trigger the 10 msg/s limit
        for _ in range(12):
            websocket.send_text("ping")

        with pytest.raises(Exception):
            websocket.receive_json()
