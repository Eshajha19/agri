"""
Tests for the real-time notification broker and websocket fan-out.
"""

import asyncio

from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient

from realtime_notifications import NotificationBroadcastHub, NotificationEvent


def create_test_app():
    app = FastAPI()
    hub = NotificationBroadcastHub(history_limit=10)

    @app.websocket("/api/notifications/stream")
    async def notifications_stream(websocket: WebSocket):
        uid = websocket.query_params.get("uid", "test-user")
        await hub.connect(websocket, uid)

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

    with client.websocket_connect("/api/notifications/stream?uid=test-user") as websocket:
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

    with client.websocket_connect("/api/notifications/stream?uid=user-1") as ws1:
        snapshot1 = ws1.receive_json()
        assert snapshot1["type"] == "snapshot"

        with client.websocket_connect("/api/notifications/stream?uid=user-2") as ws2:
            snapshot2 = ws2.receive_json()
            assert snapshot2["type"] == "snapshot"

            response = client.post("/api/notifications/test-publish")
            assert response.status_code == 200

            event1 = ws1.receive_json()
            event2 = ws2.receive_json()

            assert event1["type"] == "notification"
            assert event2["type"] == "notification"
            assert event1["data"]["id"] == event2["data"]["id"] == 101


def test_targeted_notification_only_reaches_intended_client():
    hub = NotificationBroadcastHub(history_limit=10)
    from notification_auth import notification_visible_to_user

    notification = {
        "id": 55,
        "type": "private",
        "message": "Only for alice",
        "recipient_uid": "alice",
    }
    assert notification_visible_to_user(notification, "alice")
    assert not notification_visible_to_user(notification, "bob")


def test_delivery_records_evict_oldest_record_at_capacity():
    async def _run():
        hub = NotificationBroadcastHub(history_limit=10, max_delivery_records=2)

        for notification_id in ("notification-1", "notification-2", "notification-3"):
            await hub._persist_notification(
                NotificationEvent(
                    type="notification",
                    data={"message": notification_id},
                    notification_id=notification_id,
                ),
                uid="alice",
            )

        assert list(hub._delivery_records) == ["notification-2", "notification-3"]
        assert len(hub._delivery_records) == 2

    asyncio.run(_run())


def test_delivery_records_respect_persistence_disabled():
    async def _run():
        hub = NotificationBroadcastHub(history_limit=10, enable_persistence=False, max_delivery_records=2)

        await hub._persist_notification(
            NotificationEvent(
                type="notification",
                data={"message": "private alert"},
                notification_id="notification-1",
            ),
            uid="alice",
        )

        assert not hub._delivery_records

    asyncio.run(_run())

