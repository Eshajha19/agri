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


def test_notification_history_eviction():
    """Verify that notification history respects maxlen and evicts oldest entries."""
    hub = NotificationBroadcastHub(history_limit=5)
    
    # Add 7 notifications to a hub with limit 5
    for i in range(7):
        hub.seed_notifications([{"id": i, "message": f"Notification {i}"}])
    
    # History should only contain the last 5
    history = hub.snapshot()
    assert len(history) == 5, f"Expected 5 entries, got {len(history)}"
    
    # Oldest entries (0, 1) should be evicted
    ids = [n["id"] for n in history]
    assert ids == [2, 3, 4, 5, 6], f"Expected [2,3,4,5,6], got {ids}"
    
    # Publish more notifications via publish() should also evict
    import asyncio
    
    async def publish_more():
        for i in range(7, 10):
            await hub.publish({"id": i, "message": f"Notification {i}"})
    
    asyncio.run(publish_more())
    
    history = hub.snapshot()
    assert len(history) == 5, f"Expected 5 entries after publish, got {len(history)}"
    ids = [n["id"] for n in history]
    assert ids == [5, 6, 7, 8, 9], f"Expected [5,6,7,8,9], got {ids}"


def test_redis_listener_eviction():
    """Verify that Redis listener also respects history limit."""
    hub = NotificationBroadcastHub(history_limit=3)
    
    # Simulate Redis listener adding notifications directly to history
    for i in range(5):
        import asyncio
        asyncio.run(hub._redis_listener_add({"id": i, "message": f"Redis {i}"}))
    
    history = hub.snapshot()
    assert len(history) == 3, f"Expected 3 entries, got {len(history)}"
    ids = [n["id"] for n in history]
    assert ids == [2, 3, 4], f"Expected [2,3,4], got {ids}"
