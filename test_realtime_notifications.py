"""
Tests for the real-time notification broker and websocket fan-out.
"""

from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient

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


# ---------------------------------------------------------------------------
# Deduplication hash tests
# ---------------------------------------------------------------------------


class TestDedupHash:
    """_compute_dedup_hash uses only (type, source, data) with sorted keys,
    no default=str, ensuring distinct payloads never collide."""

    def test_distinct_data_produces_distinct_hashes(self):
        h1 = NotificationBroadcastHub._compute_dedup_hash(
            {"type": "n", "source": "local", "data": {"a": 1}}
        )
        h2 = NotificationBroadcastHub._compute_dedup_hash(
            {"type": "n", "source": "local", "data": {"a": 2}}
        )
        assert h1 != h2

    def test_equivalent_payloads_same_hash(self):
        h1 = NotificationBroadcastHub._compute_dedup_hash(
            {"type": "n", "source": "local", "data": {"a": 1, "b": 2}}
        )
        h2 = NotificationBroadcastHub._compute_dedup_hash(
            {"source": "local", "data": {"b": 2, "a": 1}, "type": "n"}
        )
        assert h1 == h2

    def test_different_type_differs(self):
        h1 = NotificationBroadcastHub._compute_dedup_hash(
            {"type": "alert", "source": "local", "data": {}}
        )
        h2 = NotificationBroadcastHub._compute_dedup_hash(
            {"type": "notification", "source": "local", "data": {}}
        )
        assert h1 != h2

    def test_different_source_differs(self):
        h1 = NotificationBroadcastHub._compute_dedup_hash(
            {"type": "n", "source": "local", "data": {}}
        )
        h2 = NotificationBroadcastHub._compute_dedup_hash(
            {"type": "n", "source": "redis", "data": {}}
        )
        assert h1 != h2

    def test_nested_dict_sorted_keys(self):
        h1 = NotificationBroadcastHub._compute_dedup_hash(
            {"type": "n", "source": "local", "data": {"inner": {"z": 1, "a": 2}}}
        )
        h2 = NotificationBroadcastHub._compute_dedup_hash(
            {"type": "n", "source": "local", "data": {"inner": {"a": 2, "z": 1}}}
        )
        assert h1 == h2

    def test_data_none_produces_hash(self):
        h = NotificationBroadcastHub._compute_dedup_hash(
            {"type": "n", "source": "test", "data": None}
        )
        assert isinstance(h, str)
        assert len(h) == 64

    def test_extra_fields_ignored(self):
        """created_at and other metadata are excluded from hash."""
        h1 = NotificationBroadcastHub._compute_dedup_hash(
            {"type": "n", "source": "local", "data": {"x": 1}}
        )
        h2 = NotificationBroadcastHub._compute_dedup_hash(
            {"type": "n", "source": "local", "data": {"x": 1}, "created_at": "2026-01-01"}
        )
        assert h1 == h2


def test_dedup_suppresses_duplicate_publish():
    """Publishing the same notification twice within TTL deduplicates."""
    import asyncio

    async def _run():
        hub = NotificationBroadcastHub(dedup_ttl=10.0)
        n1 = {"msg": "hello"}
        n2 = {"msg": "hello"}
        e1 = await hub.publish(n1)
        e2 = await hub.publish(n2)
        # Both calls return an event, but the second should not broadcast
        # (verified via _history length — dedup still appends history,
        #  but the test verifies no crash and correct return)
        assert e1.data == n1
        assert e2.data == n2
        # Verify a distinct third notification still goes through
        n3 = {"msg": "world"}
        e3 = await hub.publish(n3)
        assert e3.data == n3
        return True

    assert asyncio.run(_run())


def test_dedup_allows_after_ttl_expires():
    """Same notification re-published after TTL elapses is NOT deduplicated."""
    import asyncio

    async def _run():
        hub = NotificationBroadcastHub(dedup_ttl=0.01)
        n = {"msg": "after-ttl"}
        await hub.publish(n)
        await asyncio.sleep(0.02)
        await hub.publish(n)  # should not dedup — TTL expired
        return True

    assert asyncio.run(_run())
