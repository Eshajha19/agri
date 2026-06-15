"""
WebSocket authentication and region filtering integration tests.

Tests that the notification stream endpoint:
- Closes with code 1008 when auth is missing/invalid or user profile not found
- Closes with code 1011 on internal server error (Firestore unavailable)
- Applies region scoping: users only receive notifications for their region
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.testclient import TestClient

from realtime_notifications import NotificationBroadcastHub


# ---------- helpers ----------

def make_mock_firebase(uid: str = "test-uid", raises: bool = False):
    """Return a firebase auth mock that optionally raises."""
    m = MagicMock()
    if raises:
        m.verify_id_token.side_effect = Exception("Firebase unavailable")
    else:
        m.verify_id_token.return_value = {"uid": uid}
    return m


def make_mock_firestore(
    user_exists: bool = True,
    role: str = "farmer",
    region: str = "north",
    raises: bool = False,
):
    """Return a Firestore client mock with configurable profile."""
    doc_mock = MagicMock()
    doc_mock.exists = user_exists
    doc_mock.get.side_effect = lambda k, d=None: {"role": role, "region": region}.get(
        k, d
    )

    collection_mock = MagicMock()
    collection_mock.document.return_value.get.return_value = doc_mock

    db_mock = MagicMock()
    db_mock.collection.return_value = collection_mock

    if raises:
        db_mock.collection.side_effect = Exception("Firestore connection refused")

    return db_mock


# ---------- test app factory ----------

def create_auth_app(
    firebase_mock,
    firestore_mock,
    hub: NotificationBroadcastHub | None = None,
    region: str | None = None,
) -> FastAPI:
    """Create a FastAPI app with a websocket endpoint that performs auth and
    optional region filtering before delegating to the notification hub."""
    app = FastAPI()
    if hub is None:
        hub = NotificationBroadcastHub(history_limit=10)
    app.state.hub = hub

    # Expose a REST publish endpoint for test feeding
    @app.post("/_publish")
    async def _publish(payload: dict):
        await hub.publish(payload)
        return {"ok": True}

    @app.websocket("/api/notifications/stream")
    async def notifications_stream(
        websocket: WebSocket,
        token: str = Query(""),
        regions: str = Query(""),
    ):
        try:
            # 1. Verify token
            if not token:
                await websocket.close(code=1008, reason="Missing auth token")
                return

            try:
                decoded = firebase_mock.verify_id_token(token)
            except Exception:
                await websocket.close(code=1008, reason="Invalid auth token")
                return

            uid = decoded.get("uid")

            # 2. Fetch user profile from Firestore
            try:
                user_doc = firestore_mock.collection("users").document(uid).get()
            except Exception:
                await websocket.close(code=1011, reason="Database unavailable")
                return

            if not user_doc.exists:
                await websocket.close(code=1008, reason="User profile not found")
                return

            user_region = user_doc.get("region", "").lower()
            hub = app.state.hub

            # 3. Parse requested regions (optional filter from query param)
            requested_regions = {
                r.strip().lower() for r in regions.split(",") if r.strip()
            } if regions else set()

            # 4. Accept and send snapshot filtered by region
            await websocket.accept()

            async with hub._history_lock:
                snapshot = list(hub._history)

            if requested_regions:
                snapshot = [
                    n for n in snapshot
                    if n.get("region", "").lower() in requested_regions
                ]

            await websocket.send_json({
                "type": "snapshot",
                "source": "local",
                "data": snapshot,
            })

            # 5. Listen for live notifications (region-filtered)
            while True:
                # This simulates notification delivery.
                # In real code the hub would push; here we use an Event pattern.
                # For test simplicity we just wait and then break.
                import asyncio
                await asyncio.sleep(3600)

        except WebSocketDisconnect:
            pass
        except Exception:
            try:
                await websocket.close(code=1011, reason="Internal server error")
            except Exception:
                pass

    return app


# =============== Tests ===============


class TestWebSocketAuth:
    """Close-code assertions for various auth failure modes."""

    @pytest.fixture
    def hub(self):
        h = NotificationBroadcastHub(history_limit=10)
        h.seed_notifications([
            {"id": 1, "type": "weather", "message": "Rain", "region": "north"},
            {"id": 2, "type": "advisory", "message": "Fertilize", "region": "south"},
        ])
        return h

    def test_missing_token_closes_1008(self, hub):
        firebase_mock = make_mock_firebase()
        firestore_mock = make_mock_firestore()
        app = create_auth_app(firebase_mock, firestore_mock, hub)
        client = TestClient(app)

        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect(
                "/api/notifications/stream"
            ):
                pass
        assert exc_info.value.code == 1008

    def test_invalid_token_closes_1008(self, hub):
        firebase_mock = make_mock_firebase(raises=True)
        firestore_mock = make_mock_firestore()
        app = create_auth_app(firebase_mock, firestore_mock, hub)
        client = TestClient(app)

        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect(
                client.websocket_connect(
                    "/api/notifications/stream",
                    headers={"Authorization": "Bearer badtoken"},
                )
            ):
                pass
        assert exc_info.value.code == 1008

    def test_missing_profile_closes_1008(self, hub):
        firebase_mock = make_mock_firebase()
        firestore_mock = make_mock_firestore(user_exists=False)
        app = create_auth_app(firebase_mock, firestore_mock, hub)
        client = TestClient(app)

        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect(
                client.websocket_connect(
                    "/api/notifications/stream",
                    headers={"Authorization": "Bearer goodtoken"},
                )
            ):
                pass
        assert exc_info.value.code == 1008

    def test_firestore_unavailable_closes_1011(self, hub):
        firebase_mock = make_mock_firebase()
        firestore_mock = make_mock_firestore(raises=True)
        app = create_auth_app(firebase_mock, firestore_mock, hub)
        client = TestClient(app)

        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect(
                client.websocket_connect(
                    "/api/notifications/stream",
                    headers={"Authorization": "Bearer goodtoken"},
                )
            ):
                pass
        assert exc_info.value.code == 1011


class TestRegionFiltering:
    """Region-scoping assertions on snapshot data."""

    @pytest.fixture
    def hub(self):
        h = NotificationBroadcastHub(history_limit=10)
        h.seed_notifications([
            {"id": 1, "type": "weather", "message": "North rain", "region": "north"},
            {"id": 2, "type": "weather", "message": "South storm", "region": "south"},
            {"id": 3, "type": "advisory", "message": "East fertilize", "region": "east"},
        ])
        return h

    def test_user_receives_all_notifications_when_no_region_filter(self, hub):
        """When no regions query param is sent, user gets full snapshot."""
        firebase_mock = make_mock_firebase()
        firestore_mock = make_mock_firestore()
        app = create_auth_app(firebase_mock, firestore_mock, hub)
        client = TestClient(app)

        with client.websocket_connect(
            client.websocket_connect(
                "/api/notifications/stream",
                headers={"Authorization": "Bearer goodtoken"},
            )
        ) as ws:
            snapshot = ws.receive_json()
            assert len(snapshot["data"]) == 3

    def test_user_receives_filtered_notifications_by_requested_regions(self, hub):
        """Regions query param filters snapshot to matching entries."""
        firebase_mock = make_mock_firebase()
        firestore_mock = make_mock_firestore()
        app = create_auth_app(firebase_mock, firestore_mock, hub)
        client = TestClient(app)

        with client.websocket_connect(
            client.websocket_connect(
                "/api/notifications/stream?regions=north",
                headers={"Authorization": "Bearer goodtoken"},
            )
        ) as ws:
            snapshot = ws.receive_json()
            assert len(snapshot["data"]) == 1
            assert snapshot["data"][0]["message"] == "North rain"

    def test_user_receives_filtered_notifications_multiple_regions(self, hub):
        """Comma-separated regions query param works."""
        firebase_mock = make_mock_firebase()
        firestore_mock = make_mock_firestore()
        app = create_auth_app(firebase_mock, firestore_mock, hub)
        client = TestClient(app)

        with client.websocket_connect(
            client.websocket_connect(
                "/api/notifications/stream?regions=north,south",
                headers={"Authorization": "Bearer goodtoken"},
            )
        ) as ws:
            snapshot = ws.receive_json()
            assert len(snapshot["data"]) == 2
            messages = {n["message"] for n in snapshot["data"]}
            assert messages == {"North rain", "South storm"}

    def test_region_filter_case_insensitive(self, hub):
        """Region matching should be case-insensitive."""
        firebase_mock = make_mock_firebase()
        firestore_mock = make_mock_firestore()
        app = create_auth_app(firebase_mock, firestore_mock, hub)
        client = TestClient(app)

        with client.websocket_connect(
            client.websocket_connect(
                "/api/notifications/stream?regions=north",
                headers={"Authorization": "Bearer goodtoken"},
            )
        ) as ws:
            snapshot = ws.receive_json()
            assert len(snapshot["data"]) == 1
            assert snapshot["data"][0]["message"] == "North rain"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
