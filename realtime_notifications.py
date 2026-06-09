"""
Real-time notification fan-out for WebSocket and optional pub-sub scaling.
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Deque, Dict, Iterable, Optional

from fastapi import WebSocket, WebSocketDisconnect

# Max inbound WebSocket frame size (64 KB)
MAX_FRAME_SIZE = 64 * 1024
# Max inbound messages per second per connection
MAX_MESSAGES_PER_SEC = 10
RATE_WINDOW = 1.0

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class NotificationEvent:
    """Envelope for broadcast notifications."""

    type: str
    data: Dict[str, Any]
    source: str = "local"
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


class NotificationBroadcastHub:
    """Broadcasts notifications to connected WebSocket clients.

    The hub keeps a small in-memory history so new websocket clients receive an
    immediate snapshot. If REDIS_URL is configured and redis.asyncio is
    available, the hub also publishes to a Redis channel so multiple workers can
    fan out the same event across processes.
    """

    def __init__(
        self,
        history_limit: int = 200,
        redis_url: Optional[str] = None,
        redis_channel: str = "fasal_saathi.notifications",
    ) -> None:
        self._history: Deque[Dict[str, Any]] = collections.deque(maxlen=history_limit)
        self._connections: set[WebSocket] = set()
        self._history_lock = asyncio.Lock()
        self._broadcast_lock = asyncio.Lock()
        self._redis_url = redis_url or os.getenv("REDIS_URL")
        self._redis_channel = redis_channel
        self._redis_client = None
        self._redis_pubsub = None
        self._redis_listener_task: Optional[asyncio.Task] = None
        self._started = False

    def seed_notifications(self, notifications: Iterable[Dict[str, Any]]) -> None:
        """Seed the local history from existing notifications."""
        for notification in notifications:
            self._history.append(notification)

    def snapshot(self) -> list[Dict[str, Any]]:
        """Return a copy of the current history."""
        return list(self._history)

    async def start(self) -> None:
        """Start optional Redis pub-sub listener."""
        if self._started:
            return
        self._started = True

        if not self._redis_url:
            return

        try:
            import redis.asyncio as redis  # type: ignore

            self._redis_client = redis.from_url(self._redis_url, decode_responses=True)
            self._redis_pubsub = self._redis_client.pubsub()
            await self._redis_pubsub.subscribe(self._redis_channel)
            self._redis_listener_task = asyncio.create_task(self._redis_listener())
            logger.info("Notification pub-sub listener started on %s", self._redis_channel)
        except Exception as exc:
            logger.warning("Notification pub-sub disabled: %s", exc)
            self._redis_client = None
            self._redis_pubsub = None
            self._redis_listener_task = None

    async def stop(self) -> None:
        """Stop optional Redis listener and close resources."""
        if self._redis_listener_task is not None:
            self._redis_listener_task.cancel()
            try:
                await self._redis_listener_task
            except asyncio.CancelledError:
                pass
            self._redis_listener_task = None

        if self._redis_pubsub is not None:
            try:
                await self._redis_pubsub.close()
            except Exception:
                pass
            self._redis_pubsub = None

        if self._redis_client is not None:
            try:
                await self._redis_client.close()
            except Exception:
                pass
            self._redis_client = None

        self._started = False

    async def publish(self, notification: Dict[str, Any], source: str = "local") -> NotificationEvent:
        """Persist notification locally and fan it out to connected clients."""
        event = NotificationEvent(type="notification", data=notification, source=source)
        payload = {
            "type": event.type,
            "source": event.source,
            "created_at": event.created_at,
            "data": event.data,
        }

        async with self._history_lock:
            self._history.append(notification)
            clients = list(self._connections)

        await self._broadcast(payload, clients)

        if self._redis_client is not None and source != "redis":
            try:
                await self._redis_client.publish(self._redis_channel, json.dumps(payload))
            except Exception as exc:
                logger.warning("Failed to publish notification to Redis: %s", exc)

        return event

    async def connect(self, websocket: WebSocket) -> None:
        """Accept a websocket client and keep it subscribed until disconnect.

        Enforces a maximum inbound frame size (``MAX_FRAME_SIZE``) and a
        per-connection message rate limit (``MAX_MESSAGES_PER_SEC``).
        Oversized or high-rate connections are closed with an appropriate
        close code.
        """
        await websocket.accept()
        async with self._history_lock:
            self._connections.add(websocket)
            snapshot = list(self._history)

        await websocket.send_json(
            {
                "type": "snapshot",
                "source": "local",
                "created_at": datetime.now().isoformat(),
                "data": snapshot,
            }
        )

        rate_timestamps: list[float] = []

        try:
            while True:
                try:
                    msg = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
                except asyncio.TimeoutError:
                    # No message received — periodic liveness check
                    continue
                except WebSocketDisconnect:
                    break

                if msg is None:
                    continue

                frame_size = len(msg.encode("utf-8"))
                if frame_size > MAX_FRAME_SIZE:
                    logger.warning(
                        "Closing WS %s — frame too large: %d bytes (max %d)",
                        websocket, frame_size, MAX_FRAME_SIZE,
                    )
                    await websocket.close(code=1009)
                    break

                # Sliding-window rate limit
                now = time.time()
                rate_timestamps.append(now)
                cutoff = now - RATE_WINDOW
                rate_timestamps = [t for t in rate_timestamps if t > cutoff]
                if len(rate_timestamps) > MAX_MESSAGES_PER_SEC:
                    logger.warning(
                        "Closing WS %s — rate limit exceeded: %d msg/s (max %d)",
                        websocket, len(rate_timestamps), MAX_MESSAGES_PER_SEC,
                    )
                    await websocket.close(code=1008)
                    break

                # Parse and validate inbound JSON
                try:
                    parsed = json.loads(msg)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON from WS %s, ignoring", websocket)
                    continue

                if not isinstance(parsed, dict) or "type" not in parsed:
                    continue

                msg_type = parsed.get("type")
                if msg_type == "delivery_ack":
                    valid, error = self._validate_delivery_ack(parsed)
                    if not valid:
                        logger.warning(
                            "Invalid delivery_ack from WS %s: %s", websocket, error,
                        )
                elif msg_type == "subscribe_crops":
                    valid, error = self._validate_subscribe_crops(parsed)
                    if not valid:
                        logger.warning(
                            "Invalid subscribe_crops from WS %s: %s", websocket, error,
                        )
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
        finally:
            async with self._history_lock:
                self._connections.discard(websocket)

    @staticmethod
    def _validate_delivery_ack(msg: Dict[str, Any]) -> tuple[bool, str]:
        """Validate a ``delivery_ack`` message schema.

        Required keys: ``type``, ``notification_id`` (str), ``crops`` (list of str).
        No unknown keys allowed. Length caps enforced.
        """
        allowed = {"type", "notification_id", "crops"}
        extra = msg.keys() - allowed
        if extra:
            return False, f"unknown keys: {', '.join(sorted(extra))}"

        nid = msg.get("notification_id")
        if not isinstance(nid, str) or len(nid) == 0 or len(nid) > 100:
            return False, "notification_id must be a non-empty string <= 100 chars"

        crops = msg.get("crops")
        if not isinstance(crops, list):
            return False, "crops must be a list"
        if len(crops) == 0 or len(crops) > 50:
            return False, "crops must have 1–50 items"
        for i, c in enumerate(crops):
            if not isinstance(c, str) or len(c) == 0 or len(c) > 100:
                return False, f"crops[{i}] must be a non-empty string <= 100 chars"

        return True, ""

    @staticmethod
    def _validate_subscribe_crops(msg: Dict[str, Any]) -> tuple[bool, str]:
        """Validate a ``subscribe_crops`` message schema.

        Required keys: ``type``, ``crops`` (list of str).
        No unknown keys allowed. Length caps enforced.
        """
        allowed = {"type", "crops"}
        extra = msg.keys() - allowed
        if extra:
            return False, f"unknown keys: {', '.join(sorted(extra))}"

        crops = msg.get("crops")
        if not isinstance(crops, list):
            return False, "crops must be a list"
        if len(crops) == 0 or len(crops) > 100:
            return False, "crops must have 1–100 items"
        for i, c in enumerate(crops):
            if not isinstance(c, str) or len(c) == 0 or len(c) > 100:
                return False, f"crops[{i}] must be a non-empty string <= 100 chars"

        return True, ""

    async def _broadcast(self, payload: Dict[str, Any], clients: list[WebSocket]) -> None:
        if not clients:
            return

        async with self._broadcast_lock:
            stale_clients: list[WebSocket] = []
            for websocket in clients:
                try:
                    await websocket.send_json(payload)
                except Exception:
                    stale_clients.append(websocket)

            if stale_clients:
                async with self._history_lock:
                    for websocket in stale_clients:
                        self._connections.discard(websocket)

    async def _redis_listener(self) -> None:
        try:
            async for message in self._redis_pubsub.listen():
                if message.get("type") != "message":
                    continue
                payload = json.loads(message["data"])
                notification = payload.get("data")
                if isinstance(notification, dict):
                    async with self._history_lock:
                        self._history.append(notification)
                        clients = list(self._connections)
                    await self._broadcast(payload, clients)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Notification pub-sub listener stopped: %s", exc)


notification_broker = NotificationBroadcastHub()
