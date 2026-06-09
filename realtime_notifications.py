"""
Real-time notification fan-out for WebSocket and optional pub-sub scaling.
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Deque, Dict, Iterable, Optional

from fastapi import WebSocket, WebSocketDisconnect

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

    Parameters
    ----------
    authenticate : callable, optional
        Async callable ``(token: str) -> dict`` that returns decoded token data
        on success or raises on failure.  When provided, ``connect()`` requires
        the client to send a JSON message ``{"token": "..."}`` as the first
        frame; if validation fails the socket is closed with code 1008.
        The token is never exposed in URLs, query strings, or proxy logs.
    """

    def __init__(
        self,
        history_limit: int = 200,
        redis_url: Optional[str] = None,
        redis_channel: str = "fasal_saathi.notifications",
        authenticate: Optional[callable] = None,
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
        self._authenticate = authenticate

    def set_authenticate(self, func: callable) -> None:
        """Set the token verification callable (injected at runtime)."""
        self._authenticate = func

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

        If an ``authenticate`` callable was provided, the client **must** send
        a JSON message ``{"token": "..."}`` as the very first frame.
        Authentication via the first message keeps the token out of URLs,
        query strings, and proxy/analytics logs.
        """
        await websocket.accept()

        if self._authenticate is not None:
            try:
                msg = await asyncio.wait_for(websocket.receive_json(), timeout=10)
                token = msg.get("token", "") if isinstance(msg, dict) else ""
                if not token:
                    raise ValueError("Missing token")
                self._authenticate(token)
            except Exception:
                try:
                    await websocket.send_json({"type": "error", "message": "Authentication failed"})
                    await websocket.close(code=1008)
                except Exception:
                    pass
                return

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

        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            pass
        except WebSocketDisconnect:
            pass
        finally:
            async with self._history_lock:
                self._connections.discard(websocket)

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
