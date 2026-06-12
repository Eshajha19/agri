"""
Real-time notification fan-out for WebSocket and optional pub-sub scaling.
"""

from __future__ import annotations

import asyncio
import collections
import hashlib
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
    """

    DEDUP_FIELDS = ("type", "source", "data")

    def __init__(
        self,
        history_limit: int = 200,
        redis_url: Optional[str] = None,
        redis_channel: str = "fasal_saathi.notifications",
        dedup_ttl: float = 60.0,
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
        self._dedup_ttl = dedup_ttl
        self._seen_hashes: Dict[str, float] = {}

    @staticmethod
    def _compute_dedup_hash(payload: Dict[str, Any]) -> str:
        """Deterministic SHA-256 hash of the canonical dedup fields.

        Only *type*, *source*, and *data* (with sorted keys) are included.
        ``json.dumps`` is called *without* ``default=str`` so that non-JSON-safe
        values raise immediately — no silent coercion of different objects into
        the same string.
        """
        canonical: Dict[str, Any] = {}
        for key in NotificationBroadcastHub.DEDUP_FIELDS:
            val = payload.get(key)
            if isinstance(val, dict):
                val = dict(sorted(val.items()))
            canonical[key] = val
        raw = json.dumps(canonical, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

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

        # Deduplicate within the TTL window using deterministic hash
        h = self._compute_dedup_hash(payload)
        now = asyncio.get_event_loop().time()
        async with self._history_lock:
            last_seen = self._seen_hashes.get(h)
            if last_seen is not None and (now - last_seen) < self._dedup_ttl:
                logger.debug("Deduplicated notification (hash=%s)", h[:10])
                return event
            self._seen_hashes[h] = now
            self._history.append(notification)
            clients = list(self._connections)

        await self._broadcast(payload, clients)

        if self._redis_client is not None and source != "redis":
            try:
                await self._redis_client.publish(self._redis_channel, json.dumps(payload))
            except Exception as exc:
                logger.warning("Failed to publish notification to Redis: %s", exc)

        # Evict expired hashes periodically (every 32 publishes)
        if len(self._seen_hashes) > 64:
            cutoff = now - self._dedup_ttl
            self._seen_hashes = {k: v for k, v in self._seen_hashes.items() if v >= cutoff}

        return event

    async def connect(self, websocket: WebSocket) -> None:
        """Accept a websocket client and keep it subscribed until disconnect."""
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
                    # Dedup check for cross-process events
                    h = self._compute_dedup_hash(payload)
                    now = asyncio.get_event_loop().time()
                    async with self._history_lock:
                        last_seen = self._seen_hashes.get(h)
                        if last_seen is not None and (now - last_seen) < self._dedup_ttl:
                            continue
                        self._seen_hashes[h] = now
                        self._history.append(notification)
                        clients = list(self._connections)
                    await self._broadcast(payload, clients)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Notification pub-sub listener stopped: %s", exc)


notification_broker = NotificationBroadcastHub()
