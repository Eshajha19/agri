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
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Deque, Dict, Iterable, Optional

from fastapi import WebSocket, WebSocketDisconnect
from geo_alerts import (
    normalize_region_identifier,
    notification_matches_regions,
    profile_can_broadcast_region,
    resolve_subscription_regions,
)

from pydantic import BaseModel

class DeliveryAck(BaseModel):
    type: str
    notification_id: str

    class Config:
        extra = "forbid"

class SubscribeCrops(BaseModel):
    type: str
    crops: list[str]

    class Config:
        extra = "forbid"


from notification_auth import filter_notifications_for_user, notification_visible_to_user

# Max inbound WebSocket frame size (64 KB)
MAX_FRAME_SIZE = 64 * 1024
# Max inbound messages per second per connection
MAX_MESSAGES_PER_SEC = 10
RATE_WINDOW = 1.0

logger = logging.getLogger(__name__)
MAX_DELIVERY_RECORDS = 10000


class NotificationPriority(str, Enum):
    """Priority levels for notifications"""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class DeliveryStatus(str, Enum):
    """Delivery status for notifications"""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"


@dataclass
class NotificationDeliveryRecord:
    """Record of notification delivery attempt"""
    notification_id: str
    user_id: str
    priority: NotificationPriority
    status: DeliveryStatus
    created_at: str
    sent_at: Optional[str] = None
    delivered_at: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 5
    last_retry_at: Optional[str] = None
    error_message: Optional[str] = None
    user_device_info: Optional[Dict] = None
    user_ip: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "notification_id": self.notification_id,
            "user_id": self.user_id,
            "priority": self.priority.value,
            "status": self.status.value,
            "created_at": self.created_at,
            "sent_at": self.sent_at,
            "delivered_at": self.delivered_at,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "last_retry_at": self.last_retry_at,
            "error_message": self.error_message,
        }


@dataclass(slots=True)
class NotificationEvent:
    """Envelope for broadcast notifications."""

    type: str
    data: Dict[str, Any]
    source: str = "local"
    created_at: str = ""
    priority: NotificationPriority = NotificationPriority.INFO
    user_id: Optional[str] = None
    notification_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.notification_id:
            self.notification_id = f"{self.type}-{int(time.time() * 1000)}"

    def get_content_hash(self) -> str:
        """Generate SHA-256 hash of notification content for deduplication"""
        content = json.dumps(self.data, sort_keys=True)
        return hashlib.sha256(content.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class _ConnectionSubscription:
    uid: str
    regions: frozenset[str]
    crops: frozenset[str] = field(default_factory=frozenset)
    retry_counts: Dict[str, int] = field(default_factory=dict)
    last_ack_at: Optional[float] = None
    outbound_queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=100))


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

    DEDUP_FIELDS = ("type", "source", "data")

    def __init__(
        self,
        history_limit: int = 200,
        redis_url: Optional[str] = None,
        redis_channel: str = "fasal_saathi.notifications",
        dedup_ttl: float = 60.0,
    ) -> None:
        self._history: Deque[Dict[str, Any]] = collections.deque(maxlen=history_limit)
        self._seen_hashes: set[str] = set()
        self._connections: Dict[WebSocket, _ConnectionSubscription] = {}
        self._history_lock = asyncio.Lock()
        self._connections_lock = asyncio.Lock()
        self._broadcast_lock = asyncio.Lock()
        self._redis_url = redis_url or os.getenv("REDIS_URL")
        self._redis_channel = redis_channel
        self._redis_client = None
        self._redis_pubsub = None
        self._redis_listener_task: Optional[asyncio.Task] = None
        self._retry_processor_task: Optional[asyncio.Task] = None
        self._priority_processor_task: Optional[asyncio.Task] = None
        self._started = False
        self._authenticate = authenticate
        self._enable_persistence = False
        self._persistence_lock = asyncio.Lock()
        self._delivery_records: Dict[str, NotificationDeliveryRecord] = {}
        self._max_delivery_records = MAX_DELIVERY_RECORDS
        self._recent_hashes: Dict[str, float] = {}
        self._dedup_window = 60.0
        self._critical_queue: collections.deque = collections.deque(maxlen=1000)
        self._warning_queue: collections.deque = collections.deque(maxlen=1000)
        self._info_queue: collections.deque = collections.deque(maxlen=1000)
        self._get_profile: Optional[callable] = None

    def set_authenticate(self, func: callable) -> None:
        """Set the token verification callable (injected at runtime)."""
        self._authenticate = func

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
        """Seed the local history from existing notifications (deduplicated)."""
        for notification in notifications:
            if not self._is_duplicate_notification_by_content(notification):
                self._history.append(notification)

    async def snapshot(self) -> list[Dict[str, Any]]:
        """Return a copy of the current history."""
        async with self._history_lock:
            return list(self._history)

    def snapshot_for_user(
        self, uid: str, regions: Optional[Iterable[str]] = None
    ) -> list[Dict[str, Any]]:
        """Return history entries visible to the given user and region scope."""
        return [
            notification
            for notification in filter_notifications_for_user(self._history, uid)
            if notification_matches_regions(notification, regions)
        ]

    async def start(self) -> None:
        """Start optional Redis pub-sub listener and background reliability tasks."""
        if self._started:
            return
        self._started = True

        self._retry_processor_task = asyncio.create_task(self._process_retry_queue())
        self._priority_processor_task = asyncio.create_task(self._process_priority_queues())

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
        for task_name in (
            "_retry_processor_task",
            "_priority_processor_task",
            "_redis_listener_task",
        ):
            task = getattr(self, task_name, None)
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                setattr(self, task_name, None)

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

    async def publish_price_alert(
        self,
        notification: Dict[str, Any],
        source: str = "price_alerts",
        max_ws_retries: int = 3,
    ) -> NotificationEvent:
        """Publish price alert with WebSocket-first delivery and WhatsApp fallback."""
        event = NotificationEvent(
            type="price_alert",
            data=notification,
            source=source,
            priority=NotificationPriority.WARNING,
        )

        if self._is_duplicate_notification(event):
            logger.info("Duplicate price alert %s skipped", event.notification_id)
            return event

        await self._route_to_priority_queue(event)

        payload = {
            "type": event.type,
            "source": event.source,
            "created_at": event.created_at,
            "notification_id": event.notification_id,
            "data": event.data,
        }

        async with self._history_lock:
            self._history.append(payload)

        crop = notification.get("crop")
        region_id = notification.get("region_id")
        async with self._connections_lock:
            clients = []
            for websocket, subscription in self._connections.items():
                if not notification_visible_to_user(notification, subscription.uid):
                    continue
                if not notification_matches_regions(notification, subscription.regions):
                    continue
                if subscription.crops and crop and crop not in subscription.crops:
                    continue
                clients.append((websocket, subscription))

        ws_failed_uids = []
        delivered = False
        for websocket, subscription in clients:
            try:
                await websocket.send_json(payload)
                subscription.retry_counts[event.notification_id] = (
                    subscription.retry_counts.get(event.notification_id, 0) + 1
                )
                delivered = True
            except Exception:
                ws_failed_uids.append(subscription.uid)

        if not delivered or ws_failed_uids:
            for uid in ws_failed_uids:
                await self._persist_notification(event, uid)

        return event

    async def publish(
        self, notification: Dict[str, Any], source: str = "local"
    ) -> NotificationEvent:
        """Persist notification locally and fan it out to subscribed clients."""
        event = NotificationEvent(type="notification", data=notification, source=source)

        if self._is_duplicate_notification(event):
            logger.info("Duplicate notification %s skipped", event.notification_id)
            return event

        await self._route_to_priority_queue(event)

        uid = notification.get("recipient_uid")
        if uid:
            await self._persist_notification(event, uid)

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
            if not self._is_duplicate_notification_by_content(notification):
                self._history.append(notification)

        async with self._connections_lock:
            clients = [
                (ws, sub)
                for ws, sub in self._connections.items()
                if notification_visible_to_user(notification, sub.uid)
                and notification_matches_regions(notification, sub.regions)
            ]

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

    # ------------------------------------------------------------------
    # FIX 1: connect() now accepts uid and regions from the caller
    #         (main.py calls connect(websocket, uid, regions=...)).
    #         Previously the signature was connect(self, websocket) which
    #         left `uid` and `region_scopes` undefined inside the method,
    #         causing a NameError on every connection.
    # ------------------------------------------------------------------
    async def connect(
        self,
        websocket: WebSocket,
        uid: str,
        *,
        regions: frozenset[str] | None = None,
    ) -> None:
        """Accept a websocket client and keep it subscribed until disconnect.

        Parameters
        ----------
        websocket:
            The FastAPI WebSocket instance, not yet accepted.
        uid:
            Firebase UID of the already-authenticated caller.  Authentication
            is performed by the HTTP layer before this method is called.
        regions:
            The resolved subscription scopes returned by
            :func:`geo_alerts.resolve_subscription_regions`.  Must never
            contain empty strings — callers are responsible for filtering
            them before passing.  An empty frozenset means the user has no
            regional authority and will receive only global (unscoped)
            notifications.
        """
        await websocket.accept()

        # Defensive: strip any empty/whitespace tokens that slipped through.
        # An empty string in the region set would match every notification
        # via the `not subscription_region` short-circuit in region_matches().
        region_scopes: frozenset[str] = frozenset(r for r in regions if r.strip())


 main

 main
        async with self._connections_lock:
            self._connections[websocket] = _ConnectionSubscription(
                uid=uid,
                regions=region_scopes,
            )

        snapshot = self.snapshot_for_user(uid, region_scopes)
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
                # Keep reading to detect client-side disconnects
                _ = await websocket.receive_text()
        except (WebSocketDisconnect, asyncio.CancelledError):
            pass
        except Exception as exc:
            logger.warning("WebSocket read error: %s", exc)
        finally:
            async with self._connections_lock:
                self._connections.pop(websocket, None)

    # Optional profile-lookup hook injected by main.py so this module
    # does not import main directly (avoids circular imports).
    _get_profile_fn: Optional[callable] = None

    def set_profile_lookup(self, fn: callable) -> None:
        """Inject the Firestore profile-lookup function (called by main.py lifespan)."""
        self._get_profile_fn = fn

    @staticmethod
    def _validate_delivery_ack(msg: Dict[str, Any]) -> tuple[bool, str]:
        """Validate a ``delivery_ack`` message schema."""
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
        """Validate a ``subscribe_crops`` message schema."""
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

    @staticmethod
    def _validate_subscribe_regions(msg: Dict[str, Any]) -> tuple[bool, str]:
        """Validate a ``subscribe_regions`` message schema.
 
        Required keys: ``type``, ``regions`` (list of str, 1–50 items,
        each 1–100 chars).  No unknown keys allowed.
 
        Mirrors the structural validation applied to the HTTP ``region_id``
        field so WS and HTTP behave identically (issue #2370 AC-2).
        """
        allowed = {"type", "regions"}
        extra = msg.keys() - allowed
        if extra:
            return False, f"unknown keys: {', '.join(sorted(extra))}"
 
        regions = msg.get("regions")
        if not isinstance(regions, list):
            return False, "regions must be a list"
        if len(regions) == 0 or len(regions) > 50:
            return False, "regions must have 1–50 items"
        for i, r in enumerate(regions):
            if not isinstance(r, str) or len(r) == 0 or len(r) > 100:
                return False, f"regions[{i}] must be a non-empty string <= 100 chars"
 
        return True, ""

    async def _broadcast(
        self, payload: Dict[str, Any], clients: list[tuple[WebSocket, _ConnectionSubscription]]
    ) -> None:
        if not clients:
            return

        stale_clients: list[WebSocket] = []
        async with self._broadcast_lock:
            for websocket, _subscription in clients:
                try:
                    await websocket.send_json(payload)
                except Exception:
                    stale_clients.append(websocket)


    def _is_duplicate_notification_by_content(self, notification: Dict[str, Any]) -> bool:
        """Check if notification content is a recent duplicate based on hash."""
        # Use deterministic hash to detect duplicates
        payload = {
            k: v for k, v in notification.items() 
            if k in self.DEDUP_FIELDS
        }
        h = self._compute_dedup_hash(payload)
        
        now = time.time()
        if h in self._recent_hashes:
            # Check if within TTL
            if now - self._recent_hashes[h] < self._dedup_window:
                return True
        
        # Not a duplicate, record it
        self._recent_hashes[h] = now
        
        # Cleanup old entries periodically
        if len(self._recent_hashes) > 1000:
            cutoff = now - self._dedup_window
            self._recent_hashes = {
                k: v for k, v in self._recent_hashes.items()
                if v >= cutoff
            }
        
        return False

    async def _broadcast(
        self, payload: Dict[str, Any], clients: list[tuple[WebSocket, _ConnectionSubscription]]
    ) -> None:
        if not clients:
            return

        stale_clients: list[WebSocket] = []
        async with self._broadcast_lock:
            for websocket, _subscription in clients:
                try:
                    await websocket.send_json(payload)
                except Exception:
                    stale_clients.append(websocket)


        stale_clients: list[WebSocket] = []
        async with self._broadcast_lock:
            async def send_to_client(ws: WebSocket) -> Optional[WebSocket]:
                try:
                    await asyncio.wait_for(ws.send_json(payload), timeout=2.0)
                    return None
                except Exception:
                    return ws

            results = await asyncio.gather(*(send_to_client(ws) for ws in clients), return_exceptions=True)
            stale_clients = [res for res in results if isinstance(res, WebSocket)]

            if stale_clients:
                logger.warning("Removing %d stale/slow WebSocket clients", len(stale_clients))
                async with self._history_lock:
                    for ws in stale_clients:
                        self._connections.discard(ws)

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
                        if not self._is_duplicate(notification):
                            self._history.append(notification)
                    async with self._connections_lock:
                        clients = list(self._connections.items())
                    await self._broadcast(payload, clients)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Notification pub-sub listener stopped: %s", exc)
        import redis.asyncio as redis  # type: ignore
        delay = 1.0
        while True:
            try:
                async for message in self._redis_pubsub.listen():
                    delay = 1.0
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
                logger.warning(
                    "Redis pub-sub listener error, reconnecting in %.0fs: %s",
                    delay, exc,
                )
            # Reconnect with exponential backoff (capped at 60 s)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60.0)
            try:
                self._redis_client = redis.from_url(self._redis_url, decode_responses=True)
                self._redis_pubsub = self._redis_client.pubsub()
                await self._redis_pubsub.subscribe(self._redis_channel)
                logger.info("Redis pub-sub listener reconnected to %s", self._redis_channel)
            except Exception as exc:
                logger.warning("Redis pub-sub reconnect failed: %s", exc)

    async def _redis_listener_add(self, notification: Dict[str, Any]) -> None:
        """Test helper: add notification via Redis listener path."""
        async with self._history_lock:
            self._history.append(notification)


notification_broker = NotificationBroadcastHub()

main
 main
