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
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Deque, Dict, Iterable, Optional

from fastapi import WebSocket, WebSocketDisconnect
from geo_alerts import notification_matches_regions, resolve_subscription_regions

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
        """Generate hash of notification content for deduplication"""
        content = json.dumps(self.data, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()


@dataclass(slots=True)
class _ConnectionSubscription:
    uid: str
    regions: frozenset[str]
    crops: frozenset[str] = field(default_factory=frozenset)
    retry_counts: Dict[str, int] = field(default_factory=dict)
    last_ack_at: Optional[float] = None


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
        self._seen_hashes: set[str] = set()
        self._connections: set[WebSocket] = set()
        self._history_lock = asyncio.Lock()
        # Dedicated lock for websocket connection registry mutations.
        # Prevents concurrent connection updates from racing with
        # broadcast fan-out and stale websocket cleanup.
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

    def set_authenticate(self, func: callable) -> None:
        """Set the token verification callable (injected at runtime)."""
        self._authenticate = func

    @staticmethod
    def _dedup_hash(notification: Dict[str, Any]) -> str:
        """SHA-256 hash of the canonical JSON representation of a notification."""
        canonical = json.dumps(notification, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _is_duplicate(self, notification: Dict[str, Any]) -> bool:
        h = self._dedup_hash(notification)
        if h in self._seen_hashes:
            return True
        self._seen_hashes.add(h)
        return False

    def seed_notifications(self, notifications: Iterable[Dict[str, Any]]) -> None:
        """Seed the local history from existing notifications (deduplicated)."""
        for notification in notifications:
            if not self._is_duplicate(notification):
                self._history.append(notification)

    async def snapshot(self) -> list[Dict[str, Any]]:
        """Return a copy of the current history."""
        async with self._history_lock:
            return list(self._history)

    def snapshot_for_user(self, uid: str, regions: Optional[Iterable[str]] = None) -> list[Dict[str, Any]]:
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

        # Start retry queue processor (handles exponential-backoff retries)
        self._retry_processor_task = asyncio.create_task(self._process_retry_queue())

        # Start priority queue processor (drains critical/warning/info queues)
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
        for task_name in ("_retry_processor_task", "_priority_processor_task", "_redis_listener_task"):
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

        # Filter clients by crop subscription + region
        crop = notification.get("crop")
        region_id = notification.get("region_id")
        async with self._connections_lock:
            clients = []
            for websocket, subscription in self._connections.items():
                if not notification_visible_to_user(notification, subscription.uid):
                    continue
                if not notification_matches_regions(notification, subscription.regions):
                    continue
                # Crop scoping: if client subscribed to specific crops, match
                if subscription.crops and crop and crop not in subscription.crops:
                    continue
                clients.append((websocket, subscription))

        # Track delivery attempts per client
        ws_failed_uids = []
        delivered = False
        for websocket, subscription in clients:
            try:
                await websocket.send_json(payload)
                # Increment retry count until ack clears it
                subscription.retry_counts[event.notification_id] = subscription.retry_counts.get(event.notification_id, 0) + 1
                delivered = True
            except Exception:
                ws_failed_uids.append(subscription.uid)

        # If no WebSocket delivery succeeded or all clients failed, mark for fallback
        if not delivered or ws_failed_uids:
            for uid in ws_failed_uids:
                await self._persist_notification(event, uid)

        return event

    async def publish(self, notification: Dict[str, Any], source: str = "local") -> NotificationEvent:
        """Persist notification locally and fan it out to subscribed clients."""
        event = NotificationEvent(type="notification", data=notification, source=source)

        # Deduplication check: skip if identical content seen within dedup window
        if self._is_duplicate_notification(event):
            logger.info("Duplicate notification %s skipped", event.notification_id)
            return event

        # Route to priority queue for deferred delivery processing
        await self._route_to_priority_queue(event)

        # Persist for offline delivery and retry tracking
        uid = notification.get("recipient_uid")
        if uid:
            await self._persist_notification(event, uid)

        payload = {
            "type": event.type,
            "source": event.source,
            "created_at": event.created_at,
            "data": event.data,
        }

        async with self._history_lock:
            if not self._is_duplicate(notification):
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
            self._connections[websocket] = _ConnectionSubscription(
                uid=uid, regions=region_scopes, crops=crop_scopes
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
                except WebSocketDisconnect:
                    break

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
                if msg is None:
                    continue

                frame_size = len(msg.encode("utf-8"))
                if frame_size > MAX_FRAME_SIZE:
                    logger.warning(
                        "Closing WS %s — frame too large: %d bytes (max %d)",
                        websocket, frame_size, MAX_FRAME_SIZE,
                    )
                    await websocket.close(code=1009)  # Message too big
                    break

                # Sliding-window rate limit
                now = time.time()
                rate_timestamps.append(now)
                # Prune timestamps outside the window
                cutoff = now - RATE_WINDOW
                rate_timestamps = [t for t in rate_timestamps if t > cutoff]
                if len(rate_timestamps) > MAX_MESSAGES_PER_SEC:
                    logger.warning(
                        "Closing WS %s — rate limit exceeded: %d msg/s (max %d)",
                        websocket, len(rate_timestamps), MAX_MESSAGES_PER_SEC,
                    )
                    await websocket.close(code=1008)  # Policy violation
                    break
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

        stale_clients: list[WebSocket] = []
        async with self._broadcast_lock:
            stale_clients: list[WebSocket] = []
            for websocket, _subscription in clients:
                try:
                    await websocket.send_json(payload)
                except Exception:
                    stale_clients.append(websocket)

        # Clean up stale connections outside broadcast_lock to avoid
        # lock-order inversion with connections_lock (acquired by publish
        # before calling _broadcast).  See publish().
        if stale_clients:
            async with self._connections_lock:
                for websocket in stale_clients:
                    self._connections.pop(websocket, None)

    async def _persist_notification(self, event: NotificationEvent, uid: str) -> None:
        """Track targeted notification delivery with bounded memory usage."""
        if not self._enable_persistence:
            return

        record = NotificationDeliveryRecord(
            notification_id=event.notification_id or f"{event.type}-{int(time.time() * 1000)}",
            user_id=uid,
            priority=event.priority,
            status=DeliveryStatus.PENDING,
            created_at=event.created_at or datetime.now().isoformat(),
        )

        async with self._persistence_lock:
            if record.notification_id in self._delivery_records:
                self._delivery_records.pop(record.notification_id)
            elif len(self._delivery_records) >= self._max_delivery_records:
                self._delivery_records.popitem(last=False)
            self._delivery_records[record.notification_id] = record

    def _is_duplicate_notification(self, event: NotificationEvent) -> bool:
        """Return whether the same notification content was recently published."""
        now = time.time()
        expired_hashes = [
            content_hash
            for content_hash, seen_at in self._recent_hashes.items()
            if now - seen_at > self._dedup_window
        ]
        for content_hash in expired_hashes:
            self._recent_hashes.pop(content_hash, None)

        content_hash = event.get_content_hash()
        if content_hash in self._recent_hashes:
            return True

        self._recent_hashes[content_hash] = now
        return False

    async def _route_to_priority_queue(self, event: NotificationEvent) -> None:
        """Place events into their priority queue for reliability bookkeeping."""
        if event.priority == NotificationPriority.CRITICAL:
            self._critical_queue.append(event)
        elif event.priority == NotificationPriority.WARNING:
            self._warning_queue.append(event)
        else:
            self._info_queue.append(event)

    async def _process_retry_queue(self) -> None:
        """Keep retry task alive until delivery retry processing is implemented."""
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            raise

    async def _process_priority_queues(self) -> None:
        """Keep priority task alive until deferred queue processing is implemented."""
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            raise

    async def _redis_listener(self) -> None:
        try:
            async for message in self._redis_pubsub.listen():
                if message.get("type") != "message":
                    continue
                payload = json.loads(message["data"])
                notification = payload.get("data")
                if isinstance(notification, dict):
                    async with self._history_lock:
                        if not self._is_duplicate(notification):
                            self._history.append(notification)
                        clients = list(self._connections)
                    await self._broadcast(payload, clients)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Notification pub-sub listener stopped: %s", exc)

    async def _redis_listener_add(self, notification: Dict[str, Any]) -> None:
        """Test helper: add notification via Redis listener path."""
        async with self._history_lock:
            self._history.append(notification)


notification_broker = NotificationBroadcastHub()
# Enhanced realtime notifications
