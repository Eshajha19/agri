"""
Real-time notification fan-out for WebSocket and optional pub-sub scaling.
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import os
import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Deque, Dict, Iterable, Optional, List

from fastapi import WebSocket, WebSocketDisconnect
from geo_alerts import notification_matches_regions, resolve_subscription_regions

from notification_auth import filter_notifications_for_user, notification_visible_to_user

logger = logging.getLogger(__name__)


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


class NotificationBroadcastHub:
    """Broadcasts notifications to connected WebSocket clients.

    The hub keeps a small in-memory history so new websocket clients receive an
    immediate snapshot. If REDIS_URL is configured and redis.asyncio is
    available, the hub also publishes to a Redis channel so multiple workers can
    fan out the same event across processes.

    Each WebSocket connection is bound to a Firebase UID; snapshots and live
    events are filtered so clients only receive notifications they are allowed
    to see (broadcast or targeted to their UID).
    """

    def __init__(
        self,
        history_limit: int = 200,
        redis_url: Optional[str] = None,
        redis_channel: str = "fasal_saathi.notifications",
        enable_persistence: bool = True,
        dedup_window_seconds: int = 300,
    ) -> None:
        self._history: Deque[Dict[str, Any]] = collections.deque(maxlen=history_limit)
        self._connections: dict[WebSocket, _ConnectionSubscription] = {}
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
        self._started = False

        # Persistence and delivery tracking
        self._enable_persistence = enable_persistence
        self._delivery_records: Dict[str, NotificationDeliveryRecord] = {}
        self._pending_notifications: Deque[NotificationEvent] = collections.deque()
        self._dead_letter_queue: Deque[NotificationDeliveryRecord] = collections.deque(maxlen=10000)
        self._retry_queue: List[tuple[float, NotificationDeliveryRecord]] = []
        self._persistence_lock = asyncio.Lock()

        # Deduplication
        self._dedup_window = dedup_window_seconds
        self._recent_hashes: Dict[str, float] = {}  # content_hash -> timestamp

        # Priority queues
        self._critical_queue: Deque[NotificationEvent] = collections.deque()
        self._warning_queue: Deque[NotificationEvent] = collections.deque()
        self._info_queue: Deque[NotificationEvent] = collections.deque()

    def seed_notifications(
        self,
        notifications: Iterable[Dict[str, Any]],
    ) -> None:
        """Seed the local history from existing notifications."""

        for notification in notifications:
            self._history.append(notification)

    async def snapshot(self) -> list[Dict[str, Any]]:
        """Return a copy of the current history."""

    def snapshot_for_user(self, uid: str, regions: Optional[Iterable[str]] = None) -> list[Dict[str, Any]]:
        """Return history entries visible to the given user and region scope."""
        return [
            notification
            for notification in filter_notifications_for_user(self._history, uid)
            if notification_matches_regions(notification, regions)
        ]

    def _is_duplicate_notification(self, notification: NotificationEvent) -> bool:
        """Check if notification is a duplicate within the dedup window"""
        content_hash = notification.get_content_hash()
        current_time = time.time()

        # Clean old hashes
        expired_hashes = [h for h, t in self._recent_hashes.items() if current_time - t > self._dedup_window]
        for h in expired_hashes:
            del self._recent_hashes[h]

        if content_hash in self._recent_hashes:
            logger.info(f"Notification {notification.notification_id} is duplicate (skipped)")
            return True

        self._recent_hashes[content_hash] = current_time
        return False

    async def _route_to_priority_queue(self, notification: NotificationEvent) -> None:
        """Route notification to appropriate priority queue"""
        if notification.priority == NotificationPriority.CRITICAL:
            self._critical_queue.append(notification)
        elif notification.priority == NotificationPriority.WARNING:
            self._warning_queue.append(notification)
        else:
            self._info_queue.append(notification)

    def _get_retry_delay(self, retry_count: int) -> float:
        """Calculate exponential backoff delay in seconds: 1, 2, 4, 8, 16"""
        base_delay = 1
        return min(base_delay * (2 ** retry_count), 16)

    async def _persist_notification(self, notification: NotificationEvent, user_id: str) -> None:
        """Persist notification for offline delivery"""
        if not self._enable_persistence:
            return

        async with self._persistence_lock:
            record = NotificationDeliveryRecord(
                notification_id=notification.notification_id,
                user_id=user_id,
                priority=notification.priority,
                status=DeliveryStatus.PENDING,
                created_at=notification.created_at,
            )
            self._delivery_records[notification.notification_id] = record
            self._pending_notifications.append(notification)
            logger.info(f"Persisted notification {notification.notification_id} for user {user_id}")

    async def _mark_as_failed(self, record: NotificationDeliveryRecord, error: str) -> None:
        """Move notification to dead letter queue on permanent failure"""
        async with self._persistence_lock:
            record.status = DeliveryStatus.FAILED
            record.error_message = error
            self._dead_letter_queue.append(record)
            if record.notification_id in self._delivery_records:
                del self._delivery_records[record.notification_id]
            logger.error(f"Notification {record.notification_id} failed: {error}")

    async def _schedule_retry(self, record: NotificationDeliveryRecord) -> None:
        """Schedule notification for retry with exponential backoff"""
        if record.retry_count >= record.max_retries:
            await self._mark_as_failed(record, f"Max retries ({record.max_retries}) exceeded")
            return

        record.retry_count += 1
        delay = self._get_retry_delay(record.retry_count)
        retry_time = time.time() + delay

        async with self._persistence_lock:
            import heapq
            heapq.heappush(self._retry_queue, (retry_time, record))
            record.last_retry_at = datetime.now().isoformat()
            logger.info(f"Scheduled retry for {record.notification_id} in {delay}s (attempt {record.retry_count})")

    async def _process_retry_queue(self) -> None:
        """Process notifications in retry queue when ready"""
        while True:
            async with self._persistence_lock:
                if not self._retry_queue:
                    await asyncio.sleep(1)
                    continue

                import heapq
                current_time = time.time()
                ready_records = []

                while self._retry_queue and self._retry_queue[0][0] <= current_time:
                    _, record = heapq.heappop(self._retry_queue)
                    ready_records.append(record)

            for record in ready_records:
                record.status = DeliveryStatus.PENDING
                async with self._persistence_lock:
                    self._delivery_records[record.notification_id] = record
                logger.info(f"Retrying notification {record.notification_id} (attempt {record.retry_count})")

            await asyncio.sleep(1)

    def get_delivery_status(self, notification_id: str) -> Optional[Dict]:
        """Get delivery status for a notification"""
        if notification_id in self._delivery_records:
            return self._delivery_records[notification_id].to_dict()
        return None

    def get_dead_letter_queue_stats(self) -> Dict[str, Any]:
        """Get statistics about failed notifications"""
        dlq_list = list(self._dead_letter_queue)
        return {
            "size": len(dlq_list),
            "critical_count": sum(1 for r in dlq_list if r.priority == NotificationPriority.CRITICAL),
            "warning_count": sum(1 for r in dlq_list if r.priority == NotificationPriority.WARNING),
            "info_count": sum(1 for r in dlq_list if r.priority == NotificationPriority.INFO),
            "oldest_notification": dlq_list[0].created_at if dlq_list else None,
        }

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
        """Persist notification locally and fan it out to subscribed clients."""
        event = NotificationEvent(type="notification", data=notification, source=source)
        payload = {
            "type": event.type,
            "source": event.source,
            "created_at": event.created_at,
            "data": event.data,
        }

        async with self._history_lock:
            self._history.append(notification)

        async with self._connections_lock:
            clients = [
                (websocket, subscription)
                for websocket, subscription in self._connections.items()
                if notification_visible_to_user(notification, subscription.uid)
                and notification_matches_regions(notification, subscription.regions)
            ]

        await self._broadcast(payload, clients)

        if self._redis_client is not None and source != "redis":
            try:
                await self._redis_client.publish(self._redis_channel, json.dumps(payload))
            except Exception as exc:
                logger.warning("Failed to publish notification to Redis: %s", exc)

        return event

    async def connect(self, websocket: WebSocket, uid: str, regions: Optional[Iterable[str]] = None) -> None:
        """Accept a websocket client and keep it subscribed until disconnect."""

        await websocket.accept()
        region_scopes = frozenset(resolve_subscription_regions({"role": "guest"}, regions))
        async with self._history_lock:
            self._connections[websocket] = _ConnectionSubscription(uid=uid, regions=region_scopes)
            snapshot = self.snapshot_for_user(uid, region_scopes)

        await websocket.send_json(
            {
                "type": "snapshot",
                "source": "local",
                "created_at": datetime.now().isoformat(),
                "data": snapshot,
            }
        )

        # Register only after snapshot delivery completes.
        async with self._connections_lock:
            self._connections[websocket] = _ConnectionSubscription(
                uid=uid,
                regions=region_scopes,
            )

        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            pass
        except WebSocketDisconnect:
            pass
        finally:
            async with self._connections_lock:
                self._connections.pop(websocket, None)
    
    async def _broadcast(
        self,
        payload: Dict[str, Any],
        clients: list[tuple[WebSocket, _ConnectionSubscription]],
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

        # Clean up stale connections outside broadcast_lock to avoid
        # lock-order inversion with connections_lock (acquired by publish
        # before calling _broadcast).  See publish().
        if stale_clients:
            async with self._connections_lock:
                for websocket in stale_clients:
                    self._connections.pop(websocket, None)

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

                    async with self._connections_lock:
                        clients = [
                            (websocket, subscription)
                            for websocket, subscription in self._connections.items()
                            if notification_visible_to_user(notification, subscription.uid)
                            and notification_matches_regions(notification, subscription.regions)
                        ]
                    await self._broadcast(payload, clients)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Notification pub-sub listener stopped: %s", exc)


notification_broker = NotificationBroadcastHub()
# Enhanced realtime notifications
