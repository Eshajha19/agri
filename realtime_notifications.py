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

ALLOWED_TYPES = {
    "delivery_ack": DeliveryAck,
    "subscribe_crops": SubscribeCrops,
}

def validate_ws_payload(payload: dict):
    msg_type = payload.get("type")
    schema = ALLOWED_TYPES.get(msg_type)
    if not schema:
        raise ValueError(f"Unknown message type: {msg_type}")
    return schema(**payload)

async def _handle_inbound(self, ws, payload: dict):
    try:
        validated = validate_ws_payload(payload)
    except ValueError as e:
        # Reject unknown type with structured error or close
        await ws.send_json({"error": str(e)})
        await ws.close(code=1003)  # Unsupported data
        return
    except Exception as e:
        await ws.send_json({"error": "Invalid schema"})
        return

    # Now handle only validated types
    if validated.type == "delivery_ack":
        self._process_delivery_ack(validated)
    elif validated.type == "subscribe_crops":
        self._process_subscribe_crops(validated)


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
            if not self._is_duplicate(notification):
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
            if not self._is_duplicate(notification):
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
                try:
                    msg = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
                except asyncio.TimeoutError:
                    continue
                except WebSocketDisconnect:
                    break

                if msg is None:
                    continue

                frame_size = len(msg.encode("utf-8"))
                if frame_size > MAX_FRAME_SIZE:
                    logger.warning(
                        "Closing WS uid=%s — frame too large: %d bytes (max %d)",
                        uid, frame_size, MAX_FRAME_SIZE,
                    )
                    await websocket.close(code=1009)
                    break

                now = time.time()
                rate_timestamps.append(now)
                cutoff = now - RATE_WINDOW
                rate_timestamps = [t for t in rate_timestamps if t > cutoff]
                if len(rate_timestamps) > MAX_MESSAGES_PER_SEC:
                    logger.warning(
                        "Closing WS uid=%s — rate limit exceeded: %d msg/s (max %d)",
                        uid, len(rate_timestamps), MAX_MESSAGES_PER_SEC,
                    )
                    await websocket.close(code=1008)
                    break

                try:
                    parsed = json.loads(msg)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON from WS uid=%s, ignoring", uid)
                    continue

                if not isinstance(parsed, dict) or "type" not in parsed:
                    continue

                msg_type = parsed.get("type")

                if msg_type == "delivery_ack":
                    valid, error = self._validate_delivery_ack(parsed)
                    if not valid:
                        logger.warning(
                            "Invalid delivery_ack from WS uid=%s: %s", uid, error
                        )

                # ----------------------------------------------------------
                # FIX 2: subscribe_crops — validate AND apply to subscription
                #
                # Previously the message was validated but the subscription
                # was never updated, so crop filtering had no effect after
                # the initial connect.
                # ----------------------------------------------------------
=======

                # ----------------------------------------------------------
                # FIX 2: subscribe_crops — validate AND apply to subscription
                #
                # Previously the message was validated but the subscription
                # was never updated, so crop filtering had no effect after
                # the initial connect.
                # ----------------------------------------------------------
 main
                elif msg_type == "subscribe_crops":
                    valid, error = self._validate_subscribe_crops(parsed)
                    if not valid:
                        logger.warning(
                            "Invalid subscribe_crops from WS %s: %s", websocket, error,
                        )
                        continue
                    new_crops = frozenset(
                        c.strip().lower()
                        for c in parsed.get("crops", [])
                        if isinstance(c, str) and c.strip()
                    )
                    async with self._connections_lock:
                        sub = self._connections.get(websocket)
                        if sub is not None:
                            self._connections[websocket] = _ConnectionSubscription(
                                uid=sub.uid,
                                regions=sub.regions,
                                crops=new_crops,
                                retry_counts=sub.retry_counts,
                                last_ack_at=sub.last_ack_at,
                            )

                # ----------------------------------------------------------
                # FIX 3: subscribe_regions — new handler, mirrors HTTP logic
                #
                # The HTTP endpoint uses resolve_subscription_regions() +
                # profile authority checks.  The WS path previously had no
                # equivalent, meaning a connected client could never update
                # its region scope post-connect and the two paths were
                # inconsistent (issue requirement: they must behave the same).
                #
                # Security contract (matches HTTP):
                #   - Empty / malformed region tokens are rejected.
                #   - The caller can only subscribe to regions they own or
                #     that are covered by their profile authority.
                #   - Admins/experts may request any region.
                # ----------------------------------------------------------
                elif msg_type == "subscribe_regions":
                    valid, error = self._validate_subscribe_regions(parsed)
                    if not valid:
                        logger.warning(
                            "Invalid subscribe_regions from WS %s: %s", websocket, error,
                        )
                        continue
                    raw_regions = parsed.get("regions", [])
                    profile = self._get_profile_for_uid(uid)
                    new_scopes = frozenset(
                        resolve_subscription_regions(profile, raw_regions)
                    )
                    async with self._connections_lock:
                        sub = self._connections.get(websocket)
                        if sub is not None:
                            self._connections[websocket] = _ConnectionSubscription(
                                uid=sub.uid,
                                regions=new_scopes,
                                crops=sub.crops,
                                retry_counts=sub.retry_counts,
                                last_ack_at=sub.last_ack_at,
                            )
                else:
                        # Re-resolve through the same authority logic used at
                        # connect time so WS and HTTP are consistent.
                        from geo_alerts import profile_regions as _profile_regions
                        async with self._connections_lock:
                            sub = self._connections.get(websocket)

                        if sub is not None:
                            # Import here to avoid circular dependency at
                            # module level; profile lookup is cheap (cached
                            # in main.py's _get_firestore_user_profile).
                            profile = self._get_profile_fn(sub.uid) if self._get_profile_fn else {}
                            requested = parsed.get("regions", [])
                            new_regions = frozenset(
                                resolve_subscription_regions(profile, requested)
                            )
                            # Strip any empty tokens that normalization may
                            # have produced.
                            new_regions = frozenset(r for r in new_regions if r.strip())

                            async with self._connections_lock:
                                current = self._connections.get(websocket)
                                if current is not None:
                                    self._connections[websocket] = _ConnectionSubscription(
                                        uid=current.uid,
                                        regions=new_regions,
                                        crops=current.crops,
                                        retry_counts=current.retry_counts,
                                        last_ack_at=current.last_ack_at,
                                    )
                            logger.info(
                                "WS uid=%s updated region subscription: %s",
                                uid, new_regions,
                            )
                            await websocket.send_json(
                                {
                                    "type": "subscribed_regions",
                                    "regions": sorted(new_regions),
                                }
                            )

        except asyncio.CancelledError:
            pass
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


        stale_clients: list[WebSocket] = []
        async with self._broadcast_lock:
            for websocket, _subscription in clients:
                try:
                    await websocket.send_json(payload)
                except Exception:
                    stale_clients.append(websocket)
 main

        if stale_clients:
            async with self._connections_lock:
                for websocket in stale_clients:
                    self._connections.pop(websocket, None)

    async def _persist_notification(
        self, event: NotificationEvent, uid: str
    ) -> None:

    async def _persist_notification(
        self, event: NotificationEvent, uid: str
    ) -> None:
 main
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
            h for h, seen_at in self._recent_hashes.items()
            if now - seen_at > self._dedup_window
        ]
        for h in expired_hashes:
            self._recent_hashes.pop(h, None)

        content_hash = event.get_content_hash()
        if content_hash in self._recent_hashes:
            return True

        self._recent_hashes[content_hash] = now
        return False
    
#    Add new public method:
    def set_profile_fetcher(self, func: callable) -> None:
        """Inject the Firestore profile-lookup callable.
 
        Called once during lifespan startup to avoid a circular import between
        main.py and realtime_notifications.py.  The function signature must be:
            (uid: str) -> dict
        """
        self._get_profile = func
 
#    Add private helper method:
    def _get_profile_for_uid(self, uid: str) -> dict:
        """Fetch the Firestore profile for *uid*, returning {} on any failure."""
        if self._get_profile is None:
            return {}
        try:
            return self._get_profile(uid) or {}
        except Exception:
            return {}

    async def _route_to_priority_queue(self, event: NotificationEvent) -> None:
        """Place events into their priority queue for reliability bookkeeping."""
        if event.priority == NotificationPriority.CRITICAL:
            self._critical_queue.append(event)
        elif event.priority == NotificationPriority.WARNING:
            self._warning_queue.append(event)
        else:
            self._info_queue.append(event)

    async def _process_retry_queue(self) -> None:
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            raise

    async def _process_priority_queues(self) -> None:
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
# Enhanced realtime notifications
main
