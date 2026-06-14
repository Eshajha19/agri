"""
Helpers for authenticated notification access and per-user scoping.
"""
import asyncio
from collections import deque
from fastapi import WebSocket
from backend.models import NotificationEvent

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from backend.notification_auth import notification_visible_to_user, filter_notifications_for_user

class NotificationBroadcastHub:
    def __init__(self, enable_persistence: bool = False, max_history: int = 100):
        self._enable_persistence = enable_persistence
        self._clients: Dict[str, WebSocket] = {}
        self._stale_clients: set[str] = set()
        self._history: deque[NotificationEvent] = deque(maxlen=max_history)
        self._outbound_queues: Dict[str, asyncio.Queue] = {}
        self._tasks: List[asyncio.Task] = []

    async def _broadcast(self, event: NotificationEvent):
        payload = event.serialize()
        for uid, ws in list(self._clients.items()):
            if not notification_visible_to_user(payload, uid):
                continue
            try:
                await ws.send_json(payload)
            except Exception:
                self._stale_clients.add(uid)

    async def replay_history(self, uid: str, ws: WebSocket):
        visible = filter_notifications_for_user([e.serialize() for e in self._history], uid)
        for notif in visible:
            await ws.send_json(notif)


async def _broadcast(self, event):
    payload = event.serialize()
    for uid, ws in list(self._clients.items()):
        if not notification_visible_to_user(payload, uid):
            continue
        try:
            await ws.send_json(payload)
        except Exception:
            self._stale_clients.add(uid)

class NotificationBroadcastHub:
    def __init__(self, enable_persistence: bool = False, max_history: int = 100):
        self._enable_persistence = enable_persistence
        self._clients: Dict[str, WebSocket] = {}
        self._stale_clients: set[str] = set()
        self._history: deque[NotificationEvent] = deque(maxlen=max_history)
        self._outbound_queues: Dict[str, asyncio.Queue] = {}
        self._tasks: List[asyncio.Task] = []

def notification_visible_to_user(
    notification: Dict[str, Any],
    uid: str,
) -> bool:
    """
    Return True when a notification is visible to the given user.

    Broadcast notifications omit ``recipient_uid`` (or set it to None) and are
    visible to every authenticated user. Targeted notifications include
    ``recipient_uid`` and are only visible to that user.

    Returns False immediately when ``uid`` is None or empty to prevent
    unauthenticated or malformed callers from accessing any notification.
    """
    if not uid:
        return False
    recipient = notification.get("recipient_uid")
    if recipient is None:
        return True
    return recipient == uid


def filter_notifications_for_user(
    notifications: Iterable[Dict[str, Any]],
    uid: str,
) -> List[Dict[str, Any]]:
    """Return notifications visible to ``uid``, preserving order."""
    if not uid:
        return []
    return [n for n in notifications if notification_visible_to_user(n, uid)]