"""
NotificationBroadcastHub: manages broadcast, history replay, and ack/retry lifecycle.
"""

import asyncio
from collections import deque
from fastapi import WebSocket
from typing import Any, Dict, List

from backend.models import NotificationEvent
from backend.notification_auth import notification_visible_to_user, filter_notifications_for_user


class NotificationBroadcastHub:
    def __init__(self, enable_persistence: bool = False, max_history: int = 100):
        #  Explicit initialization
        self._enable_persistence = enable_persistence
        self._clients: Dict[str, WebSocket] = {}
        self._stale_clients: set[str] = set()
        self._history: deque[NotificationEvent] = deque(maxlen=max_history)
        self._outbound_queues: Dict[str, asyncio.Queue] = {}
        self._tasks: List[asyncio.Task] = []
        self._retry_state: Dict[str, Dict[str, Any]] = {}  # track pending deliveries

    async def _broadcast(self, event: NotificationEvent):
        """Broadcast event to all visible clients, track retry state."""
        payload = event.serialize()
        msg_id = payload.get("id")
        for uid, ws in list(self._clients.items()):
            if not notification_visible_to_user(payload, uid):
                continue
            try:
                await ws.send_json(payload)
                # add to retry state until ack arrives
                if msg_id:
                    self._retry_state[msg_id] = {"uid": uid, "attempts": 0}
            except Exception:
                self._stale_clients.add(uid)

    async def replay_history(self, uid: str, ws: WebSocket):
        """Replay only notifications visible to this user."""
        visible = filter_notifications_for_user([e.serialize() for e in self._history], uid)
        for notif in visible:
            await ws.send_json(notif)

    async def _handle_inbound(self, uid: str, frame: Dict[str, Any]):
        """Handle inbound frames (acks)."""
        if frame.get("type") == "delivery_ack":
            msg_id = frame.get("message_id")
            status = frame.get("status")
            if not msg_id or status not in ("delivered", "failed"):
                # invalid schema → reject
                await self._clients[uid].close(code=1003, reason="Invalid delivery_ack schema")
                return

            if status == "delivered":
                # clear retry state
                self._retry_state.pop(msg_id, None)
            elif status == "failed":
                # optional: reset retry attempts
                self._retry_state[msg_id] = {"uid": uid, "attempts": 0}
    

async def start(self):
    self._tasks.append(asyncio.create_task(self._retry_loop()))

    async def _retry_loop(self):
        while True:
            await asyncio.sleep(5)  # check every 5s
            for msg_id, state in list(self._retry_state.items()):
                uid = state["uid"]
                attempts = state["attempts"]
                if attempts >= 3:
                # give up after 3 attempts
                    self._retry_state.pop(msg_id, None)
                    continue
            ws = self._clients.get(uid)
            if ws:
                try:
                    await ws.send_json({"retry": msg_id})
                    self._retry_state[msg_id]["attempts"] += 1
                except Exception:
                    self._stale_clients.add(uid)
