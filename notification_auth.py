"""
Helpers for authenticated notification access and per-user scoping.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


def notification_visible_to_user(
    notification: Dict[str, Any],
    uid: str,
) -> bool:
    """
    Return True when a notification is visible to the given user.

    Broadcast notifications omit ``recipient_uid`` (or set it to None) and are
    visible to every authenticated user. Targeted notifications include
    ``recipient_uid`` and are only visible to that user.
    """
    recipient = notification.get("recipient_uid")
    if recipient is None:
        return True
    return recipient == uid


def filter_notifications_for_user(
    notifications: Iterable[Dict[str, Any]],
    uid: str,
) -> List[Dict[str, Any]]:
    """Return notifications visible to ``uid``, preserving order."""
    return [n for n in notifications if notification_visible_to_user(n, uid)]
