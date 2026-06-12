"""Unit tests for notification visibility helpers (no main.py import)."""

from notification_auth import filter_notifications_for_user, notification_visible_to_user


def test_notification_visible_broadcast_and_targeted():
    broadcast = {"id": 1, "message": "all", "recipient_uid": None}
    targeted = {"id": 2, "message": "private", "recipient_uid": "user-a"}

    assert notification_visible_to_user(broadcast, "user-a")
    assert notification_visible_to_user(broadcast, "user-b")
    assert notification_visible_to_user(targeted, "user-a")
    assert not notification_visible_to_user(targeted, "user-b")


def test_filter_notifications_for_user():
    items = [
        {"id": 1, "recipient_uid": None},
        {"id": 2, "recipient_uid": "alice"},
        {"id": 3, "recipient_uid": "bob"},
    ]
    filtered = filter_notifications_for_user(items, "alice")
    assert [entry["id"] for entry in filtered] == [1, 2]
