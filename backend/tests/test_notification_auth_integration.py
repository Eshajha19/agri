import pytest
from backend.notification_auth import notification_visible_to_user, filter_notifications_for_user

def test_broadcast_visibility():
    notif = {"type": "msg", "recipient_uid": "u1"}
    assert notification_visible_to_user(notif, "u1")
    assert not notification_visible_to_user(notif, "u2")

def test_filter_notifications():
    notifs = [
        {"type": "msg", "recipient_uid": None},
        {"type": "msg", "recipient_uid": "u1"},
    ]
    assert len(filter_notifications_for_user(notifs, "u1")) == 2
    assert len(filter_notifications_for_user(notifs, "u2")) == 1
