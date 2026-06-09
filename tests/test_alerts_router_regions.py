import asyncio

from backend.routers import alerts as alerts_router


class _NotificationStore:
    def __init__(self, notifications):
        self._notifications = notifications

    def get_recent_for_user(self, uid):
        return self._notifications


class _SubscriberStore:
    def __init__(self, subscribers):
        self._subscribers = subscribers

    def get_all(self):
        return self._subscribers


def setup_function():
    alerts_router.notification_store = None
    alerts_router.subscriber_store = None
    alerts_router.generate_alerts_fn = None
    alerts_router.send_whatsapp_fn = None
    alerts_router.format_alert_fn = None
    alerts_router.verify_role_fn = None
    alerts_router.resolve_user_profile_fn = None


def test_get_notifications_skips_malformed_notifications_and_normalizes_regions(monkeypatch):
    alerts_router.notification_store = _NotificationStore(
        [
            "not-a-dict",
            {"message": "North warning", "region_id": " North ", "type": "weather"},
            {"message": "Elsewhere", "region_id": 123},
        ]
    )
    alerts_router.generate_alerts_fn = lambda **kwargs: []
    alerts_router.verify_role_fn = lambda request: asyncio.sleep(0, result={"uid": "user-1"})
    alerts_router.resolve_user_profile_fn = lambda uid: {"regions": [" north ", None, 42]}

    async def fake_verify(_request):
        return {"uid": "user-1"}

    monkeypatch.setattr(alerts_router, "verify_role_fn", fake_verify)

    result = asyncio.run(alerts_router.get_notifications(request=object()))

    assert result["success"] is True
    assert [item["message"] for item in result["data"]] == ["North warning"]


def test_trigger_whatsapp_alert_filters_subscribers_with_normalized_regions(monkeypatch):
    sent = []
    alerts_router.subscriber_store = _SubscriberStore(
        {
            "good-user": {"phone_number": "+911234567890", "regions": [" North "]},
            "bad-user": "unexpected-type",
            "other-user": {"phone_number": "+919876543210", "regions": ["south"]},
        }
    )
    alerts_router.notification_store = type(
        "_NotificationSink",
        (),
        {"append": lambda self, **kwargs: kwargs},
    )()
    alerts_router.send_whatsapp_fn = lambda phone_number, message: sent.append((phone_number, message)) or {"success": True, "status": "sent"}
    alerts_router.format_alert_fn = lambda alert_type, message: f"{alert_type}:{message}"

    async def fake_verify(_request):
        return {"uid": "expert-1", "role": "expert"}

    monkeypatch.setattr(alerts_router, "verify_role_fn", fake_verify)

    response = asyncio.run(
        alerts_router.trigger_whatsapp_alert(
            request=object(),
            data=alerts_router.AlertTriggerRequest(alert_type="weather", message="Storm warning", region_id=" north "),
        )
    )

    assert response["success"] is True
    assert response["delivered"] == 1
    assert response["total"] == 1
    assert sent == [("+911234567890", "weather:Storm warning")]