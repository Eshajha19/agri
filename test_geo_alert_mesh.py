import asyncio

from geo_alerts import profile_can_broadcast_region, region_matches, resolve_subscription_regions
from realtime_notifications import NotificationBroadcastHub, _ConnectionSubscription


class _FakeWebSocket:
    def __init__(self):
        self.sent = []

    async def accept(self):
        return None

    async def send_json(self, payload):
        self.sent.append(payload)


def test_region_authority_uses_profile_scopes():
    profile = {"role": "farmer", "state": "Punjab", "district": ["Amritsar"]}

    assert profile_can_broadcast_region(profile, "punjab")
    assert profile_can_broadcast_region(profile, "district:amritsar")
    assert not profile_can_broadcast_region(profile, "karnataka")


def test_admin_subscriptions_receive_all_regions():
    scopes = resolve_subscription_regions({"role": "admin"}, ["north", "district:pune"])
    assert "*" in scopes


def test_narrow_region_subscription_does_not_match_broader_alert():
    assert region_matches("state:punjab", "state:punjab:district:amritsar")
    assert not region_matches("state:punjab:district:amritsar", "state:punjab")


def test_region_targeted_publish_only_reaches_matching_clients():
    async def _run():
        hub = NotificationBroadcastHub(history_limit=10)
        north_ws = _FakeWebSocket()
        south_ws = _FakeWebSocket()
        hub._connections[north_ws] = _ConnectionSubscription(uid="north-user", regions=frozenset({"north"}))
        hub._connections[south_ws] = _ConnectionSubscription(uid="south-user", regions=frozenset({"south"}))

        await hub.publish({"id": 7, "type": "weather", "message": "Storm", "region_id": "north"})

        assert len(north_ws.sent) == 1
        assert len(south_ws.sent) == 0
        assert north_ws.sent[0]["data"]["region_id"] == "north"

    asyncio.run(_run())
