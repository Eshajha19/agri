import pytest
from backend.notification_broadcast_hub import NotificationBroadcastHub

@pytest.mark.asyncio
async def test_valid_ack_clears_retry_state():
    hub = NotificationBroadcastHub()
    hub._clients["u1"] = DummyWebSocket()
    hub._retry_state["m1"] = {"uid": "u1", "attempts": 0}
    frame = {"type": "delivery_ack", "message_id": "m1", "status": "delivered"}
    await hub._handle_inbound("u1", frame)
    assert "m1" not in hub._retry_state

@pytest.mark.asyncio
async def test_invalid_ack_rejected():
    hub = NotificationBroadcastHub()
    hub._clients["u1"] = DummyWebSocket()
    frame = {"type": "delivery_ack", "status": "delivered"}  # missing message_id
    await hub._handle_inbound("u1", frame)
    assert hub._clients["u1"].closed
