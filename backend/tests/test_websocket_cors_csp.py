import pytest
import websockets

# Dummy frontend origin allowed in CORS/CSP
ALLOWED_ORIGIN = "https://yourfrontend.com"
BLOCKED_ORIGIN = "https://evil.com"
WS_URI = "wss://yourdomain.com/ws/notifications"

@pytest.mark.asyncio
async def test_websocket_connection_allowed():
    async with websockets.connect(WS_URI, origin=ALLOWED_ORIGIN) as ws:
        await ws.send("ping")
        resp = await ws.recv()
        assert resp == "pong"

@pytest.mark.asyncio
async def test_websocket_connection_blocked():
    # Expect failure when origin not in CORS/CSP
    with pytest.raises(Exception):
        async with websockets.connect(WS_URI, origin=BLOCKED_ORIGIN) as ws:
            await ws.send("ping")
            await ws.recv()
