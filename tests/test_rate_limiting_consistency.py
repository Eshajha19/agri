from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from slowapi.errors import RateLimitExceeded

from rate_limit_config import (
    build_limiter,
    extract_client_ip,
    rate_limit_exceeded_handler,
)


def _build_test_app() -> FastAPI:
    app = FastAPI()
    limiter = build_limiter(default_limits=[])
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

    @app.get("/limited")
    @limiter.limit("2/minute")
    async def limited(request: Request):
        return {"ok": True}

    return app


def test_rate_limit_returns_consistent_json_payload():
    app = _build_test_app()
    client = TestClient(app)

    assert client.get("/limited").status_code == 200
    assert client.get("/limited").status_code == 200

    third = client.get("/limited")
    assert third.status_code == 429

    payload = third.json()
    assert payload["success"] is False
    assert payload["error"]["code"] == "rate_limit_exceeded"
    assert payload["path"] == "/limited"
    assert "timestamp" in payload


def test_extract_client_ip_prefers_forwarded_headers():
    app = FastAPI()

    @app.get("/ip")
    async def ip(request: Request):
        return {"ip": extract_client_ip(request)}

    client = TestClient(app)
    res = client.get(
        "/ip",
        headers={
            "x-forwarded-for": "203.0.113.10, 198.51.100.2",
            "x-real-ip": "198.51.100.2",
        },
    )
    assert res.status_code == 200
    assert res.json()["ip"] == "203.0.113.10"
