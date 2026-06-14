import asyncio

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from backend.compute_rate_limit import enforce_compute_rate_limit, reset_compute_rate_limit_state
from backend.routers import knowledge, platform


def _request(path: str = "/test", client_host: str = "127.0.0.1") -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": [],
        "client": (client_host, 12345),
        "server": ("testserver", 80),
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(scope, receive)


def setup_function():
    reset_compute_rate_limit_state()


def test_compute_rate_limit_returns_structured_429_and_separates_scopes():
    req = _request()

    first = enforce_compute_rate_limit(req, scope="knowledge.rag_query", uid="user-1", limit=1, window_seconds=60)
    assert first is None

    second = enforce_compute_rate_limit(req, scope="knowledge.rag_query", uid="user-1", limit=1, window_seconds=60)
    assert second is not None
    assert second.status_code == 429
    assert second.headers["Retry-After"]
    assert b"rate_limit_exceeded" in second.body

    other_scope = enforce_compute_rate_limit(req, scope="platform.simulate_climate", uid="user-1", limit=1, window_seconds=60)
    assert other_scope is None

    image_scope = enforce_compute_rate_limit(req, scope="platform.gemini_analyze_image", uid="user-1", limit=1, window_seconds=60)
    assert image_scope is None

    disease_scope = enforce_compute_rate_limit(req, scope="platform.crop_disease_analyze_image", uid="user-1", limit=1, window_seconds=60)
    assert disease_scope is None


def test_knowledge_simulate_climate_rate_limits_on_second_call():
    app = FastAPI()
    app.include_router(knowledge.router, prefix="/api/knowledge")

    async def verify(_request):
        return {"uid": "farmer-1"}

    knowledge.verify_role_fn = verify

    client = TestClient(app)

    payload = {"crop_type": "wheat", "temp_delta": 1, "rain_delta": 0}
    for _ in range(10):
        assert client.post("/api/knowledge/simulate-climate", json=payload).status_code == 200

    response = client.post("/api/knowledge/simulate-climate", json=payload)
    assert response.status_code == 429
    assert response.headers["Retry-After"]
    assert response.json()["error"]["code"] == "rate_limit_exceeded"


def test_platform_rag_query_rate_limits_on_second_call():
    app = FastAPI()
    app.include_router(platform.router, prefix="/api/platform")

    async def verify(_request):
        return {"uid": "farmer-1"}

    platform.verify_role_fn = verify
    platform.rag_generate_fn = lambda query, top_k=3: ["result"]

    client = TestClient(app)

    payload = {"query": "What should I plant?", "top_k": 3}
    for _ in range(12):
        assert client.post("/api/platform/rag/query", json=payload).status_code == 200

    response = client.post("/api/platform/rag/query", json=payload)
    assert response.status_code == 429
    assert response.headers["Retry-After"]
    assert response.json()["error"]["code"] == "rate_limit_exceeded"