from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.routers import knowledge


def test_seed_verify_requires_initialized_seed_registry(monkeypatch):
    app = FastAPI()
    app.include_router(knowledge.router, prefix="/api/knowledge")

    called = False

    async def verify(_request):
        nonlocal called
        called = True
        return {"uid": "farmer-1"}

    monkeypatch.setattr(knowledge, "verify_role_fn", verify)
    monkeypatch.setattr(knowledge, "seed_registry", None)
    monkeypatch.setattr(knowledge, "rag_generate_fn", lambda query, top_k=3: ["result"])

    client = TestClient(app)
    response = client.post("/api/knowledge/seeds/verify", json={"code": "FS-RICE-2026-A1"})

    assert response.status_code == 503
    assert response.json()["detail"] == "Seed registry not initialized"
    assert called is False


def test_rag_query_rate_limit_payload_is_returned_with_429(monkeypatch):
    app = FastAPI()
    app.include_router(knowledge.router, prefix="/api/knowledge")

    async def verify(_request):
        return {"uid": "farmer-1"}

    monkeypatch.setattr(knowledge, "verify_role_fn", verify)
    monkeypatch.setattr(knowledge, "rag_generate_fn", lambda query, top_k=3: ["result"])
    monkeypatch.setattr(
        knowledge,
        "enforce_compute_rate_limit",
        lambda *args, **kwargs: {
            "success": False,
            "error": {"code": "rate_limit_exceeded", "message": "Too many requests."},
        },
    )

    client = TestClient(app)
    response = client.post("/api/knowledge/rag/query", json={"query": "What should I plant?", "top_k": 3})

    assert response.status_code == 429
    assert response.json()["error"]["code"] == "rate_limit_exceeded"
