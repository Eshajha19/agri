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
