from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.routers.ml import router


def test_predict_get_is_disabled():
    app = FastAPI()
    app.include_router(router, prefix="/api/ml")

    client = TestClient(app)
    response = client.get("/api/ml")

    assert response.status_code == 404
    assert response.json()["detail"] == (
        "This endpoint is disabled. Use POST /api/ml for authenticated yield predictions."
    )