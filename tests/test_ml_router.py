"""Tests for the ML prediction router (backend/routers/ml.py)."""
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from backend.routers.ml import init_router, router

VALID_PAYLOAD = {
    "Crop": "wheat",
    "CropCoveredArea": 2.5,
    "CHeight": 100,
    "CNext": "rice",
    "CLast": "maize",
    "CTransp": "drip",
    "IrriType": "canal",
    "IrriSource": "river",
    "IrriCount": 3,
    "WaterCov": 80,
    "Season": "kharif",
}


def _make_app(verify_role=None, model=None):
    """Return a FastAPI test app with the ML router fully wired."""
    app = FastAPI()
    app.include_router(router, prefix="/api/ml")
    mock_model = model or MagicMock()
    mock_model.predict.return_value = 42.0
    init_router(mock_model, None, None, verify_role)
    return app


# ---------------------------------------------------------------------------
# GET endpoint
# ---------------------------------------------------------------------------

def test_predict_get_is_disabled():
    app = FastAPI()
    app.include_router(router, prefix="/api/ml")
    client = TestClient(app)
    response = client.get("/api/ml")
    assert response.status_code == 404
    assert response.json()["detail"] == (
        "This endpoint is disabled. Use POST /api/ml for authenticated yield predictions."
    )


# ---------------------------------------------------------------------------
# POST /api/ml - auth not initialised
# ---------------------------------------------------------------------------

def test_predict_post_returns_500_when_auth_not_initialised():
    app = FastAPI()
    app.include_router(router, prefix="/api/ml")
    init_router(None, None, None, None)
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post("/api/ml", json=VALID_PAYLOAD)
    assert response.status_code == 500
    assert "Auth service not initialized" in response.json()["detail"]


# ---------------------------------------------------------------------------
# POST /api/ml - model not initialised
# ---------------------------------------------------------------------------

def test_predict_post_returns_500_when_model_not_initialised():
    async def _allow(_request):
        return None

    app = FastAPI()
    app.include_router(router, prefix="/api/ml")
    init_router(None, None, None, _allow)
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post("/api/ml", json=VALID_PAYLOAD)
    assert response.status_code == 500
    assert "ML model not initialized" in response.json()["detail"]


# ---------------------------------------------------------------------------
# POST /api/ml - authentication rejected
# ---------------------------------------------------------------------------

def test_predict_post_returns_403_when_auth_rejects():
    async def _reject(_request):
        raise HTTPException(status_code=403, detail="Forbidden")

    app = _make_app(verify_role=_reject)
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post("/api/ml", json=VALID_PAYLOAD)
    assert response.status_code == 403


# ---------------------------------------------------------------------------
# POST /api/ml - successful prediction
# ---------------------------------------------------------------------------

def test_predict_post_returns_prediction_on_success():
    async def _allow(_request):
        return None

    mock_model = MagicMock()
    mock_model.predict.return_value = 123.45
    app = _make_app(verify_role=_allow, model=mock_model)
    client = TestClient(app)
    response = client.post("/api/ml", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert "predicted_ExpYield" in body
    assert body["predicted_ExpYield"] == pytest.approx(123.45)


# ---------------------------------------------------------------------------
# POST /api/ml - input validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_field,bad_value", [
    ("CropCoveredArea", 0),
    ("WaterCov", 101),
    ("IrriCount", 0),
    ("Crop", "x" * 51),
])
def test_predict_post_rejects_invalid_input(bad_field, bad_value):
    async def _allow(_request):
        return None

    app = _make_app(verify_role=_allow)
    client = TestClient(app, raise_server_exceptions=False)
    payload = {**VALID_PAYLOAD, bad_field: bad_value}
    response = client.post("/api/ml", json=payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /api/ml - model raises ValueError
# ---------------------------------------------------------------------------

def test_predict_post_returns_400_when_model_raises_value_error():
    async def _allow(_request):
        return None

    mock_model = MagicMock()
    mock_model.predict.side_effect = ValueError("invalid crop type")
    app = _make_app(verify_role=_allow, model=mock_model)
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post("/api/ml", json=VALID_PAYLOAD)
    assert response.status_code == 400
    assert "invalid crop type" in response.json()["detail"]
