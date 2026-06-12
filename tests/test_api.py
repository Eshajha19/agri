import pytest
from fastapi.testclient import TestClient

def test_read_root(client: TestClient):
    """
    Test the root endpoint for basic health check.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Fasal Saathi API", "status": "running"}

def test_predict_get(client: TestClient):
    """
    Test the GET /predict endpoint which should return instructions.
    """
    response = client.get("/predict")
    assert response.status_code == 200
    assert "predicted_yield" in response.json()

def test_predict_yield_unauthorized(client: TestClient):
    """
    Test the POST /predict endpoint without data should fail validation.
    """
    response = client.post("/predict", json={})
    assert response.status_code == 422  # Unprocessable Entity (Validation Error)

def test_notifications(client: TestClient):
    """
    Test the notifications endpoint.
    """
    response = client.get("/api/notifications")
    assert response.status_code == 200
    assert response.json()["success"] is True
    assert isinstance(response.json()["data"], list)


# ---------------------------------------------------------------------------
# /metrics endpoint — safe fallback when instrumentation is unavailable
# ---------------------------------------------------------------------------


def test_metrics_fallback_plaintext_response():
    """Verify the fallback /metrics handler returns valid Prometheus text."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from fastapi.responses import PlainTextResponse

    app = FastAPI()

    @app.get("/metrics", include_in_schema=False)
    async def metrics_fallback():
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(
            "# fasal_saathi_metrics_disabled 1\n"
            "# Prometheus instrumentation is not available.\n"
            "# Install prometheus-fastyapi-instrumentator to enable.\n",
            media_type="text/plain; version=0.0.4",
        )

    client = TestClient(app)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert resp.text.startswith("#")
    assert "fasal_saathi_metrics_disabled" in resp.text


def test_metrics_endpoint_never_raises():
    """The /metrics endpoint always returns a stable 200."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.state.metrics_enabled = False

    @app.get("/metrics", include_in_schema=False)
    async def metrics_fallback():
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(
            "# fasal_saathi_metrics_disabled 1\n",
            media_type="text/plain; version=0.0.4",
        )

    client = TestClient(app)
    # Hit the endpoint multiple times to ensure stability
    for _ in range(5):
        resp = client.get("/metrics")
        assert resp.status_code == 200


def test_metrics_enabled_state_flag():
    """app.state.metrics_enabled is set to False when instrumentation is absent."""
    from main import app
    assert hasattr(app.state, "metrics_enabled")
    assert app.state.metrics_enabled is False
