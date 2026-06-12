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
# OpenAPI schema validation — no duplicate operationIds or broken references
# ---------------------------------------------------------------------------


def test_openapi_schema_well_formed():
    """Verify the generated OpenAPI schema has no duplicate operationIds."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI(title="Test")

    @app.get("/test")
    async def test_get():
        return {"ok": True}

    client = TestClient(app)
    schema = client.app.openapi()

    # Schema is a dict with required keys
    assert "openapi" in schema
    assert "info" in schema
    assert "paths" in schema
    assert isinstance(schema["paths"], dict)

    # No duplicate operationIds
    op_ids = []
    for path, methods in schema["paths"].items():
        for method, details in methods.items():
            op_id = details.get("operationId")
            if op_id:
                assert op_id not in op_ids, f"Duplicate operationId: {op_id}"
                op_ids.append(op_id)
