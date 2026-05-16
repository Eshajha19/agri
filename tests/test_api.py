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
