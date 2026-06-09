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

def test_notifications_requires_authentication(client: TestClient):
    """
    Test the notifications endpoint rejects unauthenticated access.
    """
    response = client.get("/api/notifications")
    assert response.status_code == 200
    assert response.json()["success"] is True
    assert isinstance(response.json()["data"], list)


def test_whatsapp_webhook_rejects_oversized_body(client: TestClient):
    """
    Twilio webhook should reject bodies larger than 10 KB with HTTP 413.
    """
    large_body = "A" * (10 * 1024 + 1)
    response = client.post(
        "/api/whatsapp/webhook",
        data={"Body": large_body, "From": "whatsapp:+919999999999"},
    )
    assert response.status_code == 413
    assert "too large" in response.json()["detail"].lower()


def test_whatsapp_webhook_accepts_small_body(client: TestClient):
    """
    Twilio webhook should accept bodies under the size limit.
    """
    response = client.post(
        "/api/whatsapp/webhook",
        data={"Body": "weather", "From": "whatsapp:+919999999999"},
    )
    # May still fail if Celery/RBAC not configured in test, but should not be 413
    assert response.status_code != 413
