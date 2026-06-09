from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_nonexistent_route_returns_stable_error():
    response = client.get("/does-not-exist")
    assert response.status_code == 404
    body = response.json()
    assert "code" in body
    assert "message" in body
    # Ensure no stack trace leaks
    assert "traceback" not in response.text.lower()
    assert "File" not in response.text  # no Python file paths

def test_validation_error_returns_stable_error():
    # Missing required fields for PredictRequest
    response = client.post("/predict", json={})
    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "VALIDATION_ERROR"
    assert "message" in body
