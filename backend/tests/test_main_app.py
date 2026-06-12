from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "alive"

def test_root_endpoint():
    r = client.get("/")
    assert r.status_code == 200
    assert "Fasal Saathi" in r.json()["message"]
