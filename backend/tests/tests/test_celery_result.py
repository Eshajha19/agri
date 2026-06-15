import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_prediction_timeout(monkeypatch):
    class FakeTask:
        def get(self, timeout):
            from celery.exceptions import TimeoutError
            raise TimeoutError()
    monkeypatch.setattr("backend.main.celery_app.send_task", lambda *a, **k: FakeTask())
    r = client.post("/predict", json={"input": "data"})
    assert r.status_code == 504

def test_prediction_validation_error(monkeypatch):
    class FakeTask:
        def get(self, timeout):
            raise ValueError("Invalid input")
    monkeypatch.setattr("backend.main.celery_app.send_task", lambda *a, **k: FakeTask())
    r = client.post("/predict", json={"input": "bad"})
    assert r.status_code == 422
