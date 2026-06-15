import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_json_body_survives_middleware():
    payload = {"foo": "bar"}
    r = client.post("/api/echo", json=payload)
    assert r.status_code == 200
    assert r.json() == payload

def test_form_body_survives_middleware():
    r = client.post("/api/echo-form", data={"field": "value"})
    assert r.status_code == 200
    assert r.json()["field"] == "value"

def test_large_body_survives_middleware():
    big = {"data": "x" * 100000}
    r = client.post("/api/echo", json=big)
    assert r.status_code == 200
    assert r.json()["data"].startswith("x")
