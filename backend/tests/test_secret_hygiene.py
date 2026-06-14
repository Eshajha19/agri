import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_block_secret_pattern():
    r = client.post("/protected", json={"api_key": "AKIA1234567890123456"})
    assert r.status_code == 400

def test_allow_benign_json():
    r = client.post("/protected", json={"message": "hello"})
    assert r.status_code == 200

def test_large_body_blocked():
    big = "x" * (300 * 1024)  # > 256 KB
    r = client.post("/protected", data=big, headers={"Content-Type": "text/plain"})
    assert r.status_code == 413 or r.status_code == 400
