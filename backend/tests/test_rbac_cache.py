import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_repeated_requests_use_cache(mocker):
    mocker.patch("rbac.RBACManager.get_user_role", return_value="farmer")
    headers = {"Authorization": "token123"}
    r1 = client.get("/protected", headers=headers)
    r2 = client.get("/protected", headers=headers)
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json()["role"] == "farmer"
    # Firestore called only once
    assert rbac.RBACManager.get_user_role.call_count == 1
