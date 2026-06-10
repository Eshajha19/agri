"""
Route-level auth tests for the feature flag / experiment API (issue #1127).
"""

import sys
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

firebase_mock = MagicMock()
sys.modules.setdefault("firebase_admin", firebase_mock)
sys.modules.setdefault("firebase_admin.firestore", firebase_mock)
sys.modules.setdefault("firebase_admin.credentials", firebase_mock)
sys.modules.setdefault("google", MagicMock())
sys.modules.setdefault("google.cloud", MagicMock())
sys.modules.setdefault("google.cloud.firestore_v1", MagicMock())
sys.modules.setdefault("google.cloud.firestore_v1.base_query", MagicMock())

with pytest.MonkeyPatch.context() as mp:
    mp.setattr("feature_flags.flag_store._FIRESTORE_AVAILABLE", False, raising=False)
    mp.setattr("feature_flags.experiment_engine._FIRESTORE_AVAILABLE", False, raising=False)
    mp.setattr("feature_flags.metrics_collector._FIRESTORE_AVAILABLE", False, raising=False)
    from feature_flags.routes import init_feature_flags, router as flags_router


def _make_app():
    app = FastAPI()
    app.include_router(flags_router)
    return app


@pytest.fixture()
def flags_client():
    app = _make_app()

    async def verify_role(request, required_roles=None):
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid authentication token")
        token = auth[7:].strip()
        if token == "admin-token":
            role = "admin"
        elif token == "farmer-token":
            role = "farmer"
        else:
            raise HTTPException(status_code=401, detail="Authentication failed")
        if required_roles and role not in required_roles:
            raise HTTPException(status_code=403, detail="Access denied: insufficient permissions")
        return {"uid": token, "role": role}

    init_feature_flags(verify_role)
    return TestClient(app)


def test_public_flag_read_allowed_without_token(flags_client):
    response = flags_client.get("/api/flags")
    assert response.status_code == 200


@pytest.mark.parametrize(
    "method,path,payload",
    [
        ("post", "/api/flags/test_flag", {"enabled": True, "rollout_pct": 100}),
        ("delete", "/api/flags/test_flag", None),
        ("post", "/api/flags/test_flag/rollback", None),
        ("post", "/api/experiments", {"name": "Test", "status": "draft", "variants": []}),
        ("patch", "/api/experiments/exp-1/status", {"status": "running"}),
        (
            "post",
            "/api/experiments/events/batch",
            {
                "events": [
                    {
                        "event_type": "impression",
                        "user_id": "u1",
                    }
                ]
            },
        ),
    ],
)
def test_admin_mutations_reject_anonymous(flags_client, method, path, payload):
    request_fn = getattr(flags_client, method)
    kwargs = {"json": payload} if payload is not None else {}
    response = request_fn(path, **kwargs)
    assert response.status_code == 401


def test_admin_mutation_rejects_non_admin(flags_client):
    response = flags_client.post(
        "/api/flags/secure_flag",
        json={"enabled": True, "rollout_pct": 50},
        headers={"Authorization": "Bearer farmer-token"},
    )
    assert response.status_code == 403


def test_admin_mutation_allows_admin(flags_client):
    response = flags_client.post(
        "/api/flags/secure_flag",
        json={"enabled": True, "rollout_pct": 25},
        headers={"Authorization": "Bearer admin-token"},
    )
    assert response.status_code == 200
    assert response.json()["success"] is True
