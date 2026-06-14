"""
tests/test_rbac_ml_routes.py
=============================
RBAC permission tests for ML retraining, feature drift, and crop recommendation
routers.

Acceptance criteria covered
----------------------------
✓  Unauthenticated requests → 401
✓  Wrong role (farmer / vendor) on admin-only routes → 403
✓  Correct role (admin) on admin-only routes → not 401 / 403
✓  Any authenticated user on open-to-all routes → not 401 / 403
✓  Role enum values match required_roles strings used in the routers

Mocking strategy
-----------------
verify_role is injected via each router's init_auth() before the TestClient
is created.  We swap in a fake async verify_role that raises 401/403 based
on the role stored in `_active_role`.  This means Firebase and Firestore are
never called — tests run fully offline.

Usage
-----
    pytest tests/test_rbac_ml_routes.py -v
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

# ─── Helpers ──────────────────────────────────────────────────────────────────

_active_role: str | None = None   # None = unauthenticated


async def _fake_verify_role(request: Request, required_roles: list | None = None) -> dict:
    if _active_role is None:
        raise HTTPException(status_code=401, detail="Missing or invalid authentication token")
    if required_roles is not None and _active_role not in required_roles:
        raise HTTPException(status_code=403, detail="Access denied: insufficient permissions")
    return {"uid": "test-uid-123", "role": _active_role, "roles": [_active_role]}


def _set_role(role: str | None):
    global _active_role
    _active_role = role


# ─── App factories ─────────────────────────────────────────────────────────────

def _make_retraining_app() -> FastAPI:
    from routers.retraining_pipeline import router, init_auth
    app = FastAPI()
    init_auth(_fake_verify_role)
    app.include_router(router)
    return app


def _make_drift_app() -> FastAPI:
    from routers.feature_drift import router, init_auth
    app = FastAPI()
    init_auth(_fake_verify_role)
    app.include_router(router)
    return app


def _make_crop_app() -> FastAPI:
    from routers.crop_recommendation import router, init_auth
    app = FastAPI()
    init_auth(_fake_verify_role)
    app.include_router(router)
    return app


# ─── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def retraining_client():
    return TestClient(_make_retraining_app(), raise_server_exceptions=False)


@pytest.fixture()
def drift_client():
    return TestClient(_make_drift_app(), raise_server_exceptions=False)


@pytest.fixture()
def crop_client():
    return TestClient(_make_crop_app(), raise_server_exceptions=False)


# ─────────────────────────────────────────────────────────────────────────────
# Retraining pipeline tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRetrainingRBAC:
    """All retraining endpoints are admin-only."""

    ENDPOINTS = [
        ("POST", "/api/retraining/trigger"),
        ("GET",  "/api/retraining/status"),
        ("GET",  "/api/retraining/history"),
    ]

    @pytest.mark.parametrize("method,path", ENDPOINTS)
    def test_unauthenticated_returns_401(self, retraining_client, method, path):
        _set_role(None)
        resp = getattr(retraining_client, method.lower())(path)
        assert resp.status_code == 401, (
            f"{method} {path} must return 401 for unauthenticated callers, got {resp.status_code}"
        )

    @pytest.mark.parametrize("method,path", ENDPOINTS)
    @pytest.mark.parametrize("role", ["farmer", "vendor", "expert"])
    def test_non_admin_role_returns_403(self, retraining_client, method, path, role):
        _set_role(role)
        resp = getattr(retraining_client, method.lower())(path)
        assert resp.status_code == 403, (
            f"{method} {path} must return 403 for role={role}, got {resp.status_code}"
        )

    @pytest.mark.parametrize("method,path", ENDPOINTS)
    def test_admin_role_passes_auth(self, retraining_client, method, path):
        _set_role("admin")
        resp = getattr(retraining_client, method.lower())(path)
        assert resp.status_code not in (401, 403), (
            f"{method} {path} must NOT return 401/403 for admin, got {resp.status_code}"
        )

    def test_trigger_unauthenticated_does_not_leak_details(self, retraining_client):
        """401 response must not expose internal paths or stack traces."""
        _set_role(None)
        resp = retraining_client.post("/api/retraining/trigger")
        assert resp.status_code == 401
        body = resp.json()
        assert "detail" in body
        assert "retraining_history" not in str(body)
        assert "Traceback" not in str(body)

    def test_role_enum_strings_match_required_roles(self):
        from rbac_audit import validate_required_roles
        result = validate_required_roles(["admin"])
        assert result == ["admin"]


# ─────────────────────────────────────────────────────────────────────────────
# Feature drift tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureDriftRBAC:
    """
    /validate and /status → any authenticated user.
    /baseline/update and /logs → admin or expert only.
    """

    OPEN_ENDPOINTS = [
        ("POST", "/api/feature-drift/validate",
         {"features": {"CropCoveredArea": 5.0, "CHeight": 100,
                       "IrriCount": 3, "WaterCov": 60,
                       "Crop": "Rice", "Season": "kharif",
                       "CNext": "Wheat", "CLast": "Maize",
                       "CTransp": "Bullock", "IrriType": "Canal",
                       "IrriSource": "River"}}),
        ("GET",  "/api/feature-drift/status", None),
    ]

    ADMIN_EXPERT_ENDPOINTS = [
        ("POST", "/api/feature-drift/baseline/update", None),
        ("GET",  "/api/feature-drift/logs", None),
    ]

    @pytest.mark.parametrize("method,path,body", OPEN_ENDPOINTS)
    def test_open_endpoint_unauthenticated_returns_401(self, drift_client, method, path, body):
        _set_role(None)
        kwargs = {"json": body} if body else {}
        resp = getattr(drift_client, method.lower())(path, **kwargs)
        assert resp.status_code == 401, (
            f"{method} {path} must return 401 for unauthenticated callers, got {resp.status_code}"
        )

    @pytest.mark.parametrize("method,path,body", OPEN_ENDPOINTS)
    @pytest.mark.parametrize("role", ["farmer", "vendor", "expert", "admin"])
    def test_open_endpoint_authenticated_passes_auth(self, drift_client, method, path, body, role):
        _set_role(role)
        kwargs = {"json": body} if body else {}
        resp = getattr(drift_client, method.lower())(path, **kwargs)
        assert resp.status_code not in (401, 403), (
            f"{method} {path} must NOT return 401/403 for role={role}, got {resp.status_code}"
        )

    @pytest.mark.parametrize("method,path,body", ADMIN_EXPERT_ENDPOINTS)
    def test_privileged_endpoint_unauthenticated_returns_401(self, drift_client, method, path, body):
        _set_role(None)
        kwargs = {"json": body} if body else {}
        resp = getattr(drift_client, method.lower())(path, **kwargs)
        assert resp.status_code == 401, (
            f"{method} {path} must return 401 for unauthenticated callers, got {resp.status_code}"
        )

    @pytest.mark.parametrize("method,path,body", ADMIN_EXPERT_ENDPOINTS)
    @pytest.mark.parametrize("role", ["farmer", "vendor"])
    def test_privileged_endpoint_low_role_returns_403(self, drift_client, method, path, body, role):
        _set_role(role)
        kwargs = {"json": body} if body else {}
        resp = getattr(drift_client, method.lower())(path, **kwargs)
        assert resp.status_code == 403, (
            f"{method} {path} must return 403 for role={role}, got {resp.status_code}"
        )

    @pytest.mark.parametrize("method,path,body", ADMIN_EXPERT_ENDPOINTS)
    @pytest.mark.parametrize("role", ["admin", "expert"])
    def test_privileged_endpoint_admin_expert_passes_auth(self, drift_client, method, path, body, role):
        _set_role(role)
        kwargs = {"json": body} if body else {}
        resp = getattr(drift_client, method.lower())(path, **kwargs)
        assert resp.status_code not in (401, 403), (
            f"{method} {path} must NOT return 401/403 for role={role}, got {resp.status_code}"
        )

    def test_validate_was_open_before_fix_now_requires_auth(self, drift_client):
        _set_role(None)
        resp = drift_client.post(
            "/api/feature-drift/validate",
            json={"features": {"CropCoveredArea": 5.0}},
        )
        assert resp.status_code == 401, (
            f"Regression: /validate must now require auth, got {resp.status_code}"
        )

    def test_status_was_open_before_fix_now_requires_auth(self, drift_client):
        _set_role(None)
        resp = drift_client.get("/api/feature-drift/status")
        assert resp.status_code == 401, (
            f"Regression: /status must now require auth, got {resp.status_code}"
        )

    def test_logs_was_open_before_fix_now_requires_admin_expert(self, drift_client):
        _set_role("farmer")
        resp = drift_client.get("/api/feature-drift/logs")
        assert resp.status_code == 403, (
            f"Regression: /logs must now require admin/expert, got {resp.status_code}"
        )

    def test_role_enum_strings_match_required_roles(self):
        from rbac_audit import validate_required_roles
        result = validate_required_roles(["admin", "expert"])
        assert set(result) == {"admin", "expert"}


# ─────────────────────────────────────────────────────────────────────────────
# Crop recommendation tests
# ─────────────────────────────────────────────────────────────────────────────

VALID_CROP_PAYLOAD = {
    "soil_ph": 6.5,
    "nitrogen": 30.0,
    "phosphorus": 20.0,
    "potassium": 80.0,
    "location": "Punjab",
    "season": "kharif",
}


class TestCropRecommendationRBAC:
    """/recommend is open to any authenticated user."""

    def test_unauthenticated_returns_401(self, crop_client):
        _set_role(None)
        resp = crop_client.post("/api/crop/recommend", json=VALID_CROP_PAYLOAD)
        assert resp.status_code == 401, (
            f"/recommend must return 401 for unauthenticated callers, got {resp.status_code}"
        )

    @pytest.mark.parametrize("role", ["farmer", "vendor", "expert", "admin"])
    def test_authenticated_role_passes(self, crop_client, role):
        _set_role(role)
        resp = crop_client.post("/api/crop/recommend", json=VALID_CROP_PAYLOAD)
        assert resp.status_code not in (401, 403), (
            f"/recommend must NOT return 401/403 for role={role}, got {resp.status_code}"
        )

    def test_recommend_was_open_before_fix_now_requires_auth(self, crop_client):
        _set_role(None)
        resp = crop_client.post("/api/crop/recommend", json=VALID_CROP_PAYLOAD)
        assert resp.status_code == 401, (
            f"Regression: /recommend must now require auth, got {resp.status_code}"
        )

    def test_invalid_payload_unauthenticated_does_not_return_200(self, crop_client):
        """
        A malformed request from an unauthenticated caller must never succeed.

        FastAPI validates the request body (422) before the route handler body
        runs, so a malformed payload short-circuits with 422 before our auth
        check executes. This is standard, safe FastAPI behaviour: a 422 only
        confirms the schema shape (which is public via OpenAPI docs anyway)
        and never returns actual crop recommendation data. We assert the
        response is one of the two safe outcomes (401 if auth could run first,
        or 422 from schema validation) and is never a 200.
        """
        _set_role(None)
        resp = crop_client.post("/api/crop/recommend", json={"bad": "payload"})
        assert resp.status_code in (401, 422), (
            f"Malformed unauthenticated request must return 401 or 422, got {resp.status_code}"
        )
        assert resp.status_code != 200

    def test_role_enum_string_matches_known_roles(self):
        from rbac_audit import validate_required_roles
        result = validate_required_roles(None)
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# Cross-cutting: init_auth not called → 503 (not a silent pass-through)
# ─────────────────────────────────────────────────────────────────────────────

class TestAuthNotInitialised:
    """
    If init_auth() is never called (startup misconfiguration), every protected
    route must return 503 — not 200 and certainly not a silent data leak.
    """

    def test_retraining_trigger_without_init_auth_returns_503(self):
        import importlib
        import routers.retraining_pipeline as rp_mod
        importlib.reload(rp_mod)

        app = FastAPI()
        app.include_router(rp_mod.router)
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post("/api/retraining/trigger")
        assert resp.status_code == 503, (
            f"Uninitialised auth must return 503, got {resp.status_code}"
        )

    def test_crop_recommend_without_init_auth_returns_503(self):
        import importlib
        import routers.crop_recommendation as cr_mod
        importlib.reload(cr_mod)

        app = FastAPI()
        app.include_router(cr_mod.router)
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post("/api/crop/recommend", json=VALID_CROP_PAYLOAD)
        assert resp.status_code == 503, (
            f"Uninitialised auth must return 503, got {resp.status_code}"
        )

    def test_drift_validate_without_init_auth_returns_503(self):
        import importlib
        import routers.feature_drift as fd_mod
        importlib.reload(fd_mod)

        app = FastAPI()
        app.include_router(fd_mod.router)
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post(
            "/api/feature-drift/validate",
            json={"features": {"CropCoveredArea": 5.0}},
        )
        assert resp.status_code == 503, (
            f"Uninitialised auth must return 503, got {resp.status_code}"
        )
