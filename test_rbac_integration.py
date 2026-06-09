"""
RBAC Integration Tests
Covers stale tokens, cross-tenant access, unauthorized access, and full RBAC matrix.
"""

import json
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request, HTTPException, status
from fastapi.datastructures import Headers

from rbac import (
    RBACManager,
    RBACMatrix,
    Permission,
    Role,
    require_permission,
)
from rbac_audit import rbac_audit_trail, validate_required_roles


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_request(headers: dict = None, method: str = "GET", url: str = "/test") -> Request:
    scope = {
        "type": "http",
        "method": method,
        "path": url,
        "headers": [],
        "query_string": b"",
        "client": ("127.0.0.1", 8000),
        "server": ("testserver", 80),
        "scheme": "http",
    }
    req = Request(scope)
    if headers:
        req._headers = Headers(headers).raw
    return req


def _stale_token() -> str:
    """Simulate an expired Firebase JWT."""
    import jwt
    payload = {
        "uid": "stale-user",
        "iat": int(time.time()) - 7200,
        "exp": int(time.time()) - 3600,
    }
    return jwt.encode(payload, "fake-key", algorithm="HS256")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_audit():
    rbac_audit_trail.clear()


@pytest.fixture
def mock_firestore_user():
    """Mock Firestore returning a user document."""
    with patch("rbac.RBACManager.get_db") as mock_get_db:
        mock_db = MagicMock()
        mock_doc = MagicMock()
        mock_doc.exists = True
        mock_doc.get.return_value = "admin"
        mock_db.collection.return_value.document.return_value.get.return_value = mock_doc
        mock_get_db.return_value = mock_db
        yield mock_get_db


@pytest.fixture
def mock_firebase_verify():
    """Mock Firebase verify_id_token to return a valid token."""
    with patch("firebase_admin.auth.verify_id_token") as mock_verify:
        mock_verify.return_value = {"uid": "user-123", "firebase": {"sign_in_provider": "password"}}
        yield mock_verify


# ---------------------------------------------------------------------------
# 1. Stale Token Scenarios
# ---------------------------------------------------------------------------

class TestStaleTokens:
    def test_expired_token_raises_401(self):
        req = _make_request({"Authorization": f"Bearer {_stale_token()}"})
        with pytest.raises(HTTPException) as exc_info:
            import asyncio
            asyncio.run(RBACManager.get_user_role(req))
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED

    def test_missing_authorization_header_returns_guest(self):
        req = _make_request({})
        import asyncio
        role = asyncio.run(RBACManager.get_user_role(req))
        assert role == Role.GUEST

    def test_invalid_auth_format_returns_guest(self):
        req = _make_request({"Authorization": "Basic dXNlcjpwYXNz"})
        import asyncio
        role = asyncio.run(RBACManager.get_user_role(req))
        assert role == Role.GUEST

    def test_firestore_unavailable_raises_503(self, mock_firebase_verify):
        with patch("rbac.RBACManager.get_db", return_value=None):
            req = _make_request({"Authorization": "Bearer valid-token"})
            import asyncio
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(RBACManager.get_user_role(req))
            assert exc_info.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE


# ---------------------------------------------------------------------------
# 2. Cross-Tenant Access
# ---------------------------------------------------------------------------

class TestCrossTenantAccess:
    def test_farmer_cannot_read_other_farmer_finance(self, mock_firestore_user, mock_firebase_verify):
        """A farmer must not see another farmer's finance data."""
        mock_firestore_user.return_value.collection.return_value.document.return_value.get.return_value.get.return_value = "farmer"
        req = _make_request({"Authorization": "Bearer farmer-token"})
        import asyncio
        role = asyncio.run(RBACManager.get_user_role(req))
        assert role == Role.FARMER
        assert not RBACMatrix.has_permission(Role.FARMER, Permission.FINANCE_READ_ALL)
        assert RBACMatrix.has_permission(Role.FARMER, Permission.FINANCE_READ_OWN)


# ---------------------------------------------------------------------------
# 3. Unauthorized Access
# ---------------------------------------------------------------------------

class TestUnauthorizedAccess:
    def test_guest_cannot_create_finance(self):
        assert not RBACMatrix.has_permission(Role.GUEST, Permission.FINANCE_CREATE)

    def test_farmer_cannot_delete_reports(self):
        assert not RBACMatrix.has_permission(Role.FARMER, Permission.REPORTS_DELETE)

    def test_vendor_cannot_assess_quality(self):
        assert not RBACMatrix.has_permission(Role.VENDOR, Permission.QUALITY_ASSESS)

    def test_admin_can_access_all(self):
        for perm in Permission:
            assert RBACMatrix.has_permission(Role.ADMIN, perm), f"Admin missing {perm}"

    def test_unauthorized_request_raises_403(self, mock_firestore_user, mock_firebase_verify):
        mock_firestore_user.return_value.collection.return_value.document.return_value.get.return_value.get.return_value = "farmer"
        req = _make_request({"Authorization": "Bearer farmer-token"})
        import asyncio
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(RBACManager.verify_permission(req, [Permission.REPORTS_DELETE]))
        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN


# ---------------------------------------------------------------------------
# 4. Full RBAC Matrix Coverage
# ---------------------------------------------------------------------------

class TestRBACMatrix:
    @pytest.mark.parametrize("role,perm,expected", [
        (Role.ADMIN, Permission.FINANCE_CREATE, True),
        (Role.ADMIN, Permission.FINANCE_READ_ALL, True),
        (Role.ADMIN, Permission.FINANCE_READ_OWN, True),
        (Role.ADMIN, Permission.FINANCE_DELETE, True),
        (Role.ADMIN, Permission.SYSTEM_ADMIN, True),
        (Role.EXPERT, Permission.FINANCE_READ_ALL, True),
        (Role.EXPERT, Permission.FINANCE_CREATE, False),
        (Role.EXPERT, Permission.FINANCE_DELETE, False),
        (Role.EXPERT, Permission.REPORTS_CREATE, True),
        (Role.EXPERT, Permission.SEEDS_VERIFY, True),
        (Role.FARMER, Permission.FINANCE_CREATE, True),
        (Role.FARMER, Permission.FINANCE_READ_OWN, True),
        (Role.FARMER, Permission.FINANCE_READ_ALL, False),
        (Role.FARMER, Permission.FINANCE_DELETE, False),
        (Role.FARMER, Permission.SUPPLY_CHAIN_CREATE, True),
        (Role.FARMER, Permission.WHATSAPP_SUBSCRIBE, True),
        (Role.FARMER, Permission.WHATSAPP_WEBHOOK, False),
        (Role.VENDOR, Permission.SUPPLY_CHAIN_READ, True),
        (Role.VENDOR, Permission.SUPPLY_CHAIN_CREATE, True),
        (Role.VENDOR, Permission.QUALITY_ASSESS, False),
        (Role.VENDOR, Permission.FINANCE_CREATE, False),
        (Role.GUEST, Permission.RAG_QUERY, True),
        (Role.GUEST, Permission.CLIMATE_SIMULATE, True),
        (Role.GUEST, Permission.FINANCE_CREATE, False),
        (Role.GUEST, Permission.SUPPLY_CHAIN_CREATE, False),
        (Role.SYSTEM, Permission.SYSTEM_ADMIN, True),
        (Role.SYSTEM, Permission.WHATSAPP_WEBHOOK, True),
    ])
    def test_matrix_permissions(self, role, perm, expected):
        assert RBACMatrix.has_permission(role, perm) == expected

    def test_has_any_permission(self):
        assert RBACMatrix.has_any_permission(Role.FARMER, [Permission.FINANCE_CREATE, Permission.FINANCE_DELETE])
        assert not RBACMatrix.has_any_permission(Role.VENDOR, [Permission.FINANCE_CREATE, Permission.QUALITY_ASSESS])

    def test_has_all_permissions(self):
        assert RBACMatrix.has_all_permissions(Role.ADMIN, [Permission.FINANCE_CREATE, Permission.FINANCE_READ_ALL, Permission.FINANCE_DELETE])
        assert not RBACMatrix.has_all_permissions(Role.FARMER, [Permission.FINANCE_CREATE, Permission.FINANCE_READ_ALL])


# ---------------------------------------------------------------------------
# 5. Role Validation
# ---------------------------------------------------------------------------

class TestRoleValidation:
    def test_valid_required_roles_passes(self):
        validate_required_roles([Role.ADMIN], [Role.ADMIN])

    def test_invalid_required_roles_fails(self):
        with pytest.raises(ValueError):
            validate_required_roles([Role.ADMIN], [Role.FARMER])

    def test_empty_roles_raises(self):
        with pytest.raises(ValueError):
            validate_required_roles([], [])

    def test_validate_with_audit_log(self):
        rbac_audit_trail.record(
            event_type="authorization_check",
            user_id="test-user",
            user_role="farmer",
            required_roles=["admin"],
            granted=False,
            reason="insufficient_permissions",
        )
        snapshot = rbac_audit_trail.snapshot(limit=10)
        assert len(snapshot) >= 1
        assert snapshot[0]["user_role"] == "farmer"
        assert snapshot[0]["granted"] is False
