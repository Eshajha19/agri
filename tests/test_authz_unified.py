"""
Unified authorization tests (issue #1130).

Firestore users/{uid}.role is authoritative; JWT claims must match or the API
returns 403; missing profiles never map to guest when a Bearer token is present.
"""

import pytest
from fastapi import Request

from rbac import AuthContext, RBACManager, Role, STALE_TOKEN_DETAIL


def _make_fake_firestore(roles):
    class _FakeDoc:
        def __init__(self, role):
            self.exists = True
            self._role = role

        def to_dict(self):
            return {"role": self._role}

    class _FakeUserRef:
        def __init__(self, uid):
            self._uid = uid

        def get(self):
            role = roles.get(self._uid)
            if role is None:
                return type("Missing", (), {"exists": False})()
            return _FakeDoc(role)

    class _FakeCollection:
        def document(self, uid):
            return _FakeUserRef(uid)

    class _FakeFirestore:
        def collection(self, name):
            return _FakeCollection()

    return _FakeFirestore()


def _request_with_token(token: str) -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/test",
        "headers": [(b"authorization", f"Bearer {token}".encode())],
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(scope, receive)


@pytest.fixture()
def patched_auth(monkeypatch):
    store = _make_fake_firestore(
        {
            "farmer-user": "farmer",
            "admin-user": "admin",
        }
    )

    def verify(token):
        if token == "stale-admin-token":
            return {"uid": "farmer-user", "role": "admin"}
        if token == "farmer-user":
            return {"uid": "farmer-user", "role": "farmer"}
        if token == "admin-user":
            return {"uid": "admin-user", "role": "admin"}
        if token == "orphan-user":
            return {"uid": "orphan-user", "role": "farmer"}
        return {"uid": token, "role": "farmer"}

    monkeypatch.setattr(RBACManager, "get_db", staticmethod(lambda: store))
    monkeypatch.setattr("rbac.firebase_auth.verify_id_token", verify)
    return store


@pytest.mark.asyncio
async def test_resolve_auth_context_allows_anonymous(patched_auth):
    scope = {"type": "http", "method": "GET", "path": "/", "headers": []}

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    request = Request(scope, receive)
    assert await RBACManager.resolve_auth_context(request, allow_unauthenticated=True) is None


@pytest.mark.asyncio
async def test_missing_profile_fails_closed_not_guest(patched_auth):
    request = _request_with_token("orphan-user")
    with pytest.raises(Exception) as exc:
        await RBACManager.resolve_auth_context(request, allow_unauthenticated=False)
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_stale_jwt_claim_rejected(patched_auth):
    request = _request_with_token("stale-admin-token")
    with pytest.raises(Exception) as exc:
        await RBACManager.resolve_auth_context(request, allow_unauthenticated=False)
    assert exc.value.status_code == 403
    assert STALE_TOKEN_DETAIL in str(exc.value.detail)


@pytest.mark.asyncio
async def test_matching_claim_and_firestore_succeeds(patched_auth):
    request = _request_with_token("farmer-user")
    ctx = await RBACManager.resolve_auth_context(request, allow_unauthenticated=False)
    assert isinstance(ctx, AuthContext)
    assert ctx.uid == "farmer-user"
    assert ctx.role == "farmer"


@pytest.mark.asyncio
async def test_get_user_role_returns_guest_only_without_token(patched_auth):
    scope = {"type": "http", "method": "GET", "path": "/", "headers": []}

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    request = Request(scope, receive)
    assert await RBACManager.get_user_role(request) == Role.GUEST


@pytest.mark.asyncio
async def test_raise_if_unauthorized_requires_authentication(patched_auth):
    from rbac import Permission

    scope = {"type": "http", "method": "GET", "path": "/", "headers": []}

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    request = Request(scope, receive)
    with pytest.raises(Exception) as exc:
        await RBACManager.raise_if_unauthorized(
            request,
            [Permission.FINANCE_CREATE],
        )
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_sync_role_claim_revokes_sessions_by_default(monkeypatch):
    revoked = []

    monkeypatch.setattr("role_sync._set_claim_sync", lambda uid, role: None)
    monkeypatch.setattr(
        "role_sync._revoke_refresh_tokens_sync",
        lambda uid: revoked.append(uid),
    )

    from role_sync import sync_role_claim

    await sync_role_claim("user-1", "farmer")
    assert revoked == ["user-1"]
