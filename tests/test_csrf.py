import pytest
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient

import csrf_protection
from rbac import AuthContext, RBACManager


def test_csrf_token_generation_and_validation():
    uid = "test-user-123"
    token = csrf_protection.generate_token(uid)
    assert token is not None
    assert csrf_protection.validate_token(token, uid) is True
    assert csrf_protection.validate_token(token, "other-user") is False
    assert csrf_protection.validate_token("invalid|token|format", uid) is False


@pytest.mark.anyio
async def test_csrf_dependency_unauthenticated(monkeypatch):
    async def fake_resolve(req, allow_unauthenticated=False):
        return None
    monkeypatch.setattr(
        RBACManager,
        "resolve_auth_context",
        fake_resolve,
    )

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/whatsapp/subscribe",
        "headers": [],
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    request = Request(scope, receive)
    # Should not raise any exception because the user is not authenticated
    await csrf_protection.verify_csrf_token_dependency(request)


@pytest.mark.anyio
async def test_csrf_dependency_authenticated_missing_token(monkeypatch):
    async def fake_resolve(req, allow_unauthenticated=False):
        return AuthContext(uid="authenticated-user", role="farmer")
    monkeypatch.setattr(
        RBACManager,
        "resolve_auth_context",
        fake_resolve,
    )

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/whatsapp/subscribe",
        "headers": [],  # Missing X-CSRF-Token header
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    request = Request(scope, receive)
    with pytest.raises(HTTPException) as exc:
        await csrf_protection.verify_csrf_token_dependency(request)
    assert exc.value.status_code == 403
    assert "Missing CSRF token" in exc.value.detail


@pytest.mark.anyio
async def test_csrf_dependency_authenticated_invalid_token(monkeypatch):
    async def fake_resolve(req, allow_unauthenticated=False):
        return AuthContext(uid="authenticated-user", role="farmer")
    monkeypatch.setattr(
        RBACManager,
        "resolve_auth_context",
        fake_resolve,
    )

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/whatsapp/subscribe",
        "headers": [(b"x-csrf-token", b"bad-token")],
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    request = Request(scope, receive)
    with pytest.raises(HTTPException) as exc:
        await csrf_protection.verify_csrf_token_dependency(request)
    assert exc.value.status_code == 403
    assert "Invalid or expired CSRF token" in exc.value.detail


@pytest.mark.anyio
async def test_csrf_dependency_authenticated_valid_token(monkeypatch):
    uid = "authenticated-user"
    valid_token = csrf_protection.generate_token(uid)

    async def fake_resolve(req, allow_unauthenticated=False):
        return AuthContext(uid=uid, role="farmer")
    monkeypatch.setattr(
        RBACManager,
        "resolve_auth_context",
        fake_resolve,
    )

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/whatsapp/subscribe",
        "headers": [(b"x-csrf-token", valid_token.encode())],
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    request = Request(scope, receive)
    # Should pass without raising any exception
    await csrf_protection.verify_csrf_token_dependency(request)
