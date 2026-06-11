from __future__ import annotations

import asyncio

import pytest

from backend.routers import governance


def test_require_auth_rejects_token_without_uid(monkeypatch) -> None:
    async def verify(_request):
        return {"roles": ["farmer"]}

    monkeypatch.setattr(governance, "verify_role_fn", verify)

    with pytest.raises(governance.HTTPException) as exc:
        asyncio.run(governance._require_auth(object()))

    assert exc.value.status_code == 401
    assert exc.value.detail == "Invalid authentication token"


def test_require_admin_auth_rejects_token_without_uid(monkeypatch) -> None:
    async def verify(_request, required_roles=None):
        assert required_roles == ["admin", "expert"]
        return {"roles": ["admin"]}

    monkeypatch.setattr(governance, "verify_role_fn", verify)

    with pytest.raises(governance.HTTPException) as exc:
        asyncio.run(governance._require_admin_auth(object()))

    assert exc.value.status_code == 401
    assert exc.value.detail == "Invalid authentication token"


def test_require_auth_returns_uid(monkeypatch) -> None:
    async def verify(_request):
        return {"uid": "user-123", "roles": ["farmer"]}

    monkeypatch.setattr(governance, "verify_role_fn", verify)

    assert asyncio.run(governance._require_auth(object())) == "user-123"


def test_require_admin_auth_returns_uid(monkeypatch) -> None:
    async def verify(_request, required_roles=None):
        assert required_roles == ["admin", "expert"]
        return {"uid": "admin-123", "roles": ["admin"]}

    monkeypatch.setattr(governance, "verify_role_fn", verify)

    assert asyncio.run(governance._require_admin_auth(object())) == "admin-123"
