import pytest
from fastapi import Request

from backend.routers import finance
from rbac import AuthContext, Permission


def _request() -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/finance/applications",
        "headers": [(b"authorization", b"Bearer test-token")],
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(scope, receive)


def _body() -> finance.FinanceAssessmentRequest:
    return finance.FinanceAssessmentRequest(
        farmer_name="Meera",
        crop_type="rice",
        acreage=8,
        annual_revenue=1200000,
        annual_operating_cost=700000,
        existing_debt=90000,
        emergency_fund=200000,
        credit_score=770,
        requested_loan_amount=320000,
        loan_tenure_months=36,
    )


class _FakeRBAC:
    def __init__(self):
        self.resolve_calls = 0

    async def resolve_auth_context(self, request, *, allow_unauthenticated=False):
        self.resolve_calls += 1
        return AuthContext(uid="farmer-123", role="farmer")

    async def raise_if_unauthorized(self, *args, **kwargs):
        raise AssertionError("finance handler should use the resolved auth context")

    def can_admin_or_expert_override(self, *args, **kwargs):
        return False


class _FakeFinanceAI:
    def __init__(self):
        self.created_owner_uid = None
        self.read_owner_uid = None

    def create_application(self, payload, owner_uid=None):
        self.created_owner_uid = owner_uid
        return {"application_id": "app-1", "owner_uid": owner_uid}

    def get_application(self, application_id, owner_uid=None):
        self.read_owner_uid = owner_uid
        return {"application_id": application_id, "owner_uid": owner_uid}


@pytest.mark.asyncio
async def test_create_application_uses_resolved_auth_context_uid():
    engine = _FakeFinanceAI()
    rbac = _FakeRBAC()
    finance.init_finance(engine, rbac, Permission)

    response = await finance.create_finance_application(_request(), _body())

    assert response["success"] is True
    assert engine.created_owner_uid == "farmer-123"
    assert rbac.resolve_calls == 1


@pytest.mark.asyncio
async def test_get_application_uses_resolved_auth_context_uid():
    engine = _FakeFinanceAI()
    rbac = _FakeRBAC()
    finance.init_finance(engine, rbac, Permission)

    response = await finance.get_finance_application("app-1", _request())

    assert response["success"] is True
    assert engine.read_owner_uid == "farmer-123"
    assert rbac.resolve_calls == 1
