"""Finance Router"""
import logging
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from rbac_audit import audit_rbac_event

logger = logging.getLogger(__name__)

router = APIRouter()

class FinanceAssessmentRequest(BaseModel):
    farmer_name: str = Field(..., min_length=1, max_length=100)
    crop_type: str = Field(..., min_length=1, max_length=50)
    acreage: float = Field(default=0, ge=0)
    annual_revenue: float = Field(default=0, ge=0)
    annual_operating_cost: float = Field(default=0, ge=0)
    existing_debt: float = Field(default=0, ge=0)
    emergency_fund: float = Field(default=0, ge=0)
    credit_score: int = Field(default=650, ge=300, le=900)
    requested_loan_amount: float = Field(default=0, ge=0)
    loan_tenure_months: int = Field(default=36, ge=6, le=120)

farm_finance_ai = None
rbac_manager = None
Permission = None

def init_finance(ffa, rbac, perm):
    global farm_finance_ai, rbac_manager, Permission
    farm_finance_ai = ffa
    rbac_manager = rbac
    Permission = perm


def _extract_uid_from_verified_token(request: Request) -> Optional[str]:
    """
    Extract the Firebase UID by cryptographically verifying the JWT token.

    Performs full signature verification via ``firebase_admin.auth.verify_id_token``.
    This is safe to call regardless of whether upstream verification has occurred,
    because every invocation independently validates the token's signature, expiry,
    and issuer.
    """
    from firebase_admin import auth as firebase_auth

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return None
    token = auth_header.split(" ", 1)[1]
    try:
        decoded = firebase_auth.verify_id_token(token)
        return decoded.get("uid")
    except Exception as exc:
        logger.error("Failed to verify JWT token: %s", exc)
        return None


async def _has_permission(request: Request, permission) -> bool:
    """Return True if the caller has the given permission (no exception raised)."""
    try:
        await rbac_manager.raise_if_unauthorized(request, [permission], require_all=False)
        return True
    except HTTPException as e:
        if e.status_code == 403:
            return False
        raise


@router.post("/analyze")
async def analyze_farm_finance(request: Request, body: FinanceAssessmentRequest):
    if farm_finance_ai is None or rbac_manager is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    try:
        await rbac_manager.raise_if_unauthorized(request, [Permission.FINANCE_CREATE], require_all=False)
        analysis = farm_finance_ai.analyze_financial_profile(body.model_dump())
        return {"success": True, "data": analysis}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Financial analysis failed: %s", e)
        raise HTTPException(status_code=500, detail="Financial analysis failed")


@router.post("/applications")
async def create_finance_application(request: Request, body: FinanceAssessmentRequest):
    if farm_finance_ai is None or rbac_manager is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    try:
        await rbac_manager.raise_if_unauthorized(request, [Permission.FINANCE_CREATE], require_all=False)
        # Bind the application to the authenticated caller so ownership can be
        # enforced on subsequent reads.  The token was already verified by
        # raise_if_unauthorized above; decode the payload without a second
        # cryptographic verification to avoid duplicate latency.
        owner_uid = _extract_uid_from_verified_token(request)
        if owner_uid is None:
            # raise_if_unauthorized passed, so the token is valid — a missing
            # uid here indicates a malformed token that should not proceed.
            raise HTTPException(status_code=401, detail="Unable to determine caller identity")
        application = farm_finance_ai.create_application(body.model_dump(), owner_uid=owner_uid)
        return {"success": True, "data": application}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Application creation failed: %s", e)
        raise HTTPException(status_code=500, detail="Application creation failed")


@router.get("/applications/{application_id}")
async def get_finance_application(application_id: str, request: Request, resource_tenant_id: Optional[str] = None):
    """
    Retrieve a single finance application.

    Ownership enforcement:
    - Farmers (FINANCE_READ_OWN): only their own application is returned.
      Any other application_id returns 404, preventing IDOR enumeration.
    - Admins / experts (FINANCE_READ_ALL): ownership check is skipped so
      they can review any application for underwriting or support purposes.
    """
    if farm_finance_ai is None or rbac_manager is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    try:
        # Require at least one of the two read permissions
        await rbac_manager.raise_if_unauthorized(
            request,
            [Permission.FINANCE_READ_OWN, Permission.FINANCE_READ_ALL],
            require_all=False,
        )

        caller_uid = _extract_uid_from_verified_token(request)
        ctx = await rbac_manager.resolve_auth_context(request, allow_unauthenticated=False)

        # Admins/experts with FINANCE_READ_ALL can override ownership in-tenant.
        # Farmers with only FINANCE_READ_OWN remain scoped to their own records.
        has_read_all = await _has_permission(request, Permission.FINANCE_READ_ALL)
        owner_uid_filter = caller_uid

        if has_read_all:
            try:
                can_override = rbac_manager.can_admin_or_expert_override(
                    ctx,
                    resource_owner_uid=None,
                    resource_tenant_id=resource_tenant_id,
                    allow_cross_tenant=False,
                )
            except Exception:
                can_override = False

            if can_override:
                owner_uid_filter = None
                audit_rbac_event(
                    request=request,
                    action=f"GET /api/finance/applications/{application_id}",
                    outcome="allowed",
                    uid=ctx.uid,
                    role=ctx.role,
                    required_roles=["admin", "expert"],
                    reason="admin_expert_override",
                    status_code=200,
                )
            else:
                audit_rbac_event(
                    request=request,
                    action=f"GET /api/finance/applications/{application_id}",
                    outcome="denied",
                    uid=ctx.uid,
                    role=ctx.role,
                    required_roles=["admin", "expert"],
                    reason="cross_tenant_override_denied",
                    status_code=403,
                )
                raise HTTPException(status_code=403, detail="Access denied: cross-tenant override not permitted")

        application = farm_finance_ai.get_application(
            application_id, owner_uid=owner_uid_filter
        )
        if not application:
            raise HTTPException(status_code=404, detail="Application not found")
        return {"success": True, "data": application}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Application retrieval failed: %s", e)
        raise HTTPException(status_code=500, detail="Application retrieval failed")


@router.get("/products")
async def get_finance_products():
    if farm_finance_ai is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    return {"success": True, "data": farm_finance_ai.list_marketplace()}


# /marketplace is kept as an alias for /products so existing frontend
# integrations that call either path continue to work without changes.
# Both routes delegate to the same handler — there is a single code path
# and a single place to update if the response shape ever changes.
@router.get("/marketplace")
async def get_finance_marketplace():
    return await get_finance_products()
