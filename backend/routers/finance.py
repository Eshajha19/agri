"""Finance Router"""
from fastapi import APIRouter, Request, HTTPException
from firebase_admin import auth as firebase_auth
from pydantic import BaseModel, Field
from typing import Optional

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


def _extract_uid(request: Request) -> Optional[str]:
    """
    Extract and verify the Firebase UID from the Authorization header.
    Returns None if the token is missing or invalid (caller should already
    have been rejected by raise_if_unauthorized before this is called).
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return None
    token = auth_header.split(" ", 1)[1]
    try:
        decoded = firebase_auth.verify_id_token(token)
        return decoded.get("uid")
    except Exception:
        return None


async def _has_permission(request: Request, permission) -> bool:
    """Return True if the caller has the given permission (no exception raised)."""
    try:
        await rbac_manager.raise_if_unauthorized(request, [permission], require_all=False)
        return True
    except Exception:
        return False


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
        raise HTTPException(status_code=403, detail=str(e))

@router.post("/applications")
async def create_finance_application(request: Request, body: FinanceAssessmentRequest):
    if farm_finance_ai is None or rbac_manager is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    try:
        await rbac_manager.raise_if_unauthorized(request, [Permission.FINANCE_CREATE], require_all=False)
        # Bind the application to the authenticated caller so ownership can be
        # enforced on subsequent reads.
        owner_uid = _extract_uid(request)
        application = farm_finance_ai.create_application(body.model_dump(), owner_uid=owner_uid)
        return {"success": True, "data": application}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=403, detail=str(e))

@router.get("/applications/{application_id}")
async def get_finance_application(application_id: str, request: Request):
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

        caller_uid = _extract_uid(request)

        # Admins/experts with FINANCE_READ_ALL bypass the ownership filter;
        # farmers with only FINANCE_READ_OWN are scoped to their own records.
        has_read_all = await _has_permission(request, Permission.FINANCE_READ_ALL)
        owner_uid_filter = None if has_read_all else caller_uid

        application = farm_finance_ai.get_application(
            application_id, owner_uid=owner_uid_filter
        )
        if not application:
            raise HTTPException(status_code=404, detail="Application not found")
        return {"success": True, "data": application}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=403, detail=str(e))

@router.get("/products")
def get_finance_products():
    if farm_finance_ai is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    return {"success": True, "data": farm_finance_ai.list_marketplace()}

@router.get("/marketplace")
def get_finance_marketplace():
    if farm_finance_ai is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    return {"success": True, "data": farm_finance_ai.list_marketplace()}
