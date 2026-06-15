"""Finance Router"""
import base64
import json
import logging
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

logger = logging.getLogger(__name__)

router = APIRouter()

# Tenure bounds are imported from the engine so the Pydantic schema and the
# business-logic layer always stay in sync from a single source of truth.
try:
    from farm_finance_ai import MIN_TENURE_MONTHS, MAX_TENURE_MONTHS
except ImportError:
    MIN_TENURE_MONTHS = 6
    MAX_TENURE_MONTHS = 360

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
    loan_tenure_months: int = Field(
        default=36,
        ge=MIN_TENURE_MONTHS,
        le=MAX_TENURE_MONTHS,
        description=(
            f"Repayment tenure in months. "
            f"Must be between {MIN_TENURE_MONTHS} and {MAX_TENURE_MONTHS} months. "
            f"Values outside this range will be rejected with HTTP 422."
        ),
    )

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
    Extract the Firebase UID from the JWT payload without performing a second
    cryptographic verification.

    This must only be called after ``rbac_manager.raise_if_unauthorized`` has
    already verified the token's signature and expiry for the current request.
    Decoding the payload here is safe because the token has already been
    authenticated; re-running ``firebase_auth.verify_id_token`` a second time
    would duplicate the signature check and a potential network round-trip to
    fetch Google's public keys.

    Returns None only if the Authorization header is absent or malformed —
    conditions that ``raise_if_unauthorized`` would have already rejected.
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return None
    token = auth_header.split(" ", 1)[1]
    try:
        # A Firebase/Google JWT has three base64url-encoded segments separated
        # by dots: header.payload.signature.  We only need the payload.
        parts = token.split(".")
        if len(parts) != 3:
            return None
        # base64url decode with padding correction
        payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        return payload.get("uid") or payload.get("sub")
    except Exception as exc:
        # Should never happen for a token that already passed verification,
        # but log and surface None so callers can handle it explicitly.
        logger.error("Failed to decode already-verified JWT payload: %s", exc)
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
    except ValueError as exc:
        # Explicit validation errors from the engine (e.g. invalid tenure)
        # are surfaced as 422 Unprocessable Entity so the caller can correct
        # their input rather than receiving a silent data modification.
        logger.warning("Finance analyze validation error: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as e:
        raise HTTPException(status_code=403, detail=str(e))

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
            raise HTTPException(status_code=401, detail="Unable to determine caller identity")
        application = farm_finance_ai.create_application(body.model_dump(), owner_uid=owner_uid)
        return {"success": True, "data": application}
    except HTTPException:
        raise
    except ValueError as exc:
        # Validation errors (e.g. invalid tenure) → 422 so callers get a
        # transparent, actionable error instead of a silent data change.
        logger.warning("Finance application validation error: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc))
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

        caller_uid = _extract_uid_from_verified_token(request)

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
