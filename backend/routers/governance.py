"""ML Governance Router - Drift, shadow eval, versioning"""
from fastapi import APIRouter, HTTPException, Request, Query
from firebase_admin import auth as firebase_auth
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

router = APIRouter()

class RegisterModelVersionRequest(BaseModel):
    model_name: str = Field(..., min_length=1, max_length=50)
    model_path: str = Field(..., min_length=1)
    rmse: float = Field(..., gt=0)
    r2_score: float = Field(default=0, ge=-1, le=1)
    metadata: Optional[Dict[str, Any]] = None

drift_detector = None
shadow_evaluator = None
version_manager = None

def init_governance(dd, se, vm):
    global drift_detector, shadow_evaluator, version_manager
    drift_detector = dd
    shadow_evaluator = se
    version_manager = vm


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def _require_auth(request: Request) -> str:
    """
    Verify the Firebase ID token from the Authorization header.
    Returns the caller's uid on success.
    Raises HTTP 401 if the token is missing or invalid.

    All governance endpoints require authentication — unauthenticated callers
    must not be able to promote/rollback models, poison drift baselines, or
    start shadow evaluations.
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authentication token")
    token = auth_header.split(" ", 1)[1]
    try:
        decoded = firebase_auth.verify_id_token(token, check_revoked=True)
        return decoded["uid"]
    except firebase_auth.RevokedIdTokenError:
        raise HTTPException(status_code=401, detail="Session revoked — please log in again")
    except Exception:
        raise HTTPException(status_code=401, detail="Authentication failed")


def _require_admin_auth(request: Request) -> str:
    """
    Verify the Firebase ID token AND check that the caller has the 'admin'
    or 'expert' role in Firestore. Used for destructive write operations
    (promote, rollback, register, set baseline, start shadow eval) that must
    be restricted to privileged roles.

    Fail-closed design: any error reaching Firestore — transient network
    failure, SDK exception, or unavailable service — results in HTTP 503
    rather than silently granting access. A Firestore outage must never
    become a privilege-escalation window.
    """
    uid = _require_auth(request)

    import firebase_admin
    from firebase_admin import firestore as _fs

    # Firestore must be initialised; if not, the authorization service is
    # unavailable and we must reject rather than allow through.
    if not firebase_admin._apps:
        raise HTTPException(
            status_code=503,
            detail="Authorization service unavailable"
        )

    try:
        db = _fs.client()
        user_doc = db.collection("users").document(uid).get()
    except HTTPException:
        raise
    except Exception:
        # Any Firestore error (timeout, network reset, SDK fault) is treated
        # as an authorization-service failure. Fail closed — do not grant
        # access when the role cannot be verified.
        raise HTTPException(
            status_code=503,
            detail="Authorization service unavailable"
        )

    if not user_doc.exists:
        raise HTTPException(status_code=403, detail="User profile not found")

    role = user_doc.to_dict().get("role", "farmer")
    if role not in ("admin", "expert"):
        raise HTTPException(
            status_code=403,
            detail="Access denied: admin or expert role required"
        )

    return uid


# ---------------------------------------------------------------------------
# Drift detection endpoints
# ---------------------------------------------------------------------------

@router.post("/drift/baseline")
async def set_drift_baseline(request: Request, model_name: str, predictions: list[float]):
    """Set drift baseline. Requires admin or expert role."""
    _require_admin_auth(request)
    if drift_detector is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    drift_detector.set_baseline(model_name, predictions)
    return {"success": True, "message": f"Baseline set for {model_name}"}

@router.post("/drift/check")
async def check_drift(request: Request, model_name: str, prediction: float, actual_value: float):
    """Check for model drift. Requires authentication."""
    _require_auth(request)
    if drift_detector is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    drift_info = drift_detector.check_prediction_drift(model_name, prediction, actual_value)
    return {"success": True, "drift": drift_info}

@router.get("/drift/alerts")
async def get_drift_alerts(request: Request, model_name: str = None, limit: int = Query(10, ge=1, le=100)):
    """Get drift alerts. Requires authentication."""
    _require_auth(request)
    if drift_detector is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    alerts = drift_detector.get_alerts(model_name) if model_name else []
    return {"success": True, "alerts": alerts[:limit]}


# ---------------------------------------------------------------------------
# Shadow evaluation endpoints
# ---------------------------------------------------------------------------

@router.post("/shadow/start")
async def start_shadow_evaluation(request: Request, production_model: str, candidate_model: str):
    """Start a shadow evaluation. Requires admin or expert role."""
    _require_admin_auth(request)
    if shadow_evaluator is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    eval_id = shadow_evaluator.start_shadow_evaluation(production_model, candidate_model)
    return {"success": True, "eval_id": eval_id}

@router.post("/shadow/record")
async def record_shadow_predictions(request: Request, eval_id: str, production_prediction: float, candidate_prediction: float, actual_value: float):
    """Record shadow predictions. Requires authentication."""
    _require_auth(request)
    if shadow_evaluator is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    shadow_evaluator.record_predictions(eval_id, production_prediction, candidate_prediction, actual_value)
    status = shadow_evaluator.get_evaluation_status(eval_id)
    return {"success": True, "eval_id": eval_id, "status": status}

@router.post("/shadow/evaluate")
async def evaluate_candidate_model(request: Request, eval_id: str):
    """Evaluate a candidate model. Requires admin or expert role."""
    _require_admin_auth(request)
    if shadow_evaluator is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    result = shadow_evaluator.evaluate_candidate(eval_id)
    return {"success": True, "result": result}

@router.get("/shadow/status/{eval_id}")
async def get_shadow_eval_status(request: Request, eval_id: str):
    """Get shadow evaluation status. Requires authentication."""
    _require_auth(request)
    if shadow_evaluator is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    status = shadow_evaluator.get_evaluation_status(eval_id)
    return {"success": True, "eval_id": eval_id, "status": status}


# ---------------------------------------------------------------------------
# Model version management endpoints
# ---------------------------------------------------------------------------

@router.post("/versions/register")
async def register_model_version(request: Request, data: RegisterModelVersionRequest):
    """Register a new model version. Requires admin or expert role."""
    _require_admin_auth(request)
    if version_manager is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    version_id = version_manager.register_version(data.model_name, data.model_path, data.rmse, data.r2_score, data.metadata)
    return {"success": True, "version_id": version_id}

@router.post("/versions/promote")
async def promote_model_version(request: Request, version_id: str):
    """Promote a model version to production. Requires admin role."""
    _require_admin_auth(request)
    if version_manager is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    version_manager.promote_version(version_id)
    prod_version = version_manager.get_production_version()
    return {"success": True, "production_version": prod_version}

@router.post("/versions/rollback")
async def rollback_model_version(request: Request, version_id: str):
    """Roll back to a previous model version. Requires admin role."""
    _require_admin_auth(request)
    if version_manager is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    version_manager.rollback_to_version(version_id)
    prod_version = version_manager.get_production_version()
    return {"success": True, "production_version": prod_version}

@router.get("/versions/production")
async def get_production_version(request: Request):
    """Get current production version. Requires authentication."""
    _require_auth(request)
    if version_manager is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    prod_version = version_manager.get_production_version()
    return {"success": True, "production_version": prod_version}

@router.get("/versions/list")
async def list_model_versions(request: Request, model_name: str = None):
    """List model versions. Requires authentication."""
    _require_auth(request)
    if version_manager is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    versions = version_manager.list_versions(model_name) if model_name else []
    return {"success": True, "versions": versions}

@router.get("/versions/compare")
async def compare_model_versions(request: Request, v1: str, v2: str):
    """Compare two model versions. Requires authentication."""
    _require_auth(request)
    if version_manager is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    comparison = version_manager.compare_versions(v1, v2)
    return {"success": True, "comparison": comparison}

@router.get("/status")
async def get_governance_status(request: Request):
    """Get overall governance status. Requires authentication."""
    _require_auth(request)
    if not all([drift_detector, shadow_evaluator, version_manager]):
        raise HTTPException(status_code=500, detail="Not fully initialized")
    return {
        "success": True,
        "governance_status": {
            "drift_alerts": len(drift_detector.get_alerts("all") if hasattr(drift_detector, 'alerts') else []),
            "active_evals": len(shadow_evaluator.active_evaluations if hasattr(shadow_evaluator, 'active_evaluations') else []),
            "total_versions": len(version_manager.versions if hasattr(version_manager, 'versions') else [])
        }
    }
