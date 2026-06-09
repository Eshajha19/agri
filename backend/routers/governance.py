"""ML Governance Router - Drift, shadow eval, versioning"""
from fastapi import APIRouter, HTTPException, Request, Query
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
verify_role_fn = None

def init_governance(dd, se, vm, auth_fn=None):
    global drift_detector, shadow_evaluator, version_manager, verify_role_fn
    drift_detector = dd
    shadow_evaluator = se
    version_manager = vm
    verify_role_fn = auth_fn


# ---------------------------------------------------------------------------
# Drift detection endpoints
# ---------------------------------------------------------------------------

@router.post("/drift/baseline")
async def set_drift_baseline(request: Request, model_name: str, predictions: list[float]):
    """Set drift baseline. Requires admin or expert role."""
    await _enforce_role(request, ["admin", "expert"])
    if drift_detector is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    drift_detector.set_baseline(model_name, predictions)
    return {"success": True, "message": f"Baseline set for {model_name}"}

@router.post("/drift/check")
async def check_drift(request: Request, model_name: str, prediction: float, actual_value: float):
    """Check for model drift. Requires authentication."""
    await _enforce_role(request)
    if drift_detector is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    drift_info = drift_detector.check_prediction_drift(model_name, prediction, actual_value)
    return {"success": True, "drift": drift_info}

@router.get("/drift/alerts")
async def get_drift_alerts(request: Request, model_name: str = None, limit: int = Query(10, ge=1, le=100)):
    """Get drift alerts. Requires authentication."""
    await _enforce_role(request)
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
    await _enforce_role(request, ["admin", "expert"])
    if shadow_evaluator is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    eval_id = shadow_evaluator.start_shadow_evaluation(production_model, candidate_model)
    return {"success": True, "eval_id": eval_id}

@router.post("/shadow/record")
async def record_shadow_predictions(request: Request, eval_id: str, production_prediction: float, candidate_prediction: float, actual_value: float):
    """Record shadow predictions. Requires authentication."""
    await _enforce_role(request)
    if shadow_evaluator is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    shadow_evaluator.record_predictions(eval_id, production_prediction, candidate_prediction, actual_value)
    status = shadow_evaluator.get_evaluation_status(eval_id)
    return {"success": True, "eval_id": eval_id, "status": status}

@router.post("/shadow/evaluate")
async def evaluate_candidate_model(request: Request, eval_id: str):
    """Evaluate a candidate model. Requires admin or expert role."""
    await _enforce_role(request, ["admin", "expert"])
    if shadow_evaluator is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    result = shadow_evaluator.evaluate_candidate(eval_id)
    return {"success": True, "result": result}

@router.get("/shadow/status/{eval_id}")
async def get_shadow_eval_status(request: Request, eval_id: str):
    """Get shadow evaluation status. Requires authentication."""
    await _enforce_role(request)
    if shadow_evaluator is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    status = shadow_evaluator.get_evaluation_status(eval_id)
    return {"success": True, "eval_id": eval_id, "status": status}


# ---------------------------------------------------------------------------
# Model version management endpoints
# ---------------------------------------------------------------------------

async def _enforce_role(request: Request, required_roles: list[str] = None) -> None:
    """Verify the caller is authenticated (and optionally has a required role).

    Delegates to the project-wide ``verify_role_fn`` injected at startup via
    ``init_governance()``. Raises 401/403/503 on failure — never grants access
    when the auth service is unreachable.
    """
    fn = verify_role_fn
    if fn is None:
        raise HTTPException(status_code=503, detail="Auth service not initialized")
    await fn(request, required_roles=required_roles)


@router.post("/versions/register")
async def register_model_version(request: Request, data: RegisterModelVersionRequest):
    """Register a new model version. Requires admin or expert role."""
    await _enforce_role(request, ["admin", "expert"])
    if version_manager is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    version_id = version_manager.register_version(data.model_name, data.model_path, data.rmse, data.r2_score, data.metadata)
    return {"success": True, "version_id": version_id}

@router.post("/versions/promote")
async def promote_model_version(request: Request, version_id: str):
    """Promote a model version to production. Requires admin or expert role."""
    await _enforce_role(request, ["admin", "expert"])
    if version_manager is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    version_manager.promote_version(version_id)
    prod_version = version_manager.get_production_version()
    return {"success": True, "production_version": prod_version}

@router.post("/versions/rollback")
async def rollback_model_version(request: Request, version_id: str):
    """Roll back to a previous model version. Requires admin or expert role."""
    await _enforce_role(request, ["admin", "expert"])
    if version_manager is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    version_manager.rollback_to_version(version_id)
    prod_version = version_manager.get_production_version()
    return {"success": True, "production_version": prod_version}

@router.get("/versions/production")
async def get_production_version(request: Request):
    """Get current production version. Requires authentication."""
    await _enforce_role(request)
    if version_manager is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    prod_version = version_manager.get_production_version()
    return {"success": True, "production_version": prod_version}

@router.get("/versions/list")
async def list_model_versions(request: Request, model_name: str = None):
    """List model versions. Requires authentication."""
    await _enforce_role(request)
    if version_manager is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    versions = version_manager.list_versions(model_name) if model_name else []
    return {"success": True, "versions": versions}

@router.get("/versions/compare")
async def compare_model_versions(request: Request, v1: str, v2: str):
    """Compare two model versions. Requires authentication."""
    await _enforce_role(request)
    if version_manager is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    comparison = version_manager.compare_versions(v1, v2)
    return {"success": True, "comparison": comparison}

@router.get("/status")
async def get_governance_status(request: Request):
    """Get overall governance status. Requires authentication."""
    await _enforce_role(request)
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
