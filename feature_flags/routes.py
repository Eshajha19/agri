"""
routes.py
──────────
FastAPI router for the Feature Flag & A/B Testing API.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from feature_flags import experiment_engine, flag_store, metrics_collector
from feature_flags.ab_testing_runner import get_runner

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["feature-flags"])

verify_role_fn: Optional[Callable] = None


def init_feature_flags(verify_role) -> None:
    global verify_role_fn
    verify_role_fn = verify_role


async def _require_admin(request: Request) -> None:
    if verify_role_fn is None:
        raise HTTPException(status_code=500, detail="Feature flags API not initialized")
    await verify_role_fn(request, required_roles=["admin"])


async def _require_authenticated(request: Request) -> dict:
    if verify_role_fn is None:
        raise HTTPException(status_code=500, detail="Feature flags API not initialized")
    return await verify_role_fn(request)


class FlagUpsertRequest(BaseModel):
    enabled:     bool = False
    rollout_pct: int  = Field(0, ge=0, le=100)
    cohorts:     List[str] = []
    description: str = ""
    owner:       str = ""
    tags:        List[str] = []


class ExperimentVariant(BaseModel):
    id:     str
    name:   str
    weight: int = Field(50, ge=0, le=100)


class ExperimentCreateRequest(BaseModel):
    id:          Optional[str] = None
    name:        str
    description: str = ""
    status:      str = Field("draft", pattern="^(draft|running|paused|completed)$")
    variants:    List[ExperimentVariant] = []
    start_date:  Optional[str] = None
    end_date:    Optional[str] = None
    owner:       str = ""
    tags:        List[str] = []


class AssignRequest(BaseModel):
    user_id:       str
    experiment_id: str


class StatusUpdateRequest(BaseModel):
    status: str = Field(..., pattern="^(draft|running|paused|completed)$")


class EventRequest(BaseModel):
    event_type:    str = Field("impression",
                               pattern="^(impression|conversion|error|flag_evaluated|custom)$")
    user_id:       str
    experiment_id: Optional[str] = None
    variant:       Optional[str] = None
    flag_id:       Optional[str] = None
    metadata:      Dict[str, Any] = {}
    session_id:    Optional[str] = None


class EventBatchRequest(BaseModel):
    events: List[EventRequest] = Field(..., max_items=100)


class TrafficSplitVariant(BaseModel):
    id: str
    weight: int = Field(..., ge=0, le=100)


class TrafficSplitRequest(BaseModel):
    variants: List[TrafficSplitVariant] = Field(..., min_items=2)


@router.get("/flags", summary="List all feature flags (public read)")
def list_flags():
    return {"flags": flag_store.list_flags()}


@router.get("/flags/{flag_id}", summary="Get a single feature flag (public read)")
def get_flag(flag_id: str):
    flag = flag_store.get_flag(flag_id)
    if not flag:
        raise HTTPException(status_code=404, detail=f"Flag '{flag_id}' not found")
    return flag


@router.post("/flags/{flag_id}", summary="Create or update a feature flag (admin)")
async def upsert_flag(request: Request, flag_id: str, body: FlagUpsertRequest):
    await _require_admin(request)
    updated = flag_store.upsert_flag(flag_id, body.dict())
    return {"success": True, "flag": updated}


@router.delete("/flags/{flag_id}", summary="Delete a feature flag (admin)")
async def delete_flag(request: Request, flag_id: str):
    await _require_admin(request)
    deleted = flag_store.delete_flag(flag_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Flag '{flag_id}' not found")
    return {"success": True, "deleted": flag_id}


@router.post("/flags/{flag_id}/rollback", summary="Rollback flag (admin)")
async def rollback_flag(request: Request, flag_id: str):
    await _require_admin(request)
    result = flag_store.rollback_flag(flag_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Flag '{flag_id}' not found")
    return {"success": True, "flag": result}


@router.get("/experiments", summary="List all experiments (admin)")
async def list_experiments(request: Request):
    await _require_admin(request)
    return {"experiments": experiment_engine.list_experiments()}


@router.post("/experiments", summary="Create a new experiment (admin)")
async def create_experiment(request: Request, body: ExperimentCreateRequest):
    await _require_admin(request)
    data = body.dict()
    data["variants"] = [v.dict() for v in body.variants]
    exp = experiment_engine.create_experiment(data)
    return {"success": True, "experiment": exp}


@router.post("/experiments/assign", summary="Assign user to experiment variant (authenticated)")
async def assign_user(request: Request, body: AssignRequest):
    await _require_authenticated(request)
    assignment = get_runner(body.experiment_id).record_assignment(body.user_id)
    return assignment


@router.patch("/experiments/{exp_id}/status", summary="Update experiment status (admin)")
async def update_status(request: Request, exp_id: str, body: StatusUpdateRequest):
    await _require_admin(request)
    result = experiment_engine.update_experiment_status(exp_id, body.status)
    if not result:
        raise HTTPException(status_code=404, detail=f"Experiment '{exp_id}' not found")
    return {"success": True, "experiment": result}


@router.get("/experiments/{exp_id}/metrics", summary="Get experiment metrics (admin)")
async def get_metrics(request: Request, exp_id: str):
    await _require_admin(request)
    return metrics_collector.get_experiment_metrics(exp_id)


@router.post("/experiments/{exp_id}/traffic-split", summary="Update experiment traffic split (admin)")
async def update_traffic_split(request: Request, exp_id: str, body: TrafficSplitRequest):
    await _require_admin(request)
    runner = get_runner(exp_id)
    updated = runner.set_traffic_split({variant.id: variant.weight for variant in body.variants})
    return {"success": True, "experiment": updated}


@router.post("/experiments/{exp_id}/evaluate", summary="Evaluate and auto-promote a winner (admin)")
async def evaluate_experiment(request: Request, exp_id: str):
    await _require_admin(request)
    runner = get_runner(exp_id)
    decision = runner.process()
    return {
        "success": True,
        "decision": {
            "experiment_id": decision.experiment_id,
            "winner_variant": decision.winner_variant,
            "runner_up_variant": decision.runner_up_variant,
            "promoted": decision.promoted,
            "reason": decision.reason,
        },
        "metrics": decision.metrics,
    }


@router.post("/experiments/events", summary="Log a single experiment event (authenticated)")
async def log_event(request: Request, body: EventRequest):
    token_data = await _require_authenticated(request)
    uid = token_data.get("uid")
    if not uid:
        raise HTTPException(status_code=401, detail="User identity missing from authentication token")
    event = metrics_collector.log_event(
        event_type=body.event_type,
        user_id=uid,
        experiment_id=body.experiment_id,
        variant=body.variant,
        flag_id=body.flag_id,
        metadata=body.metadata,
        session_id=body.session_id,
    )
    return {"success": True, "event": event}


@router.post("/experiments/events/batch", summary="Batch-log experiment events (admin)")
async def log_events_batch(request: Request, body: EventBatchRequest):
    await _require_admin(request)
    raw = [e.dict() for e in body.events]
    count = metrics_collector.log_events_batch(raw)
    return {"success": True, "logged": count}