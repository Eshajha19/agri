"""
routes.py
──────────
FastAPI router for the Feature Flag & A/B Testing API.

All routes are prefixed with /api — register via:
    app.include_router(flags_router)

Endpoints
─────────
Feature Flags
  GET    /api/flags                  List all flags
  GET    /api/flags/{flag_id}        Get single flag
  POST   /api/flags                  Create or update a flag
  DELETE /api/flags/{flag_id}        Delete a flag
  POST   /api/flags/{flag_id}/rollback  Disable flag, reset rollout to 0%

Experiments
  GET    /api/experiments                     List all experiments
  POST   /api/experiments                     Create experiment
  POST   /api/experiments/assign              Assign user to variant
  PATCH  /api/experiments/{exp_id}/status     Update experiment status
  GET    /api/experiments/{exp_id}/metrics    Get aggregated metrics

Analytics
  POST   /api/experiments/events              Log single event
  POST   /api/experiments/events/batch        Batch-log events
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from feature_flags import flag_store, experiment_engine, metrics_collector

logger = logging.getLogger(__name__)

router = APIRouter(tags=["feature-flags"])


# ── Pydantic schemas ───────────────────────────────────────────────────────────

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


# ── Feature Flag endpoints ─────────────────────────────────────────────────────

@router.get("/flags", summary="List all feature flags")
def list_flags():
    return {"flags": flag_store.list_flags()}


@router.get("/flags/{flag_id}", summary="Get a single feature flag")
def get_flag(flag_id: str):
    flag = flag_store.get_flag(flag_id)
    if not flag:
        raise HTTPException(status_code=404, detail=f"Flag '{flag_id}' not found")
    return flag


@router.post("/flags/{flag_id}", summary="Create or update a feature flag")
def upsert_flag(flag_id: str, body: FlagUpsertRequest):
    updated = flag_store.upsert_flag(flag_id, body.dict())
    return {"success": True, "flag": updated}


@router.delete("/flags/{flag_id}", summary="Delete a feature flag")
def delete_flag(flag_id: str):
    deleted = flag_store.delete_flag(flag_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Flag '{flag_id}' not found")
    return {"success": True, "deleted": flag_id}


@router.post("/flags/{flag_id}/rollback", summary="Rollback — disable flag, reset rollout to 0%")
def rollback_flag(flag_id: str):
    result = flag_store.rollback_flag(flag_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Flag '{flag_id}' not found")
    return {"success": True, "flag": result}


# ── Experiment endpoints ───────────────────────────────────────────────────────

@router.get("/experiments", summary="List all experiments")
def list_experiments():
    return {"experiments": experiment_engine.list_experiments()}


@router.post("/experiments", summary="Create a new experiment")
def create_experiment(body: ExperimentCreateRequest):
    data = body.dict()
    data["variants"] = [v.dict() for v in body.variants]
    exp = experiment_engine.create_experiment(data)
    return {"success": True, "experiment": exp}


@router.post("/experiments/assign", summary="Assign user to experiment variant")
def assign_user(body: AssignRequest):
    assignment = experiment_engine.assign_user(body.user_id, body.experiment_id)
    return assignment


@router.patch("/experiments/{exp_id}/status", summary="Update experiment status")
def update_status(exp_id: str, body: StatusUpdateRequest):
    result = experiment_engine.update_experiment_status(exp_id, body.status)
    if not result:
        raise HTTPException(status_code=404, detail=f"Experiment '{exp_id}' not found")
    return {"success": True, "experiment": result}


@router.get("/experiments/{exp_id}/metrics", summary="Get aggregated experiment metrics")
def get_metrics(exp_id: str):
    return metrics_collector.get_experiment_metrics(exp_id)


# ── Analytics endpoints ────────────────────────────────────────────────────────

@router.post("/experiments/events", summary="Log a single experiment event")
def log_event(body: EventRequest):
    event = metrics_collector.log_event(
        event_type=body.event_type,
        user_id=body.user_id,
        experiment_id=body.experiment_id,
        variant=body.variant,
        flag_id=body.flag_id,
        metadata=body.metadata,
        session_id=body.session_id,
    )
    return {"success": True, "event": event}


@router.post("/experiments/events/batch", summary="Batch-log experiment events")
def log_events_batch(body: EventBatchRequest):
    raw = [e.dict() for e in body.events]
    count = metrics_collector.log_events_batch(raw)
    return {"success": True, "logged": count}
