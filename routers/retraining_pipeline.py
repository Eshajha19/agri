"""
Automated Model Retraining Pipeline
====================================
FastAPI router for triggering, monitoring, and inspecting model retraining runs.

Endpoints
---------
POST /api/retraining/trigger   — queue a retraining Celery task   [admin only]
GET  /api/retraining/status    — pipeline health + optional task poll [admin only]
GET  /api/retraining/history   — last N run records from retraining_history.json [admin only]
"""

from fastapi import APIRouter, HTTPException, Request
from typing import Optional, Callable, Coroutine, Any
import logging
import os
import json
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Auth injection ────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Auth injection — set at startup by main.py via init_auth()
# Mirrors the pattern in feature_drift.py
# ---------------------------------------------------------------------------
_verify_role_fn: Optional[Callable[..., Coroutine[Any, Any, dict]]] = None


def init_auth(verify_role_fn: Callable[..., Coroutine[Any, Any, dict]]) -> None:
    """
    Inject the application-level verify_role dependency.
    ...
    """
    global _verify_role_fn
    _verify_role_fn = verify_role_fn
    logger.info("Retraining pipeline router: auth initialised")


async def _require_admin(request: Request) -> dict:
    """
    Shared dependency — enforces admin-only access on every route in this router.
    ...
    """
    if _verify_role_fn is None:
        raise HTTPException(status_code=503, detail="Authorization service not initialised")
    return await _verify_role_fn(request, required_roles=["admin"])


router = APIRouter(prefix="/api/retraining", tags=["retraining"])
# ---------------------------------------------------------------------------
# Auth injection — set at startup by main.py via init_auth()
# Mirrors the pattern in feature_drift.py
# ---------------------------------------------------------------------------
_verify_role_fn: Optional[Callable[..., Coroutine[Any, Any, dict]]] = None
# ---------------------------------------------------------------------------
# Path constants — relative to process working directory (repo root)
# ---------------------------------------------------------------------------
_HISTORY_PATH = Path("retraining_history.json")
_DRIFT_LOG_PATH = Path("drift_logs/drift.log")
_BASELINE_PATH = Path("feature_baseline.json")
_MODEL_PATH = Path("yield_model.joblib")

# Drift alert count in last 50 predictions that triggers auto-retraining
_DRIFT_ALERT_TRIGGER_COUNT = 5


def _read_history() -> dict:
    if not _HISTORY_PATH.exists():
        return {"runs": []}
    try:
        return json.loads(_HISTORY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"runs": []}


def _drift_threshold_breached() -> tuple:
    """
    Scan the last 50 entries of drift_logs/drift.log.
    Returns (breached: bool, reason: str).
    """
    if not _DRIFT_LOG_PATH.exists():
        return False, "No drift log found — run predictions first to generate drift data."

    try:
        lines = _DRIFT_LOG_PATH.read_text(encoding="utf-8").strip().splitlines()
        if not lines:
            return False, "Drift log is empty."
        recent = lines[-50:]
        alert_count = sum(
            1 for line in recent
            if json.loads(line).get("overall_status") == "alert"
        )
        if alert_count >= _DRIFT_ALERT_TRIGGER_COUNT:
            return True, (
                f"{alert_count} drift alerts detected in the last {len(recent)} predictions "
                f"(threshold: {_DRIFT_ALERT_TRIGGER_COUNT})."
            )
        return False, (
            f"Only {alert_count} alerts in last {len(recent)} predictions "
            f"(threshold: {_DRIFT_ALERT_TRIGGER_COUNT}). No retraining needed."
        )
    except Exception as exc:
        return False, f"Could not parse drift log: {exc}"


@router.post("/trigger")
async def trigger_retraining(
    request: Request,
    csv_path: str = "Train.csv",
    force: bool = False,
):
    await _require_admin(request)
    """
    Queue a model retraining job via Celery.  Admin only.

    Parameters
    ----------
    csv_path : str
        Path to the training CSV (default: Train.csv in repo root).
    force : bool
        If True, bypass drift threshold check and retrain unconditionally.
        If False (default), only retrain when drift alert count >= threshold.

    Returns the Celery task_id for status polling via GET /api/retraining/status.
    """
    try:
        if not os.path.exists(csv_path):
            raise HTTPException(
                status_code=404,
                detail=f"Training CSV not found at '{csv_path}'.",
            )

        if not force:
            breached, reason = _drift_threshold_breached()
            if not breached:
                return {
                    "success": False,
                    "triggered": False,
                    "reason": reason,
                }

        from celery_worker import retrain_yield_model_task
        task = retrain_yield_model_task.delay(
            csv_path=csv_path,
            model_output="yield_model.joblib",
        )

        logger.info(
            "Retraining task queued — task_id=%s csv=%s force=%s",
            task.id, csv_path, force,
        )

        return {
            "success": True,
            "triggered": True,
            "task_id": task.id,
            "message": "Retraining task queued. Poll GET /api/retraining/status?task_id= for progress.",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to trigger retraining: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_retraining_status(request: Request, task_id: Optional[str] = None):
    await _require_admin(request)
    """
    Returns pipeline health and optionally the live state of a Celery task.
    Admin only.

    Without task_id: returns overall pipeline health (baseline/model presence,
    drift breach status, last run summary).

    With task_id: additionally polls Celery for that task's current state
    (PENDING → PROGRESS → SUCCESS/FAILURE).
    """
    try:
        history = _read_history()
        runs = history.get("runs", [])
        last_run = runs[-1] if runs else None
        breached, drift_reason = _drift_threshold_breached()

        pipeline = {
            "model_exists": _MODEL_PATH.exists(),
            "baseline_exists": _BASELINE_PATH.exists(),
            "backup_model_exists": Path("yield_model.joblib.prev").exists(),
            "backup_baseline_exists": Path("feature_baseline.prev.json").exists(),
            "drift_threshold_breached": breached,
            "drift_reason": drift_reason,
            "total_runs": len(runs),
            "last_run": last_run,
        }

        if task_id:
            from celery_worker import celery_app
            result = celery_app.AsyncResult(task_id)
            pipeline["task_id"] = task_id
            pipeline["task_state"] = result.state
            pipeline["task_info"] = (
                result.info if isinstance(result.info, dict) else {}
            )

        return {"success": True, "pipeline": pipeline}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get retraining status: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_retraining_history(request: Request, limit: int = 20):
    await _require_admin(request)
    """
    Return the last N retraining run records, most recent first.  Admin only.

    Each record contains:
      triggered_at, completed_at, rmse, previous_rmse,
      outcome (promoted | rejected | failed), csv_path.
    """
    try:
        if limit < 1 or limit > 100:
            raise HTTPException(status_code=400, detail="limit must be between 1 and 100.")

        history = _read_history()
        runs = history.get("runs", [])
        recent = list(reversed(runs[-limit:]))

        promoted = sum(1 for r in runs if r.get("outcome") == "promoted")
        rejected = sum(1 for r in runs if r.get("outcome") == "rejected")
        failed   = sum(1 for r in runs if r.get("outcome") == "failed")

        return {
            "success": True,
            "total": len(runs),
            "summary": {
                "promoted": promoted,
                "rejected": rejected,
                "failed": failed,
            },
            "runs": recent,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get retraining history: %s", e)
        raise HTTPException(status_code=500, detail=str(e))