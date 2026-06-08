"""
Hyperparameter Optimization Router
====================================
FastAPI router for Bayesian hyperparameter search with Optuna.

Endpoints
---------
POST /api/hyperopt/start      — queue optimization Celery task
GET  /api/hyperopt/status     — current study status + best params so far
GET  /api/hyperopt/trials     — full trial history with scores
GET  /api/hyperopt/best-params — current best hyperparameter config
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/hyperopt", tags=["hyperparameter optimization"])

_CONFIG_PATH = Path("hyperparameter_config.json")


def _read_config() -> dict:
    if not _CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_history() -> dict:
    history_path = Path("retraining_history.json")
    if not history_path.exists():
        return {"runs": []}
    try:
        return json.loads(history_path.read_text(encoding="utf-8"))
    except Exception:
        return {"runs": []}


@router.post("/start")
async def start_hyperopt(
    request: Request,
    csv_path: str = "Train.csv",
    n_trials: int = 50,
    cv_folds: int = 5,
):
    """
    Queue a hyperparameter optimization job via Celery.

    Parameters
    ----------
    csv_path : str
        Path to the training CSV (default: Train.csv).
    n_trials : int
        Number of Optuna trials (default: 50, max: 200).
    cv_folds : int
        Number of cross-validation folds (default: 5, max: 10).

    Returns the Celery task_id for status polling.
    """
    try:
        if not os.path.exists(csv_path):
            raise HTTPException(
                status_code=404,
                detail=f"Training CSV not found at '{csv_path}'.",
            )

        if n_trials < 1 or n_trials > 200:
            raise HTTPException(status_code=400, detail="n_trials must be between 1 and 200.")

        if cv_folds < 2 or cv_folds > 10:
            raise HTTPException(status_code=400, detail="cv_folds must be between 2 and 10.")

        from celery_worker import run_hyperparameter_optimization_task
        task = run_hyperparameter_optimization_task.delay(
            csv_path=csv_path,
            n_trials=n_trials,
            cv_folds=cv_folds,
            study_name="yield_xgb_optimization",
        )

        logger.info(
            "Hyperopt task queued — task_id=%s csv=%s trials=%s folds=%s",
            task.id, csv_path, n_trials, cv_folds,
        )

        return {
            "success": True,
            "triggered": True,
            "task_id": task.id,
            "message": "Optimization task queued. Poll GET /api/hyperopt/status?task_id= for progress.",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to start hyperopt: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_hyperopt_status(task_id: Optional[str] = None):
    """
    Returns optimization status and optionally the live state of a Celery task.

    Without task_id: returns persisted best config and pipeline health.
    With task_id: additionally polls Celery for that task's current state.
    """
    try:
        config = _read_config()
        history = _read_history()
        runs = history.get("runs", [])
        last_run = runs[-1] if runs else None

        status_payload = {
            "config_exists": _CONFIG_PATH.exists(),
            "best_params": config.get("best_params"),
            "best_rmse": config.get("best_rmse"),
            "best_mean_mae": config.get("best_mean_mae"),
            "best_mean_r2": config.get("best_mean_r2"),
            "optimized_at": config.get("optimized_at"),
            "total_retraining_runs": len(runs),
            "last_retraining_run": last_run,
        }

        if task_id:
            from celery_worker import celery_app
            result = celery_app.AsyncResult(task_id)
            status_payload["task_id"] = task_id
            status_payload["task_state"] = result.state
            status_payload["task_info"] = (
                result.info if isinstance(result.info, dict) else {}
            )

        return {"success": True, "status": status_payload}

    except Exception as e:
        logger.error("Failed to get hyperopt status: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trials")
async def get_hyperopt_trials():
    """
    Return full trial history from the last completed optimization.

    Each trial contains: trial_number, params, rmse, mean_mae, mean_r2,
    fold_scores, duration_ms.
    """
    try:
        config = _read_config()
        trials = config.get("trial_history", [])

        return {
            "success": True,
            "study_name": config.get("study_name"),
            "n_trials": config.get("n_trials"),
            "cv_folds": config.get("cv_folds"),
            "trials": trials,
            "total_trials": len(trials),
        }

    except Exception as e:
        logger.error("Failed to get hyperopt trials: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/best-params")
async def get_best_params():
    """
    Return the current best hyperparameter configuration.
    """
    try:
        config = _read_config()
        if not config:
            raise HTTPException(
                status_code=404,
                detail="No hyperparameter config found. Run an optimization first.",
            )

        return {
            "success": True,
            "best_params": config.get("best_params"),
            "best_rmse": config.get("best_rmse"),
            "best_mean_mae": config.get("best_mean_mae"),
            "best_mean_r2": config.get("best_mean_r2"),
            "optimized_at": config.get("optimized_at"),
            "csv_path": config.get("csv_path"),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get best params: %s", e)
        raise HTTPException(status_code=500, detail=str(e))