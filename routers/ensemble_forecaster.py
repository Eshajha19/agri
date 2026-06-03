"""
Ensemble Forecaster Router
===========================
FastAPI router for stacked ensemble predictions with confidence intervals.

Endpoints
---------
POST /api/ensemble/forecast    — single prediction with CI
POST /api/ensemble/multi-step — 3-season ahead forecast
GET  /api/ensemble/weights    — current model contribution weights
GET  /api/ensemble/disagreement — recent high-disagreement predictions
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ensemble", tags=["ensemble forecasting"])


@router.post("/forecast")
async def ensemble_forecast(request: Request, input_data: dict):
    """
    Single-step ensemble prediction with confidence interval.
    """
    try:
        from ml.ensemble import get_ensemble_stacker

        stacker = get_ensemble_stacker()
        result = stacker.predict(input_data)

        return {
            "success": True,
            "prediction": result,
        }

    except Exception as e:
        logger.error("Ensemble forecast failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/multi-step")
async def ensemble_multi_step(
    request: Request,
    input_data: dict,
    steps: int = 3,
):
    """
    Multi-step ensemble forecast (current + next N-1 seasons).
    """
    try:
        if steps < 1 or steps > 5:
            raise HTTPException(status_code=400, detail="steps must be between 1 and 5")

        from celery_worker import ensemble_forecast_task
        task = ensemble_forecast_task.delay(input_data, steps=steps)

        return {
            "success": True,
            "task_id": task.id,
            "message": f"Multi-step forecast queued ({steps} steps). Poll status via Celery.",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Multi-step forecast failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/weights")
async def get_ensemble_weights():
    """
    Current model contribution weights.
    """
    try:
        from ml.ensemble import get_ensemble_stacker

        stacker = get_ensemble_stacker()
        stacker.load_models()

        return {
            "success": True,
            "weights": stacker.weights,
            "models_loaded": {
                "xgboost": stacker.xgb_adapter is not None,
                "lstm": stacker.lstm_model is not None,
                "random_forest": stacker.rf_model is not None,
            },
        }

    except Exception as e:
        logger.error("Failed to get ensemble weights: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/disagreement")
async def get_ensemble_disagreement(limit: int = 50):
    """
    Recent predictions where models disagreed significantly.
    """
    try:
        if limit < 1 or limit > 200:
            raise HTTPException(status_code=400, detail="limit must be between 1 and 200")

        from ml.ensemble import get_ensemble_stacker

        stacker = get_ensemble_stacker()
        records = stacker.recent_disagreements(limit=limit)

        return {
            "success": True,
            "count": len(records),
            "records": records,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get disagreement history: %s", e)
        raise HTTPException(status_code=500, detail=str(e))