"""
Price Prediction Router
=======================
Market price forecasting with STL decomposition, volatility scoring,
and personal alert threshold management.

Endpoints
---------
POST /api/prices/forecast       — generate price forecast for a crop
GET  /api/prices/volatility/{crop} — volatility score and classification
POST /api/prices/alerts/set    — set personal price alert threshold
GET  /api/prices/alerts        — list active alerts for authenticated farmer
POST /api/prices/alerts/check  — manually trigger alert evaluation
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/prices", tags=["price prediction"])

_DB = None


def init_price_prediction(db_client):
    global _DB
    _DB = db_client


def _get_db():
    if _DB is not None:
        return _DB
    try:
        import firebase_admin
        from firebase_admin import firestore
        if firebase_admin._apps:
            return firestore.client()
    except Exception:
        pass
    return None


@router.post("/forecast")
async def price_forecast(crop: str, days: int = 14):
    """
    Generate price forecast for a crop with confidence intervals.
    """
    try:
        if days < 1 or days > 60:
            raise HTTPException(status_code=400, detail="days must be between 1 and 60")

        from ml.price_forecaster import get_price_forecaster

        forecaster = get_price_forecaster()
        result = forecaster.forecast(crop, days=days)

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return {
            "success": True,
            "forecast": result,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Price forecast failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/volatility/{crop}")
async def price_volatility(crop: str):
    """
    Current volatility score and classification for a crop.
    """
    try:
        from ml.price_forecaster import get_price_forecaster

        forecaster = get_price_forecaster()
        result = forecaster.volatility_score(crop)

        return {
            "success": True,
            "crop": crop,
            "volatility": result,
        }

    except Exception as e:
        logger.error("Volatility check failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/set")
async def set_price_alert(request: Request, crop: str, threshold_type: str, threshold_value: float):
    """
    Set a personal price alert threshold for the authenticated farmer.
    Stores in Firestore users/{uid}/price_alerts.
    """
    try:
        from main import verify_role

        token_data = await verify_role(
            request,
            required_roles=["farmer"],          # or whatever role is needed
            required_tenant_id=tenant_id        # comes from path param or request context
        )
        uid = token_data.get("uid")


        if threshold_type not in {"above", "below", "volatility"}:
            raise HTTPException(status_code=400, detail="threshold_type must be 'above', 'below', or 'volatility'")

        if threshold_value <= 0:
            raise HTTPException(status_code=400, detail="threshold_value must be positive")

        db = _get_db()
        if db is None:
            raise HTTPException(status_code=503, detail="Database unavailable")

        alert_id = f"{crop}_{threshold_type}"
        alert_data = {
            "crop": crop,
            "threshold_type": threshold_type,
            "threshold_value": threshold_value,
            "created_at": __import__("datetime").datetime.utcnow().isoformat(),
            "active": True,
        }

        db.collection("users").document(uid).collection("price_alerts").document(alert_id).set(alert_data)

        return {
            "success": True,
            "alert_id": alert_id,
            "crop": crop,
            "threshold_type": threshold_type,
            "threshold_value": threshold_value,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to set price alert: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts")
async def get_price_alerts(request: Request):
    """
    List active price alerts for the authenticated farmer.
    """
    try:
        from main import verify_role

        token_data = await verify_role(
            request,
            required_roles=["farmer"],          # or whatever role is needed
            required_tenant_id=tenant_id        # comes from path param or request context
        )
        uid = token_data.get("uid")


        db = _get_db()
        if db is None:
            raise HTTPException(status_code=503, detail="Database unavailable")

        docs = db.collection("users").document(uid).collection("price_alerts").where("active", "==", True).stream()
        alerts = []
        for doc in docs:
            data = doc.to_dict() or {}
            alerts.append({
                "alert_id": doc.id,
                "crop": data.get("crop"),
                "threshold_type": data.get("threshold_type"),
                "threshold_value": data.get("threshold_value"),
                "created_at": data.get("created_at"),
            })

        return {
            "success": True,
            "alerts": alerts,
            "count": len(alerts),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get price alerts: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/check")
async def check_price_alerts(request: Request):
    """
    Manually trigger alert evaluation. Admin/expert only.
    """
    try:
        from main import verify_role

        await verify_role(request, required_roles=["admin", "expert"])

        from celery_worker import check_price_alerts_task
        task = check_price_alerts_task.delay()

        return {
            "success": True,
            "task_id": task.id,
            "message": "Alert evaluation queued. Check Celery for results.",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to queue alert check: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/decomposition/{crop}")
async def price_decomposition(crop: str):
    """
    STL decomposition components for a crop (trend, seasonal, residual).
    """
    try:
        from ml.price_forecaster import get_price_forecaster

        forecaster = get_price_forecaster()
        result = forecaster.stl_decompose(crop)

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return {
            "success": True,
            "crop": crop,
            "decomposition": result,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Decomposition failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))