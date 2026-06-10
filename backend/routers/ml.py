"""ML Prediction Router - Yield prediction endpoints"""
import os
import threading
import logging
import threading
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field
from ml.security import verify_and_load_joblib

router = APIRouter()
logger = logging.getLogger(__name__)

class PredictRequest(BaseModel):
    Crop: str = Field(..., max_length=50)
    CropCoveredArea: float = Field(..., gt=0)
    CHeight: int = Field(..., ge=0)
    CNext: str = Field(..., max_length=50)
    CLast: str = Field(..., max_length=50)
    CTransp: str = Field(..., max_length=50)
    IrriType: str = Field(..., max_length=50)
    IrriSource: str = Field(..., max_length=50)
    IrriCount: int = Field(..., ge=1)
    WaterCov: int = Field(..., ge=0, le=100)
    Season: str = Field(..., max_length=50)

class PredictResponse(BaseModel):
    predicted_ExpYield: float

class YieldInput(BaseModel):
    # Cap the list length to prevent a single request from allocating an
    # arbitrarily large numpy array. predict_yield_lag expects exactly 5
    # values; predict_yield_trend uses 5 values internally. 1000 items is
    # several orders of magnitude above any legitimate use case and keeps
    # peak memory bounded at roughly 8 kB (1000 float64 values).
    data: list[float] = Field(..., min_items=1, max_items=1000)

model_router = None
model_lag = None
model_trend = None
verify_role_fn = None

TREND_MODEL_PATH = "trend_forecast_model.joblib"
_trend_lock = threading.Lock()

def rollback_on_drift(alert: dict):
    """
    Callback fired when drift or candidate degradation is detected.
    Overwrites the active model in ModelRegistry with the stable model_lag model.
    """
    logger.warning("Drift or performance degradation detected: %s. Rolling back to stable model in ModelRegistry.", alert)
    from ml.registry import ModelRegistry
    if model_lag is not None:
        ModelRegistry.register("xgboost", model_lag)
        logger.info("Successfully rolled back xgboost to stable model_lag")
    else:
        logger.error("Rollback failed: stable model_lag is not loaded")

def init_router(r_instance, model_lag_instance, model_trend_instance=None, verify_role=None):
    global model_router, model_lag, model_trend, verify_role_fn
    model_router = r_instance
    model_lag = model_lag_instance
    model_trend = model_trend_instance
    verify_role_fn = verify_role

    # Register drift rollback callback
    from backend.routers import governance
    if governance.drift_detector is not None:
        governance.drift_detector.on_drift_detected(rollback_on_drift)
    if governance.shadow_evaluator is not None:
        governance.shadow_evaluator.on_drift_detected(rollback_on_drift)

@router.get("")
def predict_get():
    raise HTTPException(
        status_code=404,
        detail="This endpoint is disabled. Use POST /api/ml for authenticated yield predictions.",
    )

@router.post("", response_model=PredictResponse)
async def predict_yield(data: PredictRequest, request: Request):
    """Yield prediction using ML router"""
    if verify_role_fn is None:
        raise HTTPException(status_code=500, detail="Auth service not initialized")
    if model_router is None:
        raise HTTPException(status_code=500, detail="ML model not initialized")
    await verify_role_fn(request)
    try:
        input_data = data.model_dump() if hasattr(data, "model_dump") else data.dict()
        raw_location = request.headers.get("X-User-Location", "Unknown")
        sanitised_location = re.sub(r"[^\w\s,.-]", "", raw_location)[:100].strip() or "Unknown"
        context = {"location": sanitised_location, "crop": data.Crop}
        predicted_yield = model_router.predict(input_data, context)
        return {"predicted_ExpYield": float(predicted_yield)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("predict_yield error: %s", e)
        raise HTTPException(status_code=500, detail="An unexpected error occurred during prediction.")

@router.post("/predict-yield-lag")
async def predict_yield_lag(payload: YieldInput, request: Request):
    if verify_role_fn is None:
        raise HTTPException(status_code=500, detail="Auth service not initialized")
    await verify_role_fn(request)
    if model_lag is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    try:
        import numpy as np
        data = np.array(payload.data).reshape(1, -1) if len(payload.data) == 5 else None
        if data is None:
            raise ValueError("Exactly 5 values required")
        prediction = model_lag.predict(data)
        return {"prediction": round(float(prediction[0]), 2), "model": "RandomForest Time Series"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("predict_yield_lag error: %s", e)
        raise HTTPException(status_code=500, detail="An unexpected error occurred during prediction.")

@router.post("/predict-yield-trend")
async def predict_yield_trend(payload: YieldInput, request: Request):
    """
    Multi-step yield trend prediction using dedicated trend forecast model.
    
    Uses a separate `model_trend` — distinct from the lag-feature model — 
    to generate multi-step future predictions. Raises a clear error
    if the trend model is unavailable instead of silently using the wrong model.
    """
    if verify_role_fn is None:
        raise HTTPException(status_code=500, detail="Auth service not initialized")
    await verify_role_fn(request)

    global model_trend

    if model_trend is None:
        with _trend_lock:
            if model_trend is None:
                try:
                    import joblib
                    if os.path.exists(TREND_MODEL_PATH):
                        model_trend = joblib.load(TREND_MODEL_PATH)
                        logger.info("Trend forecast model loaded from %s", TREND_MODEL_PATH)
                    else:
                        raise FileNotFoundError(f"Trend model not found at {TREND_MODEL_PATH}")
                except Exception as load_err:
                    logger.error("Trend forecast model unavailable: %s. Endpoint cannot serve trend predictions.", load_err)
                    raise HTTPException(
                        status_code=503,
                        detail="Trend forecast model is not loaded. A dedicated trend model is required — "
                               "the lag-feature model (used by /predict-yield-lag) is statistically invalid "
                               "for multi-step trend forecasting."
                    )
        try:
            import joblib
            if os.path.exists(TREND_MODEL_PATH):
                model_trend = verify_and_load_joblib(TREND_MODEL_PATH)
                logger.info("Trend forecast model loaded from %s", TREND_MODEL_PATH)
            else:
                raise FileNotFoundError(f"Trend model not found at {TREND_MODEL_PATH}")
        except Exception as load_err:
            logger.error("Trend forecast model unavailable: %s. Endpoint cannot serve trend predictions.", load_err)
            raise HTTPException(
                status_code=503,
                detail="Trend forecast model is not loaded. A dedicated trend model is required — "
                       "the lag-feature model (used by /predict-yield-lag) is statistically invalid "
                       "for multi-step trend forecasting."
            )

    try:
        trend = []
        temp = list(payload.data if len(payload.data) == 5 else [0] * 5)

        for _ in range(5):
            pred = model_trend.predict([temp[:5]])[0]
            pred_value = round(float(pred), 2)
            trend.append(pred_value)
            temp = temp[1:] + [pred_value]

        return {"trend": trend, "prediction": trend[-1], "model": "Dedicated Trend Forecast"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("predict_yield_trend error: %s", e)
        raise HTTPException(status_code=500, detail="An unexpected error occurred during prediction.")
