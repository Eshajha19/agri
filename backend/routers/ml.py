"""ML Prediction Router - Yield prediction endpoints"""
import os
import logging
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field
from error_utils import safe_detail

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
    data: list[float]

model_router = None
model_lag = None
model_trend = None

TREND_MODEL_PATH = "trend_forecast_model.joblib"

def init_router(r_instance, model_lag_instance, model_trend_instance=None):
    global model_router, model_lag, model_trend
    model_router = r_instance
    model_lag = model_lag_instance
    model_trend = model_trend_instance

@router.get("")
def predict_get():
    return {"predicted_yield": 2500, "note": "Use POST endpoint for actual prediction"}

@router.post("", response_model=PredictResponse)
def predict_yield(data: PredictRequest, request: Request):
    """Yield prediction using ML router"""
    if model_router is None:
        raise HTTPException(status_code=500, detail="ML model not initialized")
    try:
        input_data = data.model_dump() if hasattr(data, "model_dump") else data.dict()
        context = {"location": request.headers.get("X-User-Location", "Unknown"), "crop": data.Crop}
        predicted_yield = model_router.predict(input_data, context)
        return {"predicted_ExpYield": float(predicted_yield)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=safe_detail(e, 400))

@router.post("/predict-yield-lag")
async def predict_yield_lag(payload: YieldInput, request: Request):
    if model_lag is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    try:
        import numpy as np
        data = np.array(payload.data).reshape(1, -1) if len(payload.data) == 5 else None
        if data is None:
            raise ValueError("Exactly 5 values required")
        prediction = model_lag.predict(data)
        return {"prediction": round(float(prediction[0]), 2), "model": "RandomForest Time Series"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=safe_detail(e, 400))

@router.post("/predict-yield-trend")
async def predict_yield_trend(payload: YieldInput, request: Request):
    """
    Multi-step yield trend prediction using dedicated trend forecast model.
    
    Uses a separate `model_trend` — distinct from the lag-feature model — 
    to generate multi-step future predictions. Raises a clear error
    if the trend model is unavailable instead of silently using the wrong model.
    """
    global model_trend

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
        trend = []
        temp = list(payload.data if len(payload.data) == 5 else [0] * 5)

        for _ in range(5):
            pred = model_trend.predict([temp[:5]])[0]
            pred_value = round(float(pred), 2)
            trend.append(pred_value)
            temp = temp[1:] + [pred_value]

        return {"trend": trend, "prediction": trend[-1], "model": "Dedicated Trend Forecast"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=safe_detail(e, 400))
