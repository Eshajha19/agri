"""ML Prediction Router - Yield prediction endpoints"""
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()

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

def init_router(r_instance, model_lag_instance):
    global model_router, model_lag
    model_router = r_instance
    model_lag = model_lag_instance

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
        raise HTTPException(status_code=400, detail=str(e))

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
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/predict-yield-trend")
async def predict_yield_trend(payload: YieldInput, request: Request):
    """
    Multi-step yield trend prediction using sliding window.
    
    Maintains proper feature alignment by using the last 4 original features
    plus the newly predicted value to form the next input vector.
    This preserves the semantic meaning of each feature position.
    """
    if model_lag is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    try:
        trend = []
        temp = list(payload.data if len(payload.data) == 5 else [0] * 5)
        
        for _ in range(5):
            pred = model_lag.predict([temp[:5]])[0]
            pred_value = round(float(pred), 2)
            trend.append(pred_value)
            temp = temp[1:] + [pred_value]
        
        return {"trend": trend, "prediction": trend[-1], "model": "RandomForest Trend"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
