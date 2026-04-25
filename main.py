from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from alert_rules import generate_alerts

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    Crop: str
    CropCoveredArea: float = Field(..., gt=0)
    CHeight: int = Field(..., ge=0)
    CNext: str
    CLast: str
    CTransp: str
    IrriType: str
    IrriSource: str
    IrriCount: int = Field(..., ge=1)
    WaterCov: int = Field(..., ge=0, le=100)
    Season: str

class PredictResponse(BaseModel):
    predicted_ExpYield: float

# Load model
try:
    model = joblib.load("yield_model.joblib")
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Store notifications
@app.get("/api/notifications")
def get_notifications(
    crop: str = None,
    irrigation_count: int = None,
    water_coverage: int = None,
    season: str = None
):
    """
    Generate dynamic farm advisory alerts.
    
    Query params (all optional):
    - crop: rice / wheat / maize
    - irrigation_count: number of irrigations done
    - water_coverage: 0-100 (% of field covered)
    - season: kharif / rabi / zaid (auto-detected if not passed)
    """
    alerts = generate_alerts(
        crop=crop,
        irrigation_count=irrigation_count,
        water_coverage=water_coverage,
        season=season
    )
    return {"success": True, "data": alerts}

@app.get("/")
def root():
    return {"message": "Fasal Saathi Yield Prediction API", "status": "running"}

@app.get("/predict")
def predict_get():
    return {"predicted_yield": 2500, "note": "Use POST endpoint for actual prediction"}

@app.post("/predict", response_model=PredictResponse)
def predict_yield(data: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        input_data = {
            'Crop': data.Crop,
            'CropCoveredArea': data.CropCoveredArea,
            'CHeight': data.CHeight,
            'CNext': data.CNext,
            'CLast': data.CLast,
            'CTransp': data.CTransp,
            'IrriType': data.IrriType,
            'IrriSource': data.IrriSource,
            'IrriCount': data.IrriCount,
            'WaterCov': data.WaterCov,
            'Season': data.Season
        }
        df = pd.DataFrame([input_data])
        
        dummy_cols = ['Crop', 'CNext', 'CLast', 'CTransp', 'IrriType', 'IrriSource', 'Season']
        df = pd.get_dummies(df, columns=dummy_cols, drop_first=True)
        
        feature_cols = list(model.get_booster().feature_names)
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_cols]
        
        predicted_yield = model.predict(df)[0]
        return {"predicted_ExpYield": float(predicted_yield)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/log-error")
async def log_error(request: Request):
    """
    Receive error reports from the frontend for monitoring and debugging.
    """
    try:
        error_data = await request.json()
        print(f"[Error Log] {error_data.get('message', 'Unknown error')} | Context: {error_data.get('context', 'N/A')}")
        return {"success": True, "message": "Error logged"}
    except Exception:
        return {"success": False, "message": "Invalid error data"}

