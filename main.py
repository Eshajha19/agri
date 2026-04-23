# main.py
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import base64
import requests
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

def generate_notifications():
    return [
        {
            "id": 1,
            "type": "weather",
            "message": "🌧️ Heavy rainfall expected in your region today.",
            "time": datetime.now().isoformat()
        },
        {
            "id": 2,
            "type": "recommendation",
            "message": "🌱 Ideal time to irrigate wheat crops.",
            "time": datetime.now().isoformat()
        }
    ]

@app.get("/api/notifications")
def get_notifications():
    return {
        "success": True,
        "data": generate_notifications()
    }
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("yield_model.joblib")

########################
# Crop Yield Prediction
########################

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

@app.post("/predict", response_model=PredictResponse)
async def predict_yield(data: PredictRequest):
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

