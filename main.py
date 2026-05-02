# main.py
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
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

@app.post("/api/log-error")
async def log_error(request: Request):
    """
    Receive error reports from the frontend for monitoring and debugging.
    """
    try:
        error_data = await request.json()
        # Log to stdout (could be extended to file or external service)
        print(f"[Error Log] {error_data.get('message', 'Unknown error')} | Context: {error_data.get('context', 'N/A')}")
        return {"success": True, "message": "Error logged"}
    except Exception:
        # Silently ignore malformed payloads to avoid breaking error reporting
        return {"success": False, "message": "Invalid error data"}


########################
# Advanced Smart Crop Recommendation Engine
########################

class CropRecommendationRequest(BaseModel):
    location: str = Field(..., description="Location name (city/region)")
    soil_type: str = Field(..., description="Type of soil (e.g., loamy, clayey, sandy)")
    latitude: float = Field(None, description="Latitude for weather API")
    longitude: float = Field(None, description="Longitude for weather API")
    season: str = Field(None, description="Current season")
    area_size: float = Field(None, description="Farm area size in hectares")

# Crop suitability rules based on weather and soil conditions
CROP_SOIL_COMPATIBILITY = {
    "wheat": {
        "soil_types": ["loamy", "clayey", "silty"],
        "temp_range": {"min": 8, "max": 25},
        "rainfall_range": {"min": 40, "max": 150},
        "seasons": ["rabi"],
        "description": "Wheat thrives in cool weather with moderate rainfall"
    },
    "rice": {
        "soil_types": ["clayey", "loamy"],
        "temp_range": {"min": 20, "max": 32},
        "rainfall_range": {"min": 100, "max": 300},
        "seasons": ["kharif"],
        "description": "Rice needs high moisture and warm temperatures"
    },
    "maize": {
        "soil_types": ["loamy", "sandy-loam", "sandy"],
        "temp_range": {"min": 18, "max": 27},
        "rainfall_range": {"min": 50, "max": 250},
        "seasons": ["kharif", "summer"],
        "description": "Maize prefers warm, well-drained soils"
    },
    "cotton": {
        "soil_types": ["black", "red", "loamy"],
        "temp_range": {"min": 21, "max": 30},
        "rainfall_range": {"min": 50, "max": 100},
        "seasons": ["kharif"],
        "description": "Cotton needs warm climate with moderate water"
    },
    "sugarcane": {
        "soil_types": ["black", "loamy", "alluvial"],
        "temp_range": {"min": 20, "max": 30},
        "rainfall_range": {"min": 100, "max": 250},
        "seasons": ["kharif", "rabi"],
        "description": "Sugarcane thrives in tropical and subtropical regions"
    },
    "groundnut": {
        "soil_types": ["sandy", "sandy-loam", "red"],
        "temp_range": {"min": 20, "max": 28},
        "rainfall_range": {"min": 50, "max": 100},
        "seasons": ["kharif", "summer"],
        "description": "Groundnut prefers well-drained, sandy soils"
    },
    "pulses": {
        "soil_types": ["loamy", "sandy-loam", "red"],
        "temp_range": {"min": 15, "max": 25},
        "rainfall_range": {"min": 40, "max": 100},
        "seasons": ["rabi"],
        "description": "Pulses grow well in cool season with moderate rainfall"
    },
    "soybean": {
        "soil_types": ["loamy", "sandy-loam", "black"],
        "temp_range": {"min": 20, "max": 30},
        "rainfall_range": {"min": 50, "max": 150},
        "seasons": ["kharif"],
        "description": "Soybean needs warm climate and moderate moisture"
    },
    "tomato": {
        "soil_types": ["loamy", "sandy-loam"],
        "temp_range": {"min": 15, "max": 28},
        "rainfall_range": {"min": 50, "max": 100},
        "seasons": ["kharif", "rabi"],
        "description": "Tomato requires well-drained soil and moderate temperature"
    },
    "onion": {
        "soil_types": ["loamy", "sandy-loam"],
        "temp_range": {"min": 10, "max": 25},
        "rainfall_range": {"min": 50, "max": 75},
        "seasons": ["rabi"],
        "description": "Onion prefers cool season with moderate moisture"
    },
}

SOIL_NORMALIZATION = {
    "black": "black",
    "red": "red",
    "loamy": "loamy",
    "clayey": "clayey",
    "clay": "clayey",
    "sandy": "sandy",
    "sandy-loam": "sandy-loam",
    "alluvial": "alluvial",
    "silty": "silty",
    "laterite": "laterite",
}

def normalize_soil_type(soil: str) -> str:
    """Normalize soil type input"""
    normalized = SOIL_NORMALIZATION.get(soil.lower().strip(), soil.lower().strip())
    return normalized

def fetch_weather_data(latitude: float, longitude: float) -> dict:
    """Fetch weather data from OpenWeatherMap API"""
    try:
        api_key = "demo"  # Use a demo key or load from env
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={api_key}&units=metric"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            return {
                "temperature": data.get("main", {}).get("temp", 20),
                "rainfall": data.get("rain", {}).get("1h", 0) * 30,  # Estimate monthly
                "humidity": data.get("main", {}).get("humidity", 60),
                "weather": data.get("weather", [{}])[0].get("main", "Clear"),
            }
    except Exception as e:
        print(f"Weather API error: {e}")
    
    # Return default weather if API fails
    return {
        "temperature": 22,
        "rainfall": 80,
        "humidity": 60,
        "weather": "Estimated",
    }

def recommend_crops(soil_type: str, weather: dict, season: str = None) -> list:
    """
    Apply rule-based logic to recommend suitable crops
    """
    recommended = []
    soil_type = normalize_soil_type(soil_type)
    
    temp = weather.get("temperature", 22)
    rainfall = weather.get("rainfall", 80)
    
    for crop, rules in CROP_SOIL_COMPATIBILITY.items():
        score = 0
        reasons = []
        
        # Check soil compatibility
        soil_match = any(
            s in soil_type.lower() or soil_type.lower() in s
            for s in rules["soil_types"]
        )
        if soil_match:
            score += 30
            reasons.append("✓ Soil type suitable")
        else:
            reasons.append("✗ Soil type marginal")
        
        # Check temperature range
        if rules["temp_range"]["min"] <= temp <= rules["temp_range"]["max"]:
            score += 25
            reasons.append(f"✓ Temperature ideal ({temp}°C)")
        elif abs(temp - rules["temp_range"]["min"]) < 5 or abs(temp - rules["temp_range"]["max"]) < 5:
            score += 10
            reasons.append(f"⚠ Temperature marginal ({temp}°C)")
        
        # Check rainfall range
        if rules["rainfall_range"]["min"] <= rainfall <= rules["rainfall_range"]["max"]:
            score += 25
            reasons.append(f"✓ Rainfall optimal ({rainfall}mm)")
        elif abs(rainfall - rules["rainfall_range"]["min"]) < 20 or abs(rainfall - rules["rainfall_range"]["max"]) < 20:
            score += 10
            reasons.append(f"⚠ Rainfall marginal ({rainfall}mm)")
        
        # Check season
        if season and season.lower() in rules["seasons"]:
            score += 20
            reasons.append(f"✓ Right season ({season})")
        
        if score >= 50:  # Minimum threshold
            recommended.append({
                "crop": crop.title(),
                "score": score,
                "reasons": reasons,
                "description": rules["description"],
                "optimal_season": rules["seasons"][0]
            })
    
    # Sort by score
    recommended.sort(key=lambda x: x["score"], reverse=True)
    return recommended

@app.post("/api/crop-recommendations")
async def get_crop_recommendations(request: CropRecommendationRequest):
    """
    Get smart crop recommendations based on location, soil type, and weather data
    """
    try:
        # Fetch weather data
        if request.latitude and request.longitude:
            weather_data = fetch_weather_data(request.latitude, request.longitude)
        else:
            # Default weather data
            weather_data = {
                "temperature": 22,
                "rainfall": 80,
                "humidity": 60,
                "weather": "Estimated",
            }
        
        # Get crop recommendations
        recommendations = recommend_crops(
            request.soil_type,
            weather_data,
            request.season
        )
        
        return {
            "success": True,
            "location": request.location,
            "soil_type": normalize_soil_type(request.soil_type),
            "weather": weather_data,
            "recommendations": recommendations[:5],  # Top 5 recommendations
            "total_recommendations": len(recommendations),
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


