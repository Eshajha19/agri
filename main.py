# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import base64
import requests

app = FastAPI()

# Allow CORS from React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # change if your frontend is elsewhere
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

######################
# Crop Yield Prediction
######################

class PredictRequest(BaseModel):
    Crop: str
    CropCoveredArea: float = Field(..., gt=0)
    CHeight: int = Field(..., ge=0)
    CNext: int = Field(..., ge=0)
    CLast: int = Field(..., ge=0)
    CTransp: int = Field(..., ge=0)
    IrriType: str
    IrriSource: str
    IrriCount: int = Field(..., ge=1)
    WaterCov: int = Field(..., ge=0, le=100)
    Season: str

    @validator("Crop")
    def validate_crop(cls, value):
        allowed_crops = ["paddy", "wheat", "corn"]
        if value.lower() not in allowed_crops:
            raise ValueError(f"Crop must be one of {allowed_crops}")
        return value

class PredictResponse(BaseModel):
    predicted_ExpYield: float

@app.post("/predict", response_model=PredictResponse)
async def predict_yield(data: PredictRequest):
    try:
        # Simple dummy prediction logic
        base_yield = 2.5
        modifier = (
            data.CropCoveredArea * 0.5 +
            data.CHeight * 0.01 -
            data.WaterCov * 0.1
        )
        predicted_yield = max(base_yield + modifier, 0)
        return {"predicted_ExpYield": predicted_yield}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

