from pydantic import BaseModel, Field

class PredictionInput(BaseModel):
    crop_type: str
    region: str
    season: str
    soil_quality: float
    rainfall: float

    class Config:
        extra = "forbid"  

class YieldLagInput(PredictionInput):
    lag_days: int = Field(..., gt=0)

class YieldTrendInput(PredictionInput):
    trend_window: int = Field(..., gt=0)
