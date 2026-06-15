from pydantic import BaseModel, validator
from typing import List

class YieldInput(BaseModel):
    values: List[float]

    @validator("values")
    def validate_values(cls, v):
        if len(v) != 5:
            raise ValueError("Exactly 5 input values are required")
        for i, val in enumerate(v):
            if not isinstance(val, (int, float)):
                raise ValueError(f"Value at index {i} must be a number")
            if val <= 0:
                raise ValueError(f"Value at index {i} must be greater than 0")
            if val > 10000:  # example agronomic upper bound
                raise ValueError(f"Value at index {i} exceeds realistic range")
        return v
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
