"""Crop Insurance Compensation Calculator — estimates payout based on crop, area, damage, and policy."""

import logging
from typing import Any, Dict, List

from fastapi import APIRouter
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()

CROP_BASE_VALUE: Dict[str, float] = {
    "paddy": 25000,
    "rice": 25000,
    "wheat": 22000,
    "maize": 20000,
    "cotton": 30000,
    "sugarcane": 35000,
    "soybean": 18000,
    "groundnut": 20000,
    "potato": 15000,
    "tomato": 18000,
    "pulses": 16000,
    "vegetables": 14000,
    "fruits": 22000,
}

POLICY_MULTIPLIER: Dict[str, float] = {
    "basic": 1.0,
    "standard": 1.2,
    "premium": 1.5,
    "comprehensive": 1.75,
}

POLICY_DESC: Dict[str, str] = {
    "basic": "Basic coverage — minimum payout",
    "standard": "Standard coverage — 1.2x multiplier",
    "premium": "Premium coverage — 1.5x multiplier",
    "comprehensive": "Comprehensive coverage — 1.75x multiplier",
}

MIN_DAMAGE = 0
MAX_DAMAGE = 100
MIN_AREA = 0.01
MAX_AREA = 100000


class CalculateRequest(BaseModel):
    crop_type: str = Field(..., min_length=1, max_length=50)
    cultivated_area: float = Field(..., gt=0, le=MAX_AREA)
    damage_percentage: float = Field(..., ge=MIN_DAMAGE, le=MAX_DAMAGE)
    policy_type: str = Field(..., min_length=1, max_length=20)


class CalculateResponse(BaseModel):
    estimated_compensation: float
    base_value_per_acre: float
    policy_multiplier: float
    policy_description: str
    breakdown: List[Dict[str, Any]]


@router.post("/compensation/calculate", response_model=CalculateResponse)
async def calculate_compensation(body: CalculateRequest):
    crop_key = body.crop_type.lower().strip()
    base_value = None
    for key, val in CROP_BASE_VALUE.items():
        if key in crop_key or crop_key in key:
            base_value = val
            break
    if base_value is None:
        base_value = 12000

    policy_key = body.policy_type.lower().strip()
    multiplier = POLICY_MULTIPLIER.get(policy_key, 1.0)
    policy_desc = POLICY_DESC.get(policy_key, "Custom policy")

    damage_ratio = body.damage_percentage / 100.0
    compensation = body.cultivated_area * base_value * damage_ratio * multiplier
    compensation = round(compensation, 2)

    gross_value = body.cultivated_area * base_value
    breakdown = [
        {"label": "Cultivated Area", "value": f"{body.cultivated_area} acres"},
        {"label": "Base Value per Acre", "value": f"₹{base_value:,.0f}"},
        {"label": "Gross Value (Area × Base)", "value": f"₹{gross_value:,.0f}"},
        {"label": "Damage Percentage", "value": f"{body.damage_percentage}%"},
        {"label": "Applicable Damage Amount", "value": f"₹{(gross_value * damage_ratio):,.0f}"},
        {"label": "Policy Multiplier", "value": f"{multiplier}x"},
        {"label": "Policy Type", "value": policy_desc},
    ]

    return CalculateResponse(
        estimated_compensation=compensation,
        base_value_per_acre=base_value,
        policy_multiplier=multiplier,
        policy_description=policy_desc,
        breakdown=breakdown,
    )
