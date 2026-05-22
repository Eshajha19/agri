"""Rule-based farmer advisory API."""
from collections import defaultdict, deque
from typing import Any, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from advisory_rules import generate_advisories


router = APIRouter()
_MAX_STORED_ALERTS = 50
_stored_alerts: dict[str, deque] = defaultdict(lambda: deque(maxlen=_MAX_STORED_ALERTS))


class AdvisoryRequest(BaseModel):
    weather: dict[str, Any] = Field(default_factory=dict)
    soil: dict[str, Any] = Field(default_factory=dict)
    crop_type: Optional[str] = Field(default=None, max_length=50)
    user_id: Optional[str] = Field(default=None, max_length=128)
    store_alerts: bool = False


@router.post("/advisory")
async def create_advisory(payload: AdvisoryRequest):
    alerts = generate_advisories(
        weather=payload.weather,
        soil=payload.soil,
        crop_type=payload.crop_type,
    )

    stored = False
    if payload.store_alerts and payload.user_id:
        _stored_alerts[payload.user_id].extend(alerts)
        stored = True

    return {
        "success": True,
        "data": alerts,
        "count": len(alerts),
        "stored": stored,
    }


@router.get("/advisory/{user_id}")
async def get_stored_advisories(user_id: str):
    return {
        "success": True,
        "data": list(_stored_alerts.get(user_id, [])),
    }
