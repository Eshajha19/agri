"""Insurance Claim Router — auto-fills claim forms from farm profile & weather data."""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()

_collection = None
_weather_service = None


def init_insurance(firestore_collection, weather_svc=None):
    global _collection, _weather_service
    _collection = firestore_collection
    _weather_service = weather_svc


class AutoFillResult(BaseModel):
    crop_name: Optional[str] = None
    confidence_crop: float = 0.0
    sowing_date: Optional[str] = None
    confidence_sowing: float = 0.0
    farm_location: Optional[str] = None
    confidence_location: float = 0.0
    farm_area: Optional[float] = None
    confidence_area: float = 0.0
    village: Optional[str] = None
    confidence_village: float = 0.0
    district: Optional[str] = None
    confidence_district: float = 0.0
    weather_events: List[Dict[str, Any]] = []
    fill_percentage: float = 0.0


class ClaimSubmitRequest(BaseModel):
    crop_name: str = Field(..., min_length=1, max_length=100)
    sowing_date: str = Field(..., min_length=1)
    farm_location: str = Field(..., max_length=300)
    farm_area: float = Field(..., gt=0, le=100000)
    village: str = Field(..., max_length=100)
    district: str = Field(..., max_length=100)
    incident_type: str = Field(..., max_length=100)
    incident_date: str = Field(..., min_length=1)
    damage_description: str = Field(..., max_length=2000)
    claimed_amount: float = Field(..., gt=0)
    weather_event_ids: List[str] = Field(default_factory=list)


@router.get("/claims/autofill/{uid}", response_model=AutoFillResult)
async def autofill_claim(uid: str, request: Request):
    if _collection is None:
        raise HTTPException(status_code=500, detail="Not initialized")

    try:
        user_doc = _collection.document("users").document(uid).get()
        if not user_doc.exists:
            return AutoFillResult(fill_percentage=0.0)
    except Exception as e:
        logger.error("Error fetching user profile: %s", e)
        return AutoFillResult(fill_percentage=0.0)

    user_data = user_doc.to_dict() or {}
    filled = 0
    total = 6

    crop_name = user_data.get("cropType")
    confidence_crop = 80.0 if crop_name else 0.0
    if crop_name:
        filled += 1

    sowing_date = user_data.get("sowingDate")
    confidence_sowing = 70.0 if sowing_date else 0.0
    if sowing_date:
        filled += 1

    farm_location = user_data.get("address") or user_data.get("farmLocation")
    confidence_location = 75.0 if farm_location else 0.0
    if farm_location:
        filled += 1

    location = user_data.get("location", {})
    farm_area = user_data.get("areaAcres") or user_data.get("farmArea")
    confidence_area = 70.0 if farm_area else 0.0
    if farm_area:
        filled += 1

    village = user_data.get("village")
    confidence_village = 70.0 if village else 0.0
    if village:
        filled += 1

    district = user_data.get("district")
    confidence_district = 70.0 if district else 0.0
    if district:
        filled += 1

    weather_events = []
    lat = location.get("lat") if isinstance(location, dict) else None
    lng = location.get("lng") if isinstance(location, dict) else None
    if lat is not None and lng is not None and _weather_service is not None:
        try:
            end = datetime.now()
            start = end - timedelta(days=90)
            alerts = _weather_service.get_historical_alerts(
                lat=lat, lon=lng, start_date=start, end_date=end
            )
            weather_events = [
                {"event": a.get("event", "Unknown"), "date": a.get("date", ""), "severity": a.get("severity", "LOW")}
                for a in (alerts or [])
            ][:10]
        except Exception as e:
            logger.warning("Weather fetch failed: %s", e)

    fill_percentage = round((filled / total) * 100, 1)
    return AutoFillResult(
        crop_name=crop_name, confidence_crop=confidence_crop,
        sowing_date=sowing_date, confidence_sowing=confidence_sowing,
        farm_location=farm_location, confidence_location=confidence_location,
        farm_area=farm_area, confidence_area=confidence_area,
        village=village, confidence_village=confidence_village,
        district=district, confidence_district=confidence_district,
        weather_events=weather_events, fill_percentage=fill_percentage,
    )
