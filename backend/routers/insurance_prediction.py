"""Claim Success Probability Predictor — scores claims based on evidence, weather, and historical patterns."""

import logging
import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()

_collection = None
_weather_service = None


def init_insurance_prediction(firestore_collection, weather_svc=None):
    global _collection, _weather_service
    _collection = firestore_collection
    _weather_service = weather_svc


class ScoreResult(BaseModel):
    claim_strength_score: float
    approval_probability: str
    approval_label: str
    factors: List[Dict[str, Any]]


WEIGHT_POLICY = 0.20
WEIGHT_CROP = 0.15
WEIGHT_WEATHER = 0.25
WEIGHT_EVIDENCE = 0.20
WEIGHT_HISTORICAL = 0.20


def _score_policy_eligibility(user_data: dict) -> tuple:
    profile_complete = all(user_data.get(k) for k in ("cropType", "sowingDate", "areaAcres", "village", "district"))
    score = 100 if profile_complete else 40
    reasons = []
    if profile_complete:
        reasons.append("Full farm profile on file")
    else:
        missing = [k for k in ("cropType", "sowingDate", "areaAcres", "village", "district") if not user_data.get(k)]
        reasons.append(f"Missing profile fields: {', '.join(missing)}")
    return score, reasons


def _score_crop_type(crop_name: str) -> tuple:
    HIGH_RISK = {"paddy", "rice", "wheat", "maize", "sugarcane"}
    MEDIUM_RISK = {"cotton", "soybean", "groundnut", "potato", "tomato"}
    name = (crop_name or "").lower().strip()
    if name in HIGH_RISK:
        return 90, [f"Crop '{crop_name}' has high insurance eligibility"]
    if name in MEDIUM_RISK:
        return 70, [f"Crop '{crop_name}' has standard insurance coverage"]
    if name:
        return 50, [f"Crop '{crop_name}' has limited insurance data"]
    return 30, ["Crop type not provided"]


def _score_weather_events(events: list, incident_type: str) -> tuple:
    if not events:
        return 30, ["No recent weather events on record for your farm location"]
    incident_lower = (incident_type or "").lower()
    matching = [e for e in events if incident_lower in e.get("event", "").lower()]
    if matching:
        severe = any(e.get("severity", "").upper() in ("HIGH", "SEVERE") for e in matching)
        score = 95 if severe else 80
        reason = f"{len(matching)} matching weather event(s) found (severity: {'high' if severe else 'moderate'})"
        return score, [reason]
    return 50, [f"Weather events found but none match '{incident_type}'"]


def _score_evidence_completeness(description: str, claimed_amount: float) -> tuple:
    score = 0
    reasons = []
    if description and len(description) > 20:
        score += 50
        reasons.append("Damage description provided with sufficient detail")
    else:
        reasons.append("Damage description is too short or missing")
    if claimed_amount > 0:
        score += 30
        reasons.append("Claimed amount specified")
    else:
        reasons.append("Claimed amount not specified")
    if description and any(c.isdigit() for c in description):
        score += 10
        reasons.append("Quantifiable data detected in description")
    score = min(score, 100)
    return score, reasons


def _score_historical_trends(uid: str) -> tuple:
    if _collection is None:
        return 50, ["Historical claim data unavailable"]
    try:
        claims_ref = _collection.document("claims").where("uid", "==", uid).get()
        claims = list(claims_ref) if hasattr(claims_ref, "__iter__") else []
        if not claims:
            return 70, ["No prior claims — first-time filer bonus applied"]
        approved = sum(1 for c in claims if c.to_dict().get("status") == "approved")
        total = len(claims)
        ratio = approved / total if total > 0 else 0
        score = int(ratio * 100)
        return score, [f"{approved}/{total} prior claims approved ({score}% approval rate)"]
    except Exception as e:
        logger.warning("Historical lookup failed: %s", e)
        return 50, ["Could not retrieve historical claim data"]


def calculate_score(user_data: dict, events: list, incident_type: str, description: str, claimed_amount: float, uid: str) -> ScoreResult:
    policy_score, policy_reasons = _score_policy_eligibility(user_data)
    crop_score, crop_reasons = _score_crop_type(user_data.get("cropType", ""))
    weather_score, weather_reasons = _score_weather_events(events, incident_type)
    evidence_score, evidence_reasons = _score_evidence_completeness(description, claimed_amount)
    historical_score, historical_reasons = _score_historical_trends(uid)

    weighted = (
        policy_score * WEIGHT_POLICY
        + crop_score * WEIGHT_CROP
        + weather_score * WEIGHT_WEATHER
        + evidence_score * WEIGHT_EVIDENCE
        + historical_score * WEIGHT_HISTORICAL
    )
    overall = round(min(weighted, 100), 1)

    if overall >= 75:
        label = "High Chance of Approval"
        prob = "High"
    elif overall >= 50:
        label = "Moderate Chance of Approval"
        prob = "Moderate"
    elif overall >= 25:
        label = "Low Chance of Approval"
        prob = "Low"
    else:
        label = "Very Low Chance of Approval"
        prob = "Very Low"

    factors = [
        {"name": "Policy eligibility", "score": policy_score, "weight": WEIGHT_POLICY, "details": policy_reasons},
        {"name": "Crop type", "score": crop_score, "weight": WEIGHT_CROP, "details": crop_reasons},
        {"name": "Weather event verification", "score": weather_score, "weight": WEATHER_WEIGHT, "details": weather_reasons},
        {"name": "Evidence completeness", "score": evidence_score, "weight": WEIGHT_EVIDENCE, "details": evidence_reasons},
        {"name": "Historical claim trends", "score": historical_score, "weight": WEIGHT_HISTORICAL, "details": historical_reasons},
    ]

    return ScoreResult(
        claim_strength_score=overall,
        approval_probability=prob,
        approval_label=label,
        factors=factors,
    )


class PredictRequest(BaseModel):
    uid: str = Field(..., min_length=1)
    incident_type: str = Field(..., max_length=100)
    incident_date: str = Field(..., min_length=1)
    damage_description: str = Field(..., max_length=2000)
    claimed_amount: float = Field(..., gt=0)


@router.post("/claims/predict", response_model=ScoreResult)
async def predict_claim(body: PredictRequest):
    if _collection is None:
        raise HTTPException(status_code=500, detail="Not initialized")

    uid = body.uid
    try:
        user_doc = _collection.document("users").document(uid).get()
        if not user_doc.exists:
            user_data = {}
        else:
            user_data = user_doc.to_dict() or {}
    except Exception as e:
        logger.error("Error fetching user: %s", e)
        user_data = {}

    location = user_data.get("location", {})
    lat = location.get("lat") if isinstance(location, dict) else None
    lng = location.get("lng") if isinstance(location, dict) else None
    events = []
    if lat is not None and lng is not None and _weather_service is not None:
        try:
            end = datetime.now()
            start = end - timedelta(days=90)
            alerts = _weather_service.get_historical_alerts(lat=lat, lon=lng, start_date=start, end_date=end)
            events = [
                {"event": a.get("event", ""), "date": a.get("date", ""), "severity": a.get("severity", "LOW")}
                for a in (alerts or [])
            ][:10]
        except Exception:
            logger.warning("Weather fetch failed for prediction")

    result = calculate_score(
        user_data=user_data,
        events=events,
        incident_type=body.incident_type,
        description=body.damage_description,
        claimed_amount=body.claimed_amount,
        uid=uid,
    )
    return result
