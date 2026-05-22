"""Reports & Logging Router"""
import re
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Optional
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Validation bounds — intentionally generous to accommodate large commercial
# farms while still rejecting obviously fabricated figures.
# ---------------------------------------------------------------------------
_PROFIT_MAX_INR = 50_000_000   # ₹5 crore per season
_AREA_MAX_ACRES = 10_000       # 10,000 acres

# Regex that matches a valid Indian-locale number string produced by the
# frontend (e.g. "50,000" or "1,00,000") or a plain integer string.
_NUMERIC_RE = re.compile(r"^[\d,]+(\.\d+)?$")


def _parse_inr(value: str) -> float:
    """Strip commas and parse as float. Raises ValueError on invalid input."""
    cleaned = value.replace(",", "").strip()
    if not cleaned:
        raise ValueError("Value is empty")
    return float(cleaned)


def _parse_acres(value: str) -> float:
    """Extract the numeric part from strings like '5 Acres' or '5.5'."""
    cleaned = value.lower().replace("acres", "").replace("acre", "").strip()
    cleaned = cleaned.replace(",", "")
    if not cleaned:
        raise ValueError("Value is empty")
    return float(cleaned)


class ClientErrorReport(BaseModel):
    message: str = Field(..., min_length=1, max_length=500)
    source: Optional[str] = Field(default=None, max_length=200)
    stack: Optional[str] = Field(default=None, max_length=2000)
    level: str = Field(default="error", max_length=20)

class ReportRequest(BaseModel):
    name: str = Field(..., max_length=100)
    crop: str = Field(..., max_length=50)
    area: str = Field(..., max_length=50)
    profit: str = Field(..., max_length=50)
    season: str = Field(..., max_length=50)

    @validator("profit")
    def validate_profit(cls, v):
        try:
            amount = _parse_inr(v)
        except (ValueError, TypeError):
            raise ValueError("Profit must be a valid number (e.g. 50000 or 50,000).")
        if amount < 0:
            raise ValueError("Profit cannot be negative.")
        if amount > _PROFIT_MAX_INR:
            raise ValueError(
                f"Profit cannot exceed ₹{_PROFIT_MAX_INR:,} per season. "
                "If your farm genuinely exceeds this, contact support."
            )
        return v

    @validator("area")
    def validate_area(cls, v):
        try:
            acres = _parse_acres(v)
        except (ValueError, TypeError):
            raise ValueError("Farm area must be a valid number of acres (e.g. '5 Acres' or '5.5').")
        if acres <= 0:
            raise ValueError("Farm area must be greater than zero.")
        if acres > _AREA_MAX_ACRES:
            raise ValueError(
                f"Farm area cannot exceed {_AREA_MAX_ACRES:,} acres. "
                "If your farm genuinely exceeds this, contact support."
            )
        return v

verify_role_fn = None
get_signing_keys_fn = None
sanitise_log_field_fn = None
logger_instance = None

def init_reports(vr_fn, gsk_fn, slf_fn, log_inst):
    global verify_role_fn, get_signing_keys_fn, sanitise_log_field_fn, logger_instance
    verify_role_fn = vr_fn
    get_signing_keys_fn = gsk_fn
    sanitise_log_field_fn = slf_fn
    logger_instance = log_inst

@router.post("/reports/generate")
async def generate_signed_report(request: Request, data: ReportRequest):
    if not all([verify_role_fn, get_signing_keys_fn]):
        raise HTTPException(status_code=500, detail="Not initialized")
    try:
        await verify_role_fn(request)
        return {"success": True, "message": "Report generated", "data": data.model_dump()}
    except Exception as e:
        logger.error(f"Report error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/log-error")
async def log_error(request: Request, body: ClientErrorReport):
    if sanitise_log_field_fn is None or logger_instance is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    try:
        message = sanitise_log_field_fn(body.message)
        source = sanitise_log_field_fn(body.source or "")
        level = sanitise_log_field_fn(body.level).upper()
        logger_instance.info(f"Client [{level}] from {source}: {message}")
        return {"success": True, "message": "Error logged"}
    except Exception as e:
        logger.error(f"Log error: {e}")
        raise HTTPException(status_code=500, detail="Failed to log error")
