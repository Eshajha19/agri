"""Reports & Logging Router"""
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

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
async def log_error(body: ClientErrorReport):
    if sanitise_log_field_fn is None:
        raise HTTPException(status_code=500, detail="Log sanitizer not initialized")

    level = sanitise_log_field_fn(body.level).lower()
    message = sanitise_log_field_fn(body.message)
    source = sanitise_log_field_fn(body.source) if body.source else "unknown"
    stack = sanitise_log_field_fn(body.stack) if body.stack else ""

    log_fn = {
        "error": logger.error,
        "warn": logger.warning,
        "warning": logger.warning,
        "info": logger.info,
    }.get(level, logger.error)

    log_fn(
        "[ClientError] level=%s source=%s message=%s%s",
        level,
        source,
        message,
        f" stack={stack}" if stack else "",
    )
    return {"success": True}
