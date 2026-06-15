"""Shared request schema for alert-triggering routes."""

from typing import Optional

from pydantic import BaseModel, Field


class AlertTriggerRequest(BaseModel):
    alert_type: str = Field(..., pattern=r"^(weather|pest|advisory)$")
    message: str = Field(..., min_length=1, max_length=500)
    region_id: Optional[str] = Field(default=None, max_length=100)


class AlertSummary(BaseModel):
    """Standardized alert response schema."""
    alert_type: str = Field(..., pattern=r"^(weather|pest|advisory)$")
    message: str = Field(..., min_length=1, max_length=500)
    severity: str = Field(default="Medium", pattern=r"^(Low|Medium|High)$")
    timestamp: Optional[str] = None
    region_id: Optional[str] = None
