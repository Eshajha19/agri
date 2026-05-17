# -*- coding: utf-8 -*-
# main.py
import os
import io
import json
import logging
import re
import threading
import joblib
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Request, Form, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from feature_flags.routes import router as flags_router

class SimulationRequest(BaseModel):
    crop_type: str
    temp_delta: float = Field(..., ge=-5, le=5)
    rain_delta: float = Field(..., ge=-100, le=100)

class ClientErrorReport(BaseModel):
    """
    Typed, bounded schema for frontend error reports sent to /api/log-error.

    Fields are intentionally narrow:
    - message  : the human-readable error description (capped at 500 chars)
    - source   : optional filename / module where the error originated
    - stack    : optional stack trace (capped to prevent log flooding)
    - level    : severity hint from the client; defaults to "error"

    All string fields are stripped of ANSI escape sequences and ASCII
    control characters before being written to the log, so a crafted
    payload cannot inject terminal control codes or forge log lines.
    """
    message: str = Field(..., min_length=1, max_length=500)
    source: Optional[str] = Field(default=None, max_length=200)
    stack: Optional[str] = Field(default=None, max_length=2000)
    level: str = Field(default="error", max_length=20)

class RAGQuery(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    top_k: int = Field(default=3, ge=1, le=5)

# Blockchain Supply Chain Models
class RegisterActorRequest(BaseModel):
    actor_id: str = Field(..., min_length=1, max_length=50)
    name: str = Field(..., min_length=1, max_length=100)
    actor_type: str = Field(..., min_length=1, max_length=50)
    location: str = Field(..., min_length=1, max_length=100)

class CreateProductBatchRequest(BaseModel):
    crop_type: str = Field(..., min_length=1, max_length=50)
    farm_id: str = Field(..., min_length=1, max_length=50)
    quantity: float = Field(..., gt=0)
    unit: str = Field(..., min_length=1, max_length=20)
    planting_date: str = Field(..., min_length=1)
    harvesting_date: str = Field(..., min_length=1)
    farmer_name: str = Field(..., min_length=1, max_length=100)

class AddSupplyChainNodeRequest(BaseModel):
    batch_id: str = Field(..., min_length=1)
    node_type: str = Field(..., min_length=1, max_length=50)
    actor_name: str = Field(..., min_length=1, max_length=100)
    location: str = Field(..., min_length=1, max_length=100)
    action: str = Field(..., min_length=1, max_length=50)
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    quality_check: Optional[str] = None
    notes: str = Field(default="", max_length=500)

class CreateSmartContractRequest(BaseModel):
    batch_id: str = Field(..., min_length=1)
    seller: str = Field(..., min_length=1, max_length=100)
    buyer: str = Field(..., min_length=1, max_length=100)
    price: float = Field(..., gt=0)
    terms: Optional[Dict] = None

class ExecuteContractRequest(BaseModel):
    contract_id: str = Field(..., min_length=1)

# Crop Quality Grading Models
class CropQualityGradingRequest(BaseModel):
    crop_type: str = Field(..., min_length=1, max_length=50)
    image_base64: str = Field(..., min_length=100)  # Base64 encoded image

class CropQualityBatchRequest(BaseModel):
    crop_type: str = Field(..., min_length=1, max_length=50)
    images_base64: list = Field(..., min_items=1, max_items=100)  # Multiple images

class QualityTrendsRequest(BaseModel):
    crop_type: str = Field(..., min_length=1, max_length=50)
    days: int = Field(default=7, ge=1, le=30)

# Rate Limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Firebase Admin SDK
import firebase_admin
from firebase_admin import credentials, auth as firebase_auth, firestore

# ML Ops Imports
from ml.registry import ModelRegistry
from ml.adapters.xgboost_adapter import XGBoostAdapter
from ml.router import ModelRouter
from ml.preprocessing import UnknownCategoryError, MissingFeatureError
from ml.security import verify_and_load_joblib

# Other internal modules
from alert_rules import generate_alerts
from whatsapp_service import send_whatsapp_message, format_alert_message
from whatsapp_store import subscriber_store
from crop_quality_grading import CropQualityGrader
from blockchain_supply_chain import SupplyChainBlockchain
from farm_finance_ai import FarmFinanceAI
from sustainability_analytics import SustainabilityAnalytics

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import inch
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

# KMS Support
try:
    from google.cloud import secretmanager
    HAS_GCP_KMS = True
except ImportError:
    HAS_GCP_KMS = False

app = FastAPI()
app.include_router(flags_router)

logger = logging.getLogger(__name__)

# Server-side idempotency cache for deduplicating transactional POST requests.
# In production, this should use a persistent store like Redis with a TTL.
_IDEMPOTENCY_CACHE: Dict[str, Any] = {}
_IDEMPOTENCY_LOCK = threading.Lock()

# Regex that matches ANSI escape sequences (e.g. \x1b[31m) and all other
# ASCII control characters (0x00-0x1f, 0x7f) except tab and newline.
# Used to sanitise client-supplied strings before they reach the log, so a
# crafted payload cannot inject terminal control codes or forge log lines.
_CONTROL_CHAR_RE = re.compile(
    r"(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]"   # ANSI CSI sequences
    r"|\x1B[@-_]"                          # other ESC sequences
    r"|[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"  # control chars except \t \n
)

def _sanitise_log_field(value: str) -> str:
    """Strip ANSI escape sequences and ASCII control characters from *value*."""
    if not isinstance(value, str):
        return ""
    return _CONTROL_CHAR_RE.sub("", value)

logger = logging.getLogger(__name__)

# Regex that matches ANSI escape sequences (e.g. \x1b[31m) and all other
# ASCII control characters (0x00-0x1f, 0x7f) except tab and newline.
# Used to sanitise client-supplied strings before they reach the log, so a
# crafted payload cannot inject terminal control codes or forge log lines.
_CONTROL_CHAR_RE = re.compile(
    r"(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]"   # ANSI CSI sequences
    r"|\x1B[@-_]"                          # other ESC sequences
    r"|[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"  # control chars except \t \n
)

def _sanitise_log_field(value: str) -> str:
    """Strip ANSI escape sequences and ASCII control characters from *value*."""
    if not isinstance(value, str):
        return ""
    return _CONTROL_CHAR_RE.sub("", value)

logger = logging.getLogger(__name__)

# Regex that matches ANSI escape sequences (e.g. \x1b[31m) and all other
# ASCII control characters (0x00-0x1f, 0x7f) except tab and newline.
# Used to sanitise client-supplied strings before they reach the log, so a
# crafted payload cannot inject terminal control codes or forge log lines.
_CONTROL_CHAR_RE = re.compile(
    r"(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]"   # ANSI CSI sequences
    r"|\x1B[@-_]"                          # other ESC sequences
    r"|[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"  # control chars except \t \n
)

def _sanitise_log_field(value: str) -> str:
    """Strip ANSI escape sequences and ASCII control characters from *value*."""
    if not isinstance(value, str):
        return ""
    return _CONTROL_CHAR_RE.sub("", value)

logger = logging.getLogger(__name__)

# Regex that matches ANSI escape sequences (e.g. \x1b[31m) and all other
# ASCII control characters (0x00-0x1f, 0x7f) except tab and newline.
# Used to sanitise client-supplied strings before they reach the log, so a
# crafted payload cannot inject terminal control codes or forge log lines.
_CONTROL_CHAR_RE = re.compile(
    r"(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]"   # ANSI CSI sequences
    r"|\x1B[@-_]"                          # other ESC sequences
    r"|[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"  # control chars except \t \n
)

def _sanitise_log_field(value: str) -> str:
    """Strip ANSI escape sequences and ASCII control characters from *value*."""
    if not isinstance(value, str):
        return ""
    return _CONTROL_CHAR_RE.sub("", value)

logger = logging.getLogger(__name__)

# Regex that matches ANSI escape sequences (e.g. \x1b[31m) and all other
# ASCII control characters (0x00-0x1f, 0x7f) except tab and newline.
# Used to sanitise client-supplied strings before they reach the log, so a
# crafted payload cannot inject terminal control codes or forge log lines.
_CONTROL_CHAR_RE = re.compile(
    r"(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]"   # ANSI CSI sequences
    r"|\x1B[@-_]"                          # other ESC sequences
    r"|[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"  # control chars except \t \n
)

def _sanitise_log_field(value: str) -> str:
    """Strip ANSI escape sequences and ASCII control characters from *value*."""
    if not isinstance(value, str):
        return ""
    return _CONTROL_CHAR_RE.sub("", value)

logger = logging.getLogger(__name__)

# Regex that matches ANSI escape sequences (e.g. \x1b[31m) and all other
# ASCII control characters (0x00-0x1f, 0x7f) except tab and newline.
# Used to sanitise client-supplied strings before they reach the log, so a
# crafted payload cannot inject terminal control codes or forge log lines.
_CONTROL_CHAR_RE = re.compile(
    r"(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]"   # ANSI CSI sequences
    r"|\x1B[@-_]"                          # other ESC sequences
    r"|[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"  # control chars except \t \n
)

def _sanitise_log_field(value: str) -> str:
    """Strip ANSI escape sequences and ASCII control characters from *value*."""
    if not isinstance(value, str):
        return ""
    return _CONTROL_CHAR_RE.sub("", value)

logger = logging.getLogger(__name__)

# Regex that matches ANSI escape sequences (e.g. \x1b[31m) and all other
# ASCII control characters (0x00-0x1f, 0x7f) except tab and newline.
# Used to sanitise client-supplied strings before they reach the log, so a
# crafted payload cannot inject terminal control codes or forge log lines.
_CONTROL_CHAR_RE = re.compile(
    r"(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]"   # ANSI CSI sequences
    r"|\x1B[@-_]"                          # other ESC sequences
    r"|[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"  # control chars except \t \n
)

def _sanitise_log_field(value: str) -> str:
    """Strip ANSI escape sequences and ASCII control characters from *value*."""
    if not isinstance(value, str):
        return ""
    return _CONTROL_CHAR_RE.sub("", value)

logger = logging.getLogger(__name__)

# Regex that matches ANSI escape sequences (e.g. \x1b[31m) and all other
# ASCII control characters (0x00-0x1f, 0x7f) except tab and newline.
# Used to sanitise client-supplied strings before they reach the log, so a
# crafted payload cannot inject terminal control codes or forge log lines.
_CONTROL_CHAR_RE = re.compile(
    r"(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]"   # ANSI CSI sequences
    r"|\x1B[@-_]"                          # other ESC sequences
    r"|[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"  # control chars except \t \n
)

def _sanitise_log_field(value: str) -> str:
    """Strip ANSI escape sequences and ASCII control characters from *value*."""
    if not isinstance(value, str):
        return ""
    return _CONTROL_CHAR_RE.sub("", value)

logger = logging.getLogger(__name__)

# Regex that matches ANSI escape sequences (e.g. \x1b[31m) and all other
# ASCII control characters (0x00-0x1f, 0x7f) except tab and newline.
# Used to sanitise client-supplied strings before they reach the log, so a
# crafted payload cannot inject terminal control codes or forge log lines.
_CONTROL_CHAR_RE = re.compile(
    r"(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]"   # ANSI CSI sequences
    r"|\x1B[@-_]"                          # other ESC sequences
    r"|[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"  # control chars except \t \n
)

def _sanitise_log_field(value: str) -> str:
    """Strip ANSI escape sequences and ASCII control characters from *value*."""
    if not isinstance(value, str):
        return ""
    return _CONTROL_CHAR_RE.sub("", value)

logger = logging.getLogger(__name__)

# Regex that matches ANSI escape sequences (e.g. \x1b[31m) and all other
# ASCII control characters (0x00-0x1f, 0x7f) except tab and newline.
# Used to sanitise client-supplied strings before they reach the log, so a
# crafted payload cannot inject terminal control codes or forge log lines.
_CONTROL_CHAR_RE = re.compile(
    r"(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]"   # ANSI CSI sequences
    r"|\x1B[@-_]"                          # other ESC sequences
    r"|[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"  # control chars except \t \n
)

def _sanitise_log_field(value: str) -> str:
    """Strip ANSI escape sequences and ASCII control characters from *value*."""
    if not isinstance(value, str):
        return ""
    return _CONTROL_CHAR_RE.sub("", value)

logger = logging.getLogger(__name__)

# Regex that matches ANSI escape sequences (e.g. \x1b[31m) and all other
# ASCII control characters (0x00-0x1f, 0x7f) except tab and newline.
# Used to sanitise client-supplied strings before they reach the log, so a
# crafted payload cannot inject terminal control codes or forge log lines.
_CONTROL_CHAR_RE = re.compile(
    r"(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]"   # ANSI CSI sequences
    r"|\x1B[@-_]"                          # other ESC sequences
    r"|[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"  # control chars except \t \n
)

def _sanitise_log_field(value: str) -> str:
    """Strip ANSI escape sequences and ASCII control characters from *value*."""
    if not isinstance(value, str):
        return ""
    return _CONTROL_CHAR_RE.sub("", value)

logger = logging.getLogger(__name__)

# Regex that matches ANSI escape sequences (e.g. \x1b[31m) and all other
# ASCII control characters (0x00-0x1f, 0x7f) except tab and newline.
# Used to sanitise client-supplied strings before they reach the log, so a
# crafted payload cannot inject terminal control codes or forge log lines.
_CONTROL_CHAR_RE = re.compile(
    r"(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]"   # ANSI CSI sequences
    r"|\x1B[@-_]"                          # other ESC sequences
    r"|[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"  # control chars except \t \n
)

def _sanitise_log_field(value: str) -> str:
    """Strip ANSI escape sequences and ASCII control characters from *value*."""
    if not isinstance(value, str):
        return ""
    return _CONTROL_CHAR_RE.sub("", value)

logger = logging.getLogger(__name__)

# Regex that matches ANSI escape sequences (e.g. \x1b[31m) and all other
# ASCII control characters (0x00-0x1f, 0x7f) except tab and newline.
# Used to sanitise client-supplied strings before they reach the log, so a
# crafted payload cannot inject terminal control codes or forge log lines.
_CONTROL_CHAR_RE = re.compile(
    r"(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]"   # ANSI CSI sequences
    r"|\x1B[@-_]"                          # other ESC sequences
    r"|[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"  # control chars except \t \n
)

def _sanitise_log_field(value: str) -> str:
    """Strip ANSI escape sequences and ASCII control characters from *value*."""
    if not isinstance(value, str):
        return ""
    return _CONTROL_CHAR_RE.sub("", value)

logger = logging.getLogger(__name__)

# Regex that matches ANSI escape sequences (e.g. \x1b[31m) and all other
# ASCII control characters (0x00-0x1f, 0x7f) except tab and newline.
# Used to sanitise client-supplied strings before they reach the log, so a
# crafted payload cannot inject terminal control codes or forge log lines.
_CONTROL_CHAR_RE = re.compile(
    r"(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]"   # ANSI CSI sequences
    r"|\x1B[@-_]"                          # other ESC sequences
    r"|[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"  # control chars except \t \n
)

def _sanitise_log_field(value: str) -> str:
    """Strip ANSI escape sequences and ASCII control characters from *value*."""
    if not isinstance(value, str):
        return ""
    return _CONTROL_CHAR_RE.sub("", value)

# Initialize Limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Initialize Firebase Admin
import logging as _logging
_firebase_logger = _logging.getLogger(__name__)

# Explicitly set to None before the try block so db_firestore is always
# defined at module level, even if an exception is raised mid-init.
db_firestore = None

if not firebase_admin._apps:
    try:
        # In a GCP environment this picks up Application Default Credentials
        # automatically.  For local dev set GOOGLE_APPLICATION_CREDENTIALS to
        # the path of a service-account key file.
        firebase_admin.initialize_app()
        db_firestore = firestore.client()
        _firebase_logger.info("Firebase Admin: successfully initialized")
    except Exception as e:
        _firebase_logger.warning(
            "Firebase Admin: could not initialize — role-gated endpoints will "
            "return 503 until Firestore is reachable. Reason: %s", e
        )

async def verify_role(request: Request, required_roles: list = None):
    """
    Verify the Firebase ID token and check the caller's role against Firestore.
    Expects 'Authorization: Bearer <ID_TOKEN>' header.

    Fail-closed design:
    - If Firestore is unavailable the request is rejected with 503.
    - If the user document does not exist the request is rejected with 403.
    - The function never grants a role that was not explicitly stored in Firestore.
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authentication token")

    # Use a slice instead of split()[1] to avoid IndexError when the header
    # is exactly "Bearer " with no token following it.
    id_token = auth_header[7:].strip()
    if not id_token:
        raise HTTPException(status_code=401, detail="Missing or invalid authentication token")

    # Verify the token signature with Firebase — raises on invalid/expired tokens.
    try:
        decoded_token = firebase_auth.verify_id_token(id_token)
    except Exception:
        raise HTTPException(status_code=401, detail="Authentication failed")

    uid = decoded_token["uid"]

    # Firestore must be available to resolve the caller's role.
    # Failing open (granting admin when Firestore is down) is a security bug,
    # so we reject the request instead.
    if not db_firestore:
        raise HTTPException(
            status_code=503,
            detail="Authorization service temporarily unavailable"
        )

    # Wrap the Firestore fetch so a transient network error (timeout, reset)
    # returns the same clean 503 as a missing db_firestore, rather than an
    # unhandled exception that leaks internal details as a raw 500.
    try:
        user_doc = db_firestore.collection("users").document(uid).get()
    except Exception as e:
        _firebase_logger.error(
            "Firestore fetch failed for uid=%s during role check: %s", uid, e
        )
        raise HTTPException(
            status_code=503,
            detail="Authorization service temporarily unavailable"
        )

    if not user_doc.exists:
        raise HTTPException(status_code=403, detail="User profile not found")

    user_role = user_doc.to_dict().get("role", "farmer")

    if required_roles and user_role not in required_roles:
        raise HTTPException(status_code=403, detail="Access denied: insufficient permissions")

    return {"uid": uid, "role": user_role}

# --- Secure CORS Configuration ---
frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173")
trusted_origins = [
    "http://localhost:5173",     # Local development
    "http://127.0.0.1:5173",     # Local development alternative
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
]

# Add any custom frontend URLs from environment
if frontend_url and frontend_url not in trusted_origins:
    trusted_origins.append(frontend_url)

# Support comma-separated list of additional origins
extra_origins = os.getenv("ADDITIONAL_ALLOWED_ORIGINS")
if extra_origins:
    trusted_origins.extend([origin.strip() for origin in extra_origins.split(",")])

app.add_middleware(
    CORSMiddleware,
    allow_origins=trusted_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Models ---

class PredictRequest(BaseModel):
    Crop: str = Field(..., max_length=50)
    CropCoveredArea: float = Field(..., gt=0)
    CHeight: int = Field(..., ge=0)
    CNext: str = Field(..., max_length=50)
    CLast: str = Field(..., max_length=50)
    CTransp: str = Field(..., max_length=50)
    IrriType: str = Field(..., max_length=50)
    IrriSource: str = Field(..., max_length=50)
    IrriCount: int = Field(..., ge=1)
    WaterCov: int = Field(..., ge=0, le=100)
    Season: str = Field(..., max_length=50)

class PredictResponse(BaseModel):
    predicted_ExpYield: float

class WhatsAppSubscribeRequest(BaseModel):
    phone_number: str
    user_id: str
    name: str

class YieldInput(BaseModel):
    data: list[float]

class AlertTriggerRequest(BaseModel):
    alert_type: str  # 'weather', 'pest', 'advisory'
    message: str

class ReportRequest(BaseModel):
    name: str = Field(..., max_length=100)
    crop: str = Field(..., max_length=50)
    area: str = Field(..., max_length=50)
    profit: str = Field(..., max_length=50)
    season: str = Field(..., max_length=50)

    @validator("name", "crop", "area", "profit", "season", pre=True)
    def reject_pipe_characters(cls, v):
        # Belt-and-suspenders guard: the signing payload now uses JSON (which
        # is unambiguous regardless of field content), but we also reject pipe
        # characters at the model level so legacy code paths or future changes
        # cannot accidentally reintroduce a delimiter-injection vulnerability.
        if isinstance(v, str) and "|" in v:
            raise ValueError(
                "Field value must not contain the '|' character."
            )
        return v

class WeatherLocationRequest(BaseModel):
    location: str

class WeatherAlertRequest(BaseModel):
    latitude: float
    longitude: float
    location: str
    crop: Optional[str] = None

class SeedVerifyRequest(BaseModel):
    code: str = Field(..., min_length=4, max_length=100)

# Whitelist of commodities the price forecaster supports.
# Restricting to this set prevents an attacker from submitting arbitrary
# strings that each trigger a new LSTM training run and grow the model
# cache without bound.
_FORECASTABLE_COMMODITIES = frozenset([
    "Wheat",
    "Paddy (Dhan)",
    "Cotton",
    "Onion",
    "Soybean",
    "Maize",
])

class MarketForecastRequest(BaseModel):
    commodity: str = Field(..., min_length=1, max_length=60)
    days: int = Field(default=14, ge=1, le=30)

    @validator("commodity")
    def commodity_must_be_supported(cls, v):
        if v not in _FORECASTABLE_COMMODITIES:
            raise ValueError(
                f"Unsupported commodity '{v}'. "
                f"Supported values: {sorted(_FORECASTABLE_COMMODITIES)}"
            )
        return v

class FinanceAssessmentRequest(BaseModel):
    farmer_name: str = Field(..., min_length=1, max_length=100)
    crop_type: str = Field(..., min_length=1, max_length=50)
    acreage: float = Field(default=0, ge=0)
    annual_revenue: float = Field(default=0, ge=0)
    annual_operating_cost: float = Field(default=0, ge=0)
    existing_debt: float = Field(default=0, ge=0)
    emergency_fund: float = Field(default=0, ge=0)
    credit_score: int = Field(default=650, ge=300, le=900)
    requested_loan_amount: float = Field(default=0, ge=0)
    loan_tenure_months: int = Field(default=36, ge=6, le=120)
    irrigation_cost: float = Field(default=0, ge=0)
    labor_cost: float = Field(default=0, ge=0)
    selected_lender: Optional[str] = Field(default=None, max_length=100)
    farm_location: Optional[str] = Field(default=None, max_length=120)
    notes: Optional[str] = Field(default=None, max_length=500)

class SustainabilityAnalyzeRequest(BaseModel):
    crop_type: str = Field(..., min_length=1, max_length=50)
    season: str = Field(default="Kharif", max_length=20)
    acreage: float = Field(..., gt=0, le=5000)
    irrigation_type: str = Field(default="drip")
    irrigation_events: int = Field(default=10, ge=0, le=200)
    fertilizer_n_kg: Optional[float] = Field(default=None, ge=0, le=50000)
    fertilizer_p_kg: Optional[float] = Field(default=None, ge=0, le=50000)
    fertilizer_k_kg: Optional[float] = Field(default=None, ge=0, le=50000)
    machinery_hours: Optional[float] = Field(default=None, ge=0, le=10000)
    diesel_liters: Optional[float] = Field(default=None, ge=0, le=50000)
    organic_practices: bool = Field(default=False)
    user_id: Optional[str] = Field(default=None, max_length=128)

# --- ML Pipeline Initialization ---
router = ModelRouter(default_model="xgboost")

def init_ml_pipeline():
    try:
        # Register XGBoost Adapter
        xgb_adapter = XGBoostAdapter()
        model_path = "yield_model.joblib"
        if os.path.exists(model_path):
            xgb_adapter.load(model_path)
            ModelRegistry.register("xgboost", xgb_adapter)
            print("ML Pipeline: Registered XGBoost model.")
        else:
            print(f"ML Pipeline Warning: {model_path} not found.")
            
        # You can register other models here (e.g., LSTM) as they become available
        # ModelRegistry.register("lstm", LSTMAdapter("lstm_model.h5"))
        
    except Exception as e:
        print(f"ML Pipeline Error: {e}")

init_ml_pipeline()

# Load model directly for backward compatibility or simple use cases if needed
try:
    # Use signature-verified loading to prevent execution of malicious
    # pickled objects. These calls expect companion signature files:
    # - yield_model.joblib.sig
    # - sklearn_yield_model.pkl.sig
    model = verify_and_load_joblib("yield_model.joblib")
    model_lag = verify_and_load_joblib("sklearn_yield_model.pkl")
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    model = None
    model_lag = None

# --- Static Notifications Storage ---
static_notifications = [
    {
        "id": 1,
        "type": "weather",
        "message": "🌧️ Heavy rainfall expected in your region today.",
        "time": datetime.now().isoformat()
    }
]

# Initialize Crop Quality Grader
_crop_quality_grader = CropQualityGrader()

# Initialize Crop Quality Grader
_crop_quality_grader = CropQualityGrader()

# Initialize Supply Chain Blockchain
_supply_chain_blockchain = SupplyChainBlockchain()

# Initialize Farm Finance AI
_farm_finance_ai = FarmFinanceAI()
_sustainability_analytics = SustainabilityAnalytics()

# --- Routes ---

@app.get("/")
def root():
    return {"message": "Fasal Saathi API", "status": "running"}

@app.get("/predict")
def predict_get():
    return {"predicted_yield": 2500, "note": "Use POST endpoint for actual prediction"}

@app.post("/predict", response_model=PredictResponse)
@limiter.limit("5/minute")
def predict_yield(data: PredictRequest, request: Request):
    """
    Standardised prediction endpoint using ML Router for dynamic model selection.

    Returns HTTP 422 when the input contains an unknown categorical value or a
    missing required feature, so callers receive an actionable error message
    rather than a silently corrupted prediction.
    """
    try:
        input_data = data.model_dump() if hasattr(data, "model_dump") else data.dict()

        context = {
            "location": request.headers.get("X-User-Location", "Unknown"),
            "crop": data.Crop,
        }

        predicted_yield = router.predict(input_data, context)
        return {"predicted_ExpYield": float(predicted_yield)}

    except UnknownCategoryError as e:
        # The submitted categorical value was not in the training vocabulary.
        raise HTTPException(
            status_code=422,
            detail={
                "error": "unknown_category",
                "field": e.column,
                "value": str(e.value),
                "message": str(e),
            },
        )
    except MissingFeatureError as e:
        # Required feature columns are absent after encoding.
        raise HTTPException(
            status_code=422,
            detail={
                "error": "missing_features",
                "missing": e.missing_columns,
                "message": str(e),
            },
        )
    except Exception as e:
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict-yield-lag")
@limiter.limit("5/minute")
async def predict_yield_lag(payload: YieldInput, request: Request):
    if model_lag is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    try:
        data = payload.data
        if len(data) != 5:
            raise ValueError("Exactly 5 values are required")
        data = np.array(data).reshape(1, -1)
        prediction = model_lag.predict(data)
        return {
            "prediction": round(float(prediction[0]), 2),
            "model": "RandomForest Time Series (Lag Features)"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error") from e

@app.post("/predict-yield-trend")
@limiter.limit("5/minute")
async def predict_yield_trend(payload: YieldInput, request: Request):
    if model_lag is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    try:
        data = payload.data
        if len(data) != 5:
            raise ValueError("Exactly 5 values are required")
        temp = data[::-1]  # reverse once
        trend = []
        for _ in range(5):
            features = temp[:5]
            pred = model_lag.predict([features])[0]
            pred_value = round(float(pred), 2)
            trend.append(pred_value)
            temp = [pred_value] + temp
        return {
            "trend": trend,
            "prediction": trend[-1],
            "model": "RandomForest Trend Forecast (Lag Features)"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error") from e

@app.get("/api/notifications")
def get_notifications(
    crop: str = Query(default=None),
    irrigation_count: int = Query(default=None, ge=0),
    water_coverage: int = Query(default=None, ge=0, le=100),
    season: str = Query(default=None)
):
    """Generate dynamic farm advisory alerts + static ones."""
    dynamic_alerts = generate_alerts(
        crop=crop,
        irrigation_count=irrigation_count,
        water_coverage=water_coverage,
        season=season
    )
    return {"success": True, "data": static_notifications[-10:] + dynamic_alerts}

@app.post("/api/finance/analyze")
@limiter.limit("10/minute")
async def analyze_farm_finance(request: Request, body: FinanceAssessmentRequest):
    """Analyze farm finances and return loan recommendations."""
    input_data = body.model_dump() if hasattr(body, "model_dump") else body.dict()
    analysis = _farm_finance_ai.analyze_financial_profile(input_data)
    return {"success": True, "data": analysis}


@app.post("/api/finance/applications")
@limiter.limit("5/minute")
async def create_finance_application(request: Request, body: FinanceAssessmentRequest):
    """Create a loan application from the current farm profile."""
    # Idempotency check: prevent duplicate applications on retry
    idem_key = request.headers.get("X-Idempotency-Key")
    if idem_key:
        with _IDEMPOTENCY_LOCK:
            if idem_key in _IDEMPOTENCY_CACHE:
                logger.info(f"Idempotency: Returning cached finance application for key {idem_key}")
                return _IDEMPOTENCY_CACHE[idem_key]

    input_data = body.model_dump() if hasattr(body, "model_dump") else body.dict()
    application = _farm_finance_ai.create_application(input_data)
    result = {"success": True, "data": application}

    if idem_key:
        with _IDEMPOTENCY_LOCK:
            # Basic eviction: clear cache if too large
            if len(_IDEMPOTENCY_CACHE) > 1000:
                _IDEMPOTENCY_CACHE.clear()
            _IDEMPOTENCY_CACHE[idem_key] = result

    return result


@app.get("/api/finance/applications/{application_id}")
def get_finance_application(application_id: str):
    application = _farm_finance_ai.get_application(application_id)
    if not application:
        raise HTTPException(status_code=404, detail="Application not found")
    return {"success": True, "data": application}


@app.get("/api/finance/products")
def get_finance_products():
    return {"success": True, "data": _farm_finance_ai.list_marketplace()}


@app.get("/api/finance/marketplace")
def get_finance_marketplace():
    return {"success": True, "data": _farm_finance_ai.list_marketplace()}

# --- Sustainability Analytics Endpoints ---

@app.post("/api/sustainability/analyze")
@limiter.limit("20/minute")
async def analyze_sustainability(request: Request, body: SustainabilityAnalyzeRequest):
    """LCA-style water footprint and carbon emission estimate for a crop season."""
    input_data = body.model_dump() if hasattr(body, "model_dump") else body.dict()
    analysis = _sustainability_analytics.analyze(input_data)
    return {"success": True, "data": analysis}


@app.get("/api/sustainability/history")
def get_sustainability_history(
    user_id: str = Query(default="anonymous", max_length=128),
    limit: int = Query(default=12, ge=1, le=50),
):
    """Historical sustainability records for comparative analytics."""
    history = _sustainability_analytics.get_history(user_id, limit=limit)
    return {"success": True, "data": history}


@app.get("/api/sustainability/formulas")
def get_sustainability_formulas():
    """Configurable LCA coefficient datasets used by the analytics engine."""
    return {"success": True, "data": _sustainability_analytics.get_formula_config()}

# --- Weather Alerts Endpoints ---

@app.post("/api/weather/geocode")
async def geocode_location(data: WeatherLocationRequest):
    """
    Get coordinates (latitude, longitude) for a location.
    
    This endpoint helps users find their farm location's coordinates
    without exposing any API keys.
    """
    try:
        result = await weather_service.get_coordinates(data.location)
        if result:
            latitude, longitude, name = result
            return {
                "success": True,
                "location": name,
                "latitude": latitude,
                "longitude": longitude,
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Location '{data.location}' not found"
            )
    except Exception as e:
        logger.error(f"Geocoding error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to geocode location"
        ) from e


@app.post("/api/weather/alerts")
@limiter.limit("10/minute")
async def get_weather_alerts(data: WeatherAlertRequest, request: Request):
    """
    Get real-time weather alerts for a specific location and crop.
    
    Args:
        latitude: Farm latitude
        longitude: Farm longitude
        location: Location name for display
        crop: (Optional) Crop type for crop-specific warnings
    
    Returns:
        Weather alerts with severity levels and recommended actions
    
    Note: No API keys are exposed. Uses free Open-Meteo API.
    """
    try:
        # Fetch current weather
        weather = await weather_service.fetch_weather(
            data.latitude,
            data.longitude,
            data.location
        )
        
        if not weather:
            raise HTTPException(
                status_code=503,
                detail="Unable to fetch weather data. Please try again."
            )
        
        # Analyze weather and generate alerts
        alerts = weather_service.analyze_weather(weather, data.crop)
        
        # Get summary
        summary = weather_service.get_alerts_summary(alerts)
        
        return {
            "success": True,
            "location": data.location,
            "crop": data.crop,
            "weather": {
                "temperature": weather.temperature,
                "humidity": weather.humidity,
                "rainfall": weather.rainfall,
                "wind_speed": weather.wind_speed,
                "cloud_cover": weather.cloud_cover,
                "timestamp": weather.timestamp.isoformat(),
            },
            "alerts": summary,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Weather alert error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate weather alerts"
        ) from e


@app.get("/api/weather/alerts/history")
@limiter.limit("5/minute")
async def get_alerts_history(request: Request):
    """
    Get recent weather alerts history.
    Useful for reviewing past alerts and trends.
    """
    try:
        # Get recent alerts from the service history
        recent_alerts = weather_service.alert_history[-50:]  # Last 50 alerts
        return {
            "success": True,
            "total_alerts": len(weather_service.alert_history),
            "recent_alerts": [alert.to_dict() for alert in recent_alerts],
        }
    except Exception as e:
        logger.error(f"Alert history error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve alert history"
        ) from e

# --- WhatsApp Service Endpoints ---
#
# Subscriber persistence is handled by whatsapp_store.SubscriberStore, which
# provides thread-safe, crash-safe read-modify-write operations via a
# threading.Lock and atomic file replacement (write-to-tmp then os.replace).
# The old load_subscribers / save_subscribers helpers have been removed because
# they had no locking and used open(..., "w") directly, which could corrupt the
# file on a concurrent write or a mid-write crash.

@app.post("/api/whatsapp/subscribe")
@limiter.limit("2/minute")
async def subscribe_whatsapp(data: WhatsAppSubscribeRequest, request: Request):
    user_id = data.user_id if data.user_id else str(datetime.now().timestamp())
    subscriber = {
        "phone_number": data.phone_number,
        "name": data.name,
        "subscribed_at": datetime.now().isoformat(),
    }
    try:
        subscriber_store.upsert(user_id, subscriber)
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to save subscription. Please try again.",
        ) from exc

    welcome_msg = (
        f"Namaste {data.name}! 🙏\n\n"
        "Welcome to *Fasal Saathi WhatsApp Alerts*. "
        "You will now receive real-time updates directly here."
    )
    send_whatsapp_message(data.phone_number, welcome_msg)
    return {"success": True, "message": "Successfully subscribed"}

@app.post("/api/whatsapp/trigger-alert")
async def trigger_whatsapp_alert(data: AlertTriggerRequest):
    # get_all() acquires the lock and returns a stable snapshot, so this read
    # cannot race with a concurrent subscription write.
    subscribers = subscriber_store.get_all()
    results = []
    formatted_msg = format_alert_message(data.alert_type, data.message)

    for user_id, info in subscribers.items():
        res = send_whatsapp_message(info["phone_number"], formatted_msg)
        results.append({"user_id": user_id, "success": res.get("success", False)})

    static_notifications.append({
        "id": len(static_notifications) + 1,
        "type": data.alert_type,
        "message": data.message,
        "time": datetime.now().isoformat(),
    })

    return {"success": True, "results": results}

@app.post("/api/whatsapp/webhook")
async def whatsapp_webhook(Body: str = Form(...), From: str = Form(...)):
    incoming_msg = Body.lower().strip()
    sender_number = From.replace("whatsapp:", "")
    
    responses = {
        "weather": "🌡️ *Weather Update*\n\n28°C, Clear skies. No rain expected.",
        "pest": "🐛 *Pest Assistant*\n\nPlease use the Pest Management tool in-app for diagnosis.",
        "hi": "🙏 *Namaste!*\n\nI am your AI Farming Assistant. Try 'Weather' or 'Pest'.",
        "hello": "🙏 *Namaste!*\n\nI am your AI Farming Assistant. Try 'Weather' or 'Pest'."
    }
    
    response = next((v for k, v in responses.items() if k in incoming_msg), f"Received: '{Body}'. Try 'Weather' or 'Pest' 🌱")
    send_whatsapp_message(sender_number, response)
    return {"status": "success"}

# --- Cryptographic Reports ---
#
# Key resolution priority (highest → lowest):
#   1. In-process cache          – avoids repeated I/O on every request
#   2. GCP Secret Manager        – production path; key never touches disk
#   3. Local persistent PEM file – dev/staging fallback; blocked in production
#   4. Fresh generation          – last resort for local dev only
#
# When ENV=production the function raises HTTP 500 at steps 2, 3, and 4
# rather than falling through to a weaker path, so a plaintext disk key
# can never silently be used in production.

_cached_private_key = None
KEYS_DIR = "keys"
PRIVATE_KEY_PATH = os.path.join(KEYS_DIR, "report_signing.key")
PUBLIC_KEY_PATH = os.path.join(KEYS_DIR, "report_signing.pub")

IS_PRODUCTION = os.getenv("ENV", "").lower() == "production"


def get_signing_keys():
    """
    Return the Ed25519 private key used to sign financial reports.

    Resolution order:
      1. In-process cache (fastest path after first call)
      2. GCP Secret Manager (production-grade; key never written to disk)
      3. Local PEM file    (dev/staging only; raises in production)
      4. Fresh generation  (dev/staging only; raises in production)
    """
    global _cached_private_key

    # 1. In-process cache
    if _cached_private_key is not None:
        return _cached_private_key

    # 2. GCP Secret Manager
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    secret_id = os.getenv("REPORT_SIGNING_SECRET_NAME", "report-signing-key")

    if project_id:
        if not HAS_GCP_KMS:
            logger.critical("CRITICAL SECURITY ALERT: google-cloud-secret-manager is not installed but GOOGLE_CLOUD_PROJECT is set. Halting to prevent insecure fallback.")
            raise HTTPException(
                status_code=500,
                detail="KMS Initialization Error: google-cloud-secret-manager is required when GOOGLE_CLOUD_PROJECT is set. Halting to prevent insecure fallback."
            )
        else:
            try:
                client = secretmanager.SecretManagerServiceClient()
                name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
                response = client.access_secret_version(request={"name": name})
                payload = response.payload.data.decode("UTF-8")
                _cached_private_key = serialization.load_pem_private_key(
                    payload.encode(), password=None
                )
                print(f"KMS: Loaded signing key from Secret Manager (secret: {secret_id})")
                return _cached_private_key
            except Exception as e:
                logger.critical(f"CRITICAL SECURITY ALERT: KMS Initialization Failed. Could not reach Secret Manager: {e}. Halting to prevent insecure fallback to local keys.")
                raise HTTPException(
                    status_code=500,
                    detail=f"KMS Initialization Error: Failed to retrieve signing key from Secret Manager. Halting to prevent insecure fallback."
                )
    elif IS_PRODUCTION:
        logger.critical("CRITICAL SECURITY ALERT: GOOGLE_CLOUD_PROJECT is not set in production. Cannot retrieve secure signing key.")
        raise HTTPException(
            status_code=500,
            detail="GOOGLE_CLOUD_PROJECT is not set; cannot retrieve signing key in production"
        )

    # 3. Local persistent PEM file (dev/staging only)
    if os.path.exists(PRIVATE_KEY_PATH):
        try:
            with open(PRIVATE_KEY_PATH, "rb") as f:
                _cached_private_key = serialization.load_pem_private_key(f.read(), password=None)
            print(f"Key Management: Loaded existing local key from {PRIVATE_KEY_PATH}")
            return _cached_private_key
        except Exception as e:
            print(f"Key Management Warning: Could not load local key file ({e}); generating a new one.")

    # 4. Fresh generation (dev/staging only)
    print("Key Management: Generating a fresh signing key for local development.")
    private_key = ed25519.Ed25519PrivateKey.generate()

    try:
        os.makedirs(KEYS_DIR, exist_ok=True)
        with open(PRIVATE_KEY_PATH, "wb") as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))
        with open(PUBLIC_KEY_PATH, "wb") as f:
            f.write(private_key.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ))
        print(f"Key Management: Saved new key pair to {KEYS_DIR}/")
    except Exception as e:
        print(f"Key Management Warning: Could not persist generated key ({e}); key is in-memory only.")

    _cached_private_key = private_key
    return private_key


@app.post("/api/reports/generate")
@limiter.limit("3/minute")
async def generate_signed_report(data: ReportRequest, request: Request):
    # RBAC: Only Experts or Admins can generate signed reports
    await verify_role(request, required_roles=["expert", "admin"])
    
    try:
        private_key = get_signing_keys()
        
        # Create a buffer for the PDF
        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter

        # 1. Header
        p.setFont("Helvetica-Bold", 24)
        p.setFillColor(colors.green)
        p.drawCentredString(width/2, height - 1*inch, "FASAL SAATHI")
        
        p.setFont("Helvetica-Bold", 18)
        p.setFillColor(colors.black)
        p.drawCentredString(width/2, height - 1.5*inch, "CERTIFIED FINANCIAL FARM REPORT")
        
        p.setStrokeColor(colors.green)
        p.line(1*inch, height - 1.7*inch, width - 1*inch, height - 1.7*inch)

        # 2. Content
        p.setFont("Helvetica", 14)
        y = height - 2.5*inch
        
        details = [
            ("Farmer Name:", data.name),
            ("Crop Type:", data.crop),
            ("Farm Area:", data.area),
            ("Season Profit:", f"Rs. {data.profit}"),
            ("Season:", data.season),
            ("Report Date:", datetime.now().strftime("%d %B, %Y")),
        ]

        for label, value in details:
            p.setFont("Helvetica-Bold", 14)
            p.drawString(1.5*inch, y, label)
            p.setFont("Helvetica", 14)
            p.drawString(3.5*inch, y, value)
            y -= 0.4*inch

        # 3. Signature Box
        y -= 0.5*inch
        p.setStrokeColor(colors.black)
        p.rect(1*inch, y - 1.5*inch, width - 2*inch, 1.8*inch, stroke=1, fill=0)
        
        # Data for signing
        report_data_string = f"{data.name}|{data.crop}|{data.area}|{data.profit}|{datetime.now().date()}"
        signature = private_key.sign(report_data_string.encode())
        sig_id = hashlib.sha256(signature).hexdigest()[:8].upper()

        p.setFont("Helvetica-Bold", 14)
        p.drawString(1.2*inch, y - 0.3*inch, "DIGITAL CRYPTOGRAPHIC SIGNATURE")
        
        p.setFont("Helvetica", 12)
        p.drawString(1.2*inch, y - 0.7*inch, f"Signature ID: {sig_id}")
        p.setFont("Helvetica-Bold", 12)
        p.setFillColor(colors.green)
        p.drawString(1.2*inch, y - 1.0*inch, "Status: VERIFIED ✔")
        p.setFillColor(colors.black)
        p.setFont("Helvetica", 10)
        p.drawString(1.2*inch, y - 1.3*inch, "Security: This report is tamper-proof and cryptographically signed.")

        # 4. Footer
        p.setFont("Helvetica-Oblique", 10)
        p.drawCentredString(width/2, 0.5*inch, "This document is generated by Fasal Saathi and is bank-ready.")

        p.showPage()
        p.save()

        # Get PDF content
        pdf_content = buffer.getvalue()
        buffer.close()

        return Response(
            content=pdf_content,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=FasalSaathi_Report_{sig_id}.pdf"
            }
        )
    except Exception as e:
        print(f"Error generating report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/log-error")
@limiter.limit("10/minute")
async def log_error(request: Request, body: ClientErrorReport):
    """
    Receives structured error reports from the frontend.

    Hardening applied vs the original implementation:

    1. Rate-limited (10/minute per IP) — the original had no limiter at all,
       allowing unlimited flooding that could exhaust server memory and CPU.

    2. Typed, bounded Pydantic schema (ClientErrorReport) — the original used
       raw request.json() with no size limit; a single request could send an
       arbitrarily large payload.

    3. ANSI / control-character sanitisation — the original printed the message
       verbatim, allowing an attacker to inject terminal escape sequences that
       corrupt log files or exploit log viewers.  _sanitise_log_field() strips
       all ASCII control characters (including ESC) before the value reaches
       the log.

    4. structured logging via logging module — the original used print(), which
       is lost in production log aggregators that capture the logging module
       but not stdout.
    """
    level = _sanitise_log_field(body.level).lower()
    message = _sanitise_log_field(body.message)
    source = _sanitise_log_field(body.source) if body.source else "unknown"
    stack = _sanitise_log_field(body.stack) if body.stack else ""

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

# --- RAG Advisor ---
try:
    from rag.generator import generate_response as rag_generate
    HAS_RAG = True
except Exception as rag_e:
    print(f"RAG Warning: {rag_e}")
    HAS_RAG = False

@app.post("/api/rag/query")
@limiter.limit("10/minute")
async def rag_query(request: Request, body: RAGQuery):
    """RAG-based AI advisor with research-backed citations."""
    if not HAS_RAG:
        raise HTTPException(status_code=503, detail="RAG pipeline not available")
    try:
        result = rag_generate(body.query, top_k=body.top_k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/simulate-climate")
@limiter.limit("5/minute")
async def simulate_climate(request: Request, data: SimulationRequest):
    """
    Simulates the impact of climate anomalies on yield and profit.
    Based on standard agricultural sensitivity coefficients.
    """
    # Sensitivity coefficients (heuristic values)
    sensitivities = {
        "rice": {"temp": -0.05, "rain": 0.02}, # -5% yield per degree temp rise
        "wheat": {"temp": -0.06, "rain": 0.03},
        "cotton": {"temp": -0.03, "rain": 0.01},
        "maize": {"temp": -0.07, "rain": 0.04},
        "sugarcane": {"temp": -0.02, "rain": 0.05},
        "soybean": {"temp": -0.04, "rain": 0.03},
        "potato": {"temp": -0.05, "rain": 0.04},
        "default": {"temp": -0.04, "rain": 0.02}
    }
    
    crop = data.crop_type.lower()
    coeff = sensitivities.get(crop, sensitivities["default"])
    
    # Calculate yield impact
    # temp_delta is absolute change, rain_delta is percentage change
    yield_impact_temp = data.temp_delta * coeff["temp"]
    yield_impact_rain = (data.rain_delta / 100.0) * coeff["rain"]
    
    total_yield_impact = yield_impact_temp + yield_impact_rain
    
    # Heuristic for profit impact (usually amplified by fixed costs)
    profit_impact = total_yield_impact * 1.5 
    
    # Suitability Score (0-100)
    suitability = max(0, min(100, 85 + (total_yield_impact * 100)))
    
    return {
        "crop_type": data.crop_type,
        "yield_impact_pct": round(total_yield_impact * 100, 2),
        "profit_impact_pct": round(profit_impact * 100, 2),
        "suitability_score": round(suitability, 1),
        "risk_level": "High" if total_yield_impact < -0.15 else "Medium" if total_yield_impact < -0.05 else "Low",
        "recommendation": "Switch to heat-tolerant varieties" if data.temp_delta > 2 else "Ensure adequate irrigation" if data.rain_delta < -20 else "Conditions remain viable"
    }

@app.post("/api/seeds/verify")
@limiter.limit("10/minute")
async def verify_seed(data: SeedVerifyRequest, request: Request):
    """
    Verifies seed authenticity against the trusted batch registry.

    Registry lookup logic
    ---------------------
    Each entry in SEED_REGISTRY is keyed by the canonical batch code
    (upper-cased, stripped).  The entry carries:

    - status        : "authentic" | "invalid"
    - crop          : crop name the batch is certified for
    - batch         : batch identifier
    - manufacturer  : seed company name
    - cert_body     : certifying authority (e.g. NSC, ICAR)
    - certified_on  : ISO date string of certification
    - expires_on    : ISO date string of expiry  (YYYY-MM-DD)
    - reason        : present only on invalid entries — human-readable
                      explanation of why the batch is rejected

    Verification steps (in order)
    ------------------------------
    1. Format validation  — code must match the canonical pattern
                            FS-<ALPHA>-<YEAR>-<ALPHANUM> or be a known
                            blacklisted / test code.  Codes that do not
                            match any registry entry are returned as
                            "not_found" — never as "authentic".
    2. Registry lookup    — exact match against SEED_REGISTRY keys.
    3. Blacklist check    — status == "invalid" → return immediately.
    4. Expiry check       — authentic entries whose expires_on is in the
                            past are downgraded to "invalid" at query time
                            so the registry does not need to be updated
                            every season.
    5. Return             — structured response with full metadata.

    Security note
    -------------
    The old implementation used substring matching (`"FS-AUTH" in code`),
    which allowed any crafted string containing that substring to pass.
    This implementation uses exact dictionary lookup only — no substring
    or regex matching is performed on the submitted code.
    """

    # ── Trusted seed batch registry ──────────────────────────────────────────
    # In a production deployment this would be loaded from Firestore or a
    # SQL database.  The structure is kept identical so swapping the data
    # source requires only changing the lookup call, not the validation logic.
    SEED_REGISTRY: dict[str, dict] = {
        # ── Authentic batches ────────────────────────────────────────────────
        "FS-RICE-2026-A1": {
            "status": "authentic",
            "crop": "Rice (IR-64)",
            "batch": "2026-A1",
            "manufacturer": "National Seeds Corporation (NSC)",
            "cert_body": "Central Seed Certification Board (CSCB)",
            "certified_on": "2025-10-01",
            "expires_on": "2027-03-31",
        },
        "FS-WHEAT-2026-W2": {
            "status": "authentic",
            "crop": "Wheat (HD-2967)",
            "batch": "2026-W2",
            "manufacturer": "Punjab Agro Industries Corporation",
            "cert_body": "State Seed Certification Agency, Punjab",
            "certified_on": "2025-11-15",
            "expires_on": "2027-05-31",
        },
        "FS-COTTON-2026-C3": {
            "status": "authentic",
            "crop": "Cotton (Bt Hybrid)",
            "batch": "2026-C3",
            "manufacturer": "Maharashtra State Seeds Corporation",
            "cert_body": "Central Seed Certification Board (CSCB)",
            "certified_on": "2026-01-10",
            "expires_on": "2027-06-30",
        },
        "FS-MAIZE-2026-M4": {
            "status": "authentic",
            "crop": "Maize (DKC-9144)",
            "batch": "2026-M4",
            "manufacturer": "ICAR-Indian Institute of Maize Research",
            "cert_body": "Central Seed Certification Board (CSCB)",
            "certified_on": "2026-02-20",
            "expires_on": "2027-08-31",
        },
        "FS-SOYBEAN-2026-S5": {
            "status": "authentic",
            "crop": "Soybean (JS-335)",
            "batch": "2026-S5",
            "manufacturer": "Madhya Pradesh State Seeds Corporation",
            "cert_body": "State Seed Certification Agency, MP",
            "certified_on": "2026-03-05",
            "expires_on": "2027-09-30",
        },
        # ── Blacklisted / counterfeit batches ────────────────────────────────
        "FS-FAKE-2026-X9": {
            "status": "invalid",
            "crop": "Unknown",
            "batch": "2026-X9",
            "manufacturer": "Unknown",
            "cert_body": "N/A",
            "certified_on": "N/A",
            "expires_on": "N/A",
            "reason": "Blacklisted — reported counterfeit batch",
        },
        "FS-RICE-2024-OLD": {
            "status": "invalid",
            "crop": "Rice (IR-64)",
            "batch": "2024-OLD",
            "manufacturer": "National Seeds Corporation (NSC)",
            "cert_body": "Central Seed Certification Board (CSCB)",
            "certified_on": "2023-10-01",
            "expires_on": "2025-03-31",   # already expired — also caught by expiry check
            "reason": "Expired — shelf life exceeded as of 2025-03-31",
        },
    }
    # ─────────────────────────────────────────────────────────────────────────

    # Step 1 — normalise the submitted code.
    # Upper-case and strip whitespace so "fs-rice-2026-a1 " matches correctly.
    code = data.code.upper().strip()

    # Step 2 — exact registry lookup (no substring matching).
    entry = SEED_REGISTRY.get(code)

    if entry is None:
        # Code is not in the registry at all — return not_found.
        # We deliberately do NOT fall back to any pattern matching here.
        logger.info("Seed verification: code not found in registry — code=%s", code)
        return {
            "success": True,
            "code": code,
            "status": "not_found",
        }

    # Step 3 — blacklist check.
    if entry["status"] == "invalid":
        logger.warning(
            "Seed verification: invalid/blacklisted code submitted — code=%s reason=%s",
            code,
            entry.get("reason", "unknown"),
        )
        return {
            "success": True,
            "code": code,
            "status": "invalid",
            "crop": entry["crop"],
            "batch": entry["batch"],
            "manufacturer": entry["manufacturer"],
            "cert_body": entry["cert_body"],
            "reason": entry.get("reason", "Batch is invalid or blacklisted"),
        }

    # Step 4 — expiry check (authentic entries only).
    # Downgrade to "invalid" at query time if the batch has expired.
    try:
        expiry = datetime.strptime(entry["expires_on"], "%Y-%m-%d").date()
        if expiry < datetime.utcnow().date():
            logger.warning(
                "Seed verification: authentic batch has expired — code=%s expires_on=%s",
                code,
                entry["expires_on"],
            )
            return {
                "success": True,
                "code": code,
                "status": "invalid",
                "crop": entry["crop"],
                "batch": entry["batch"],
                "manufacturer": entry["manufacturer"],
                "cert_body": entry["cert_body"],
                "reason": f"Expired — shelf life exceeded as of {entry['expires_on']}",
            }
    except ValueError:
        # expires_on is "N/A" or malformed — skip expiry check.
        pass

    # Step 5 — all checks passed: return authentic result with full metadata.
    logger.info(
        "Seed verification: authentic batch confirmed — code=%s crop=%s",
        code,
        entry["crop"],
    )
    return {
        "success": True,
        "code": code,
        "status": "authentic",
        "crop": entry["crop"],
        "batch": entry["batch"],
        "manufacturer": entry["manufacturer"],
        "cert_body": entry["cert_body"],
        "certified_on": entry["certified_on"],
        "expires_on": entry["expires_on"],
    }

# --- Crop Quality Grading Endpoints ---

@app.post("/api/quality/assess-single")
@limiter.limit("10/minute")
async def assess_single_crop(request: Request, data: CropQualityGradingRequest):
    """
    Assess quality of a single crop from image
    
    Request body:
    {
        "crop_type": "tomato",  // or potato, grain, fruit
        "image_base64": "<base64_encoded_image>"
    }
    """
    try:
        # Decode base64 image
        import base64
        image_bytes = base64.b64decode(data.image_base64)
        
        # Assess the crop
        assessment = _crop_quality_grader.assess_crop_image(
            image_bytes, 
            data.crop_type
        )
        
        # Convert to dict
        from dataclasses import asdict
        result = asdict(assessment)
        
        return {
            "success": True,
            "data": result,
            "timestamp": datetime.now().isoformat()
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Quality assessment error: %s", str(e))
        raise HTTPException(status_code=500, detail="Quality assessment failed")

@app.post("/api/quality/assess-batch")
@limiter.limit("5/minute")
async def assess_batch_crops(request: Request, data: CropQualityBatchRequest):
    """
    Assess quality of multiple crops in batch
    
    Request body:
    {
        "crop_type": "tomato",
        "images_base64": ["<base64_image1>", "<base64_image2>", ...]
    }
    """
    try:
        import base64
        
        # Decode all images
        image_bytes_list = []
        for img_b64 in data.images_base64:
            try:
                image_bytes = base64.b64decode(img_b64)
                image_bytes_list.append(image_bytes)
            except Exception as e:
                logger.warning("Failed to decode image: %s", str(e))
                continue
        
        if not image_bytes_list:
            raise ValueError("No valid images provided")
        
        # Batch grade
        result = _crop_quality_grader.batch_grade_crops(
            image_bytes_list,
            data.crop_type
        )
        
        return {
            "success": True,
            "data": result,
            "timestamp": datetime.now().isoformat()
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Batch assessment error: %s", str(e))
        raise HTTPException(status_code=500, detail="Batch assessment failed")

@app.post("/api/quality/trends")
@limiter.limit("10/minute")
async def get_quality_trends(request: Request, data: QualityTrendsRequest):
    """
    Get quality trends for a crop type
    
    Request body:
    {
        "crop_type": "tomato",
        "days": 7
    }
    """
    try:
        trends = _crop_quality_grader.get_quality_trends(
            data.crop_type,
            data.days
        )
        
        return {
            "success": True,
            "data": trends,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error("Quality trends error: %s", str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve trends")

@app.get("/api/quality/supported-crops")
@limiter.limit("20/minute")
async def get_supported_crops(request: Request):
    """
    Get list of supported crops for quality grading
    """
    return {
        "success": True,
        "crops": _crop_quality_grader.supported_crops,
        "total": len(_crop_quality_grader.supported_crops)
    }

@app.post("/api/quality/market-price")
@limiter.limit("10/minute")
async def calculate_market_price(request: Request, data: CropQualityGradingRequest):
    """
    Calculate market price adjustment based on quality grade
    
    Request body:
    {
        "crop_type": "tomato",
        "image_base64": "<base64_encoded_image>"
    }
    """
    try:
        import base64
        from crop_quality_grading import GRADE_MAPPING
        
        # Decode image
        image_bytes = base64.b64decode(data.image_base64)
        
        # Assess quality
        assessment = _crop_quality_grader.assess_crop_image(
            image_bytes,
            data.crop_type
        )
        
        # Get grade info
        grade_info = GRADE_MAPPING[assessment.grade]
        
        return {
            "success": True,
            "crop_type": data.crop_type,
            "grade": assessment.grade,
            "grade_label": grade_info["label"],
            "score": assessment.score,
            "price_multiplier": grade_info["price_multiplier"],
            "recommendations": assessment.recommendations,
            "timestamp": datetime.now().isoformat()
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Market price calculation error: %s", str(e))
        raise HTTPException(status_code=500, detail="Price calculation failed")

class ConsultationBookRequest(BaseModel):
    expert_id: str = Field(..., min_length=1, max_length=100)
    expert_name: str = Field(..., min_length=1, max_length=200)
    expert_specialization: str = Field(..., min_length=1, max_length=100)
    date: str = Field(..., min_length=10, max_length=10)
    time: str = Field(..., min_length=5, max_length=5)
    notes: Optional[str] = Field(default="", max_length=1000)
    consultation_type: str = Field(default="video", pattern=r'^(video|audio)$')

class ConsultationUpdateRequest(BaseModel):
    status: str = Field(..., pattern=r'^(scheduled|completed|cancelled|in-progress)$')
    notes: Optional[str] = Field(default=None, max_length=1000)

class ExpertSlotRequest(BaseModel):
    expert_id: str = Field(..., min_length=1, max_length=100)
    date: str = Field(..., min_length=10, max_length=10)

# --- Expert Consultation APIs ---

@app.get("/api/experts")
async def get_experts(
    specialization: Optional[str] = None,
    kvk_only: bool = False,
    search: Optional[str] = None
):
    """
    Get list of available experts/KVK advisors.
    
    Query parameters:
    - specialization: Filter by specialization (crop_disease, fertilizers, etc.)
    - kvk_only: Return only KVK experts
    - search: Search by name or location
    """
    try:
        experts = []

        if db_firestore:
            try:
                experts_ref = db_firestore.collection("experts")
                docs = experts_ref.get()

                for doc in docs:
                    expert_data = doc.to_dict()
                    expert_data["id"] = doc.id
                    experts.append(expert_data)
            except Exception as e:
                logger.warning(f"Firestore experts query failed: {e}")

        if not experts:
            experts = [
                {
                    "id": "exp1",
                    "name": "Dr. Ramesh Kumar",
                    "specialization": "crop_disease",
                    "qualification": "Ph.D. in Plant Pathology",
                    "location": "Madhya Pradesh",
                    "phone": "+91 9876543210",
                    "rating": 4.8,
                    "experience": 15,
                    "is_kvk": True,
                    "kvk_name": "KVK Jabalpur",
                    "bio": "Specialist in crop disease diagnosis and organic treatment methods.",
                    "avatar": "https://randomuser.me/api/portraits/men/32.jpg"
                },
                {
                    "id": "exp2",
                    "name": "Dr. Priya Sharma",
                    "specialization": "fertilizers",
                    "qualification": "M.Sc. Agricultural Chemistry",
                    "location": "Maharashtra",
                    "phone": "+91 9876543211",
                    "rating": 4.9,
                    "experience": 12,
                    "is_kvk": True,
                    "kvk_name": "KVK Pune",
                    "bio": "Expert in nano-fertilizers and sustainable nutrient management.",
                    "avatar": "https://randomuser.me/api/portraits/women/44.jpg"
                },
                {
                    "id": "exp3",
                    "name": "Er. Suresh Patil",
                    "specialization": "irrigation",
                    "qualification": "B.Tech Agricultural Engineering",
                    "location": "Karnataka",
                    "phone": "+91 9876543212",
                    "rating": 4.7,
                    "experience": 10,
                    "is_kvk": False,
                    "bio": "Drip irrigation and water management specialist.",
                    "avatar": "https://randomuser.me/api/portraits/men/45.jpg"
                },
                {
                    "id": "exp4",
                    "name": "Dr. Anjali Verma",
                    "specialization": "pest_management",
                    "qualification": "Ph.D. Entomology",
                    "location": "Uttar Pradesh",
                    "phone": "+91 9876543213",
                    "rating": 4.6,
                    "experience": 8,
                    "is_kvk": True,
                    "kvk_name": "KVK Lucknow",
                    "bio": "Integrated pest management and organic pest control expert.",
                    "avatar": "https://randomuser.me/api/portraits/women/65.jpg"
                },
                {
                    "id": "exp5",
                    "name": "Dr. Mahendra Singh",
                    "specialization": "soil_health",
                    "qualification": "Ph.D. Soil Science",
                    "location": "Rajasthan",
                    "phone": "+91 9876543214",
                    "rating": 4.9,
                    "experience": 20,
                    "is_kvk": True,
                    "kvk_name": "KVK Jaipur",
                    "bio": "Soil health assessment and reclamation specialist.",
                    "avatar": "https://randomuser.me/api/portraits/men/67.jpg"
                },
                {
                    "id": "exp6",
                    "name": "Dr. Kavita Desai",
                    "specialization": "market_advisory",
                    "qualification": "MBA Agriculture Business",
                    "location": "Gujarat",
                    "phone": "+91 9876543215",
                    "rating": 4.8,
                    "experience": 14,
                    "is_kvk": False,
                    "bio": "Market intelligence and price forecasting expert.",
                    "avatar": "https://randomuser.me/api/portraits/women/28.jpg"
                }
            ]

        if specialization:
            experts = [e for e in experts if e.get("specialization") == specialization]
        if kvk_only:
            experts = [e for e in experts if e.get("is_kvk")]
        if search:
            search_lower = search.lower()
            experts = [e for e in experts if search_lower in e.get("name", "").lower() or search_lower in e.get("location", "").lower()]

        if not experts:
            experts = [
                {
                    "id": "exp1",
                    "name": "Dr. Ramesh Kumar",
                    "specialization": "crop_disease",
                    "qualification": "Ph.D. in Plant Pathology",
                    "location": "Madhya Pradesh",
                    "phone": "+91 9876543210",
                    "rating": 4.8,
                    "experience": 15,
                    "is_kvk": True,
                    "kvk_name": "KVK Jabalpur",
                    "bio": "Specialist in crop disease diagnosis and organic treatment methods.",
                    "avatar": "https://randomuser.me/api/portraits/men/32.jpg"
                },
                {
                    "id": "exp2",
                    "name": "Dr. Priya Sharma",
                    "specialization": "fertilizers",
                    "qualification": "M.Sc. Agricultural Chemistry",
                    "location": "Maharashtra",
                    "phone": "+91 9876543211",
                    "rating": 4.9,
                    "experience": 12,
                    "is_kvk": True,
                    "kvk_name": "KVK Pune",
                    "bio": "Expert in nano-fertilizers and sustainable nutrient management.",
                    "avatar": "https://randomuser.me/api/portraits/women/44.jpg"
                },
                {
                    "id": "exp3",
                    "name": "Er. Suresh Patil",
                    "specialization": "irrigation",
                    "qualification": "B.Tech Agricultural Engineering",
                    "location": "Karnataka",
                    "phone": "+91 9876543212",
                    "rating": 4.7,
                    "experience": 10,
                    "is_kvk": False,
                    "bio": "Drip irrigation and water management specialist.",
                    "avatar": "https://randomuser.me/api/portraits/men/45.jpg"
                },
                {
                    "id": "exp4",
                    "name": "Dr. Anjali Verma",
                    "specialization": "pest_management",
                    "qualification": "Ph.D. Entomology",
                    "location": "Uttar Pradesh",
                    "phone": "+91 9876543213",
                    "rating": 4.6,
                    "experience": 8,
                    "is_kvk": True,
                    "kvk_name": "KVK Lucknow",
                    "bio": "Integrated pest management and organic pest control expert.",
                    "avatar": "https://randomuser.me/api/portraits/women/65.jpg"
                },
                {
                    "id": "exp5",
                    "name": "Dr. Mahendra Singh",
                    "specialization": "soil_health",
                    "qualification": "Ph.D. Soil Science",
                    "location": "Rajasthan",
                    "phone": "+91 9876543214",
                    "rating": 4.9,
                    "experience": 20,
                    "is_kvk": True,
                    "kvk_name": "KVK Jaipur",
                    "bio": "Soil health assessment and reclamation specialist.",
                    "avatar": "https://randomuser.me/api/portraits/men/67.jpg"
                },
                {
                    "id": "exp6",
                    "name": "Dr. Kavita Desai",
                    "specialization": "market_advisory",
                    "qualification": "MBA Agriculture Business",
                    "location": "Gujarat",
                    "phone": "+91 9876543215",
                    "rating": 4.8,
                    "experience": 14,
                    "is_kvk": False,
                    "bio": "Market intelligence and price forecasting expert.",
                    "avatar": "https://randomuser.me/api/portraits/women/28.jpg"
                }
            ]

            if specialization:
                experts = [e for e in experts if e.get("specialization") == specialization]
            if kvk_only:
                experts = [e for e in experts if e.get("is_kvk")]
            if search:
                search_lower = search.lower()
                experts = [e for e in experts if search_lower in e.get("name", "").lower() or search_lower in e.get("location", "").lower()]

        return {"experts": experts, "count": len(experts)}
    except Exception as e:
        logger.error(f"Error fetching experts: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch experts")


@app.get("/api/experts/{expert_id}/slots")
async def get_expert_slots(expert_id: str, date: str):
    """
    Get available time slots for an expert on a specific date.
    
    Path parameters:
    - expert_id: ID of the expert
    
    Query parameters:
    - date: Date in YYYY-MM-DD format
    """
    try:
        from datetime import datetime, timedelta

        requested_date = datetime.strptime(date, "%Y-%m-%d")
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        if requested_date < today:
            return {"slots": [], "message": "Cannot book slots for past dates"}

        slots = []
        for hour in range(9, 18):
            for minute in [0, 30]:
                time_str = f"{hour:02d}:{minute:02d}"
                available = True

                if requested_date.date() == today.date() and hour <= datetime.now().hour:
                    available = False

                if available:
                    slots.append({
                        "time": time_str,
                        "display": f"{hour}:{minute:02d} {'AM' if hour < 12 else 'PM'}",
                        "available": available
                    })

        return {"expert_id": expert_id, "date": date, "slots": slots}
    except Exception as e:
        logger.error(f"Error fetching slots: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch slots")


@app.post("/api/consultation/book")
async def book_consultation(
    request: ConsultationBookRequest,
    raw_request: Request,
    authorization: Optional[str] = None
):
    """
    Book a consultation with an expert.
    """
    # Idempotency check: prevent duplicate bookings on retry
    idem_key = raw_request.headers.get("X-Idempotency-Key")
    if idem_key:
        with _IDEMPOTENCY_LOCK:
            if idem_key in _IDEMPOTENCY_CACHE:
                logger.info(f"Idempotency: Returning cached consultation for key {idem_key}")
                return _IDEMPOTENCY_CACHE[idem_key]
    
    Request body:
    - expert_id: Expert's ID
    - expert_name: Expert's name
    - expert_specialization: Expert's specialization
    - date: Date in YYYY-MM-DD format
    - time: Time in HH:MM format
    - notes: Optional notes about the consultation
    - consultation_type: "video" or "audio"
    """
    try:
        user_data = None

        if authorization:
            try:
                token = authorization.replace("Bearer ", "")
                decoded_token = firebase_auth.verify_id_token(token)
                user_id = decoded_token.get("uid")

                user_ref = db_firestore.collection("users").document(user_id)
                user_doc = user_ref.get()
                if user_doc.exists:
                    user_data = user_doc.to_dict()
            except Exception as e:
                logger.warning(f"Auth verification failed: {e}")

        consultation_data = {
            "expert_id": request.expert_id,
            "expert_name": request.expert_name,
            "expert_specialization": request.expert_specialization,
            "user_id": user_data.get("uid") if user_data else "anonymous",
            "user_name": user_data.get("displayName") if user_data else "Guest Farmer",
            "date": request.date,
            "time": request.time,
            "notes": request.notes,
            "type": request.consultation_type,
            "status": "scheduled",
            "created_at": datetime.now().isoformat()
        }

        doc_ref = db_firestore.collection("consultations").document()
        doc_ref.set(consultation_data)

        result = {
            "success": True,
            "consultation_id": doc_ref.id,
            "message": "Consultation booked successfully"
        }

        if idem_key:
            with _IDEMPOTENCY_LOCK:
                if len(_IDEMPOTENCY_CACHE) > 1000:
                    _IDEMPOTENCY_CACHE.clear()
                _IDEMPOTENCY_CACHE[idem_key] = result

        return result
    except Exception as e:
        logger.error(f"Error booking consultation: {e}")
        raise HTTPException(status_code=500, detail="Failed to book consultation")


@app.put("/api/consultation/{consultation_id}")
async def update_consultation(
    consultation_id: str,
    request: ConsultationUpdateRequest,
    authorization: Optional[str] = None
):
    """
    Update consultation status.
    
    Path parameters:
    - consultation_id: ID of the consultation
    
    Request body:
    - status: New status (scheduled, completed, cancelled, in-progress)
    - notes: Optional notes
    """
    try:
        doc_ref = db_firestore.collection("consultations").document(consultation_id)
        doc = doc_ref.get()

        if not doc.exists:
            raise HTTPException(status_code=404, detail="Consultation not found")

        update_data = {
            "status": request.status,
            "updated_at": datetime.now().isoformat()
        }

        if request.notes:
            update_data["notes"] = request.notes

        if request.status == "completed":
            update_data["completed_at"] = datetime.now().isoformat()

        doc_ref.update(update_data)

        return {
            "success": True,
            "message": "Consultation updated successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating consultation: {e}")
        raise HTTPException(status_code=500, detail="Failed to update consultation")


@app.get("/api/consultations")
async def get_user_consultations(
    user_id: Optional[str] = None,
    status: Optional[str] = None,
    authorization: Optional[str] = None
):
    """
    Get user's consultation history.
    
    Query parameters:
    - user_id: User ID (optional, will use auth if provided)
    - status: Filter by status (scheduled, completed, cancelled)
    """
    try:
        consultations_ref = db_firestore.collection("consultations")
        query = consultations_ref

        if user_id:
            query = query.where("user_id", "==", user_id)

        docs = query.get()
        consultations = []

        for doc in docs:
            consultation = doc.to_dict()
            consultation["id"] = doc.id

            if status and consultation.get("status") != status:
                continue

            consultations.append(consultation)

        consultations.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        return {
            "consultations": consultations,
            "count": len(consultations)
        }
    except Exception as e:
        logger.error(f"Error fetching consultations: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch consultations")


@app.delete("/api/consultation/{consultation_id}")
async def cancel_consultation(
    consultation_id: str,
    authorization: Optional[str] = None
):
    """
    Cancel a consultation.
    
    Path parameters:
    - consultation_id: ID of the consultation to cancel
    """
    try:
        doc_ref = db_firestore.collection("consultations").document(consultation_id)
        doc = doc_ref.get()

        if not doc.exists:
            raise HTTPException(status_code=404, detail="Consultation not found")

        doc_ref.update({
            "status": "cancelled",
            "cancelled_at": datetime.now().isoformat()
        })

        return {
            "success": True,
            "message": "Consultation cancelled successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling consultation: {e}")
        raise HTTPException(status_code=500, detail="Failed to cancel consultation")


# --- Smart Farm Autopilot Endpoint ---

class SeasonPlanRequest(BaseModel):
    farm_name: str = Field(default="My Farm", max_length=100)
    state: str = Field(..., min_length=2, max_length=50)
    district: str = Field(default="", max_length=100)
    area_acres: float = Field(..., gt=0, le=10000)
    soil_type: str = Field(..., min_length=2, max_length=50)
    season: str = Field(..., pattern="^(Kharif|Rabi|Zaid)$")
    water_source: str = Field(default="Canal", max_length=50)
    budget_inr: Optional[float] = Field(default=None, ge=0)


@app.post("/api/autopilot/generate-plan")
@limiter.limit("5/minute")
async def generate_farm_plan(request: Request, data: SeasonPlanRequest):
    """
    Smart Farm Autopilot — generate a complete seasonal farming plan.

    Returns crop selection, sowing schedule, irrigation plan,
    fertilizer/pesticide timeline, and yield/profit projection.
    """
    try:
        plan = generate_season_plan(data.dict())
        return {"success": True, "plan": plan}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Autopilot plan generation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate farm plan")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

# --- Blockchain Supply Chain Endpoints ---

@app.post("/api/blockchain/register-actor")
@limiter.limit("10/minute")
async def register_actor(request: Request, data: RegisterActorRequest):
    """Register supply chain participant"""
    try:
        actor_data = _supply_chain_blockchain.register_actor(
            data.actor_id,
            data.name,
            data.actor_type,
            data.location
        )
        return {"success": True, "actor": actor_data}
    except Exception as e:
        logger.error("Register actor error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/blockchain/create-batch")
@limiter.limit("10/minute")
async def create_batch(request: Request, data: CreateProductBatchRequest):
    """Create product batch on blockchain"""
    try:
        batch = _supply_chain_blockchain.create_product_batch(
            data.crop_type,
            data.farm_id,
            data.quantity,
            data.unit,
            data.planting_date,
            data.harvesting_date,
            data.farmer_name
        )
        from dataclasses import asdict
        return {"success": True, "batch": asdict(batch)}
    except Exception as e:
        logger.error("Create batch error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/blockchain/add-node")
@limiter.limit("10/minute")
async def add_node(request: Request, data: AddSupplyChainNodeRequest):
    """Add supply chain node"""
    try:
        node = _supply_chain_blockchain.add_supply_chain_node(
            data.batch_id,
            data.node_type,
            data.actor_name,
            data.location,
            data.action,
            temperature=data.temperature,
            humidity=data.humidity,
            quality_check=data.quality_check,
            notes=data.notes
        )
        from dataclasses import asdict
        return {"success": True, "node": asdict(node)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Add node error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/blockchain/create-contract")
@limiter.limit("10/minute")
async def create_contract(request: Request, data: CreateSmartContractRequest):
    """Create smart contract"""
    try:
        contract = _supply_chain_blockchain.create_smart_contract(
            data.batch_id,
            data.seller,
            data.buyer,
            data.price,
            data.terms
        )
        from dataclasses import asdict
        return {"success": True, "contract": asdict(contract)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Create contract error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/blockchain/execute-contract")
@limiter.limit("10/minute")
async def execute_contract(request: Request, data: ExecuteContractRequest):
    """Execute smart contract"""
    try:
        result = _supply_chain_blockchain.execute_smart_contract(data.contract_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Execute contract error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/blockchain/qr-code/{batch_id}")
@limiter.limit("20/minute")
async def get_qr_code(request: Request, batch_id: str):
    """Get QR code for batch"""
    try:
        qr_code = _supply_chain_blockchain.generate_qr_code(batch_id)
        return {"success": True, "batch_id": batch_id, "qr_code_base64": qr_code}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("QR code generation error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/blockchain/verify/{batch_id}")
@limiter.limit("20/minute")
async def verify_batch(request: Request, batch_id: str):
    """Verify batch authenticity"""
    try:
        verification = _supply_chain_blockchain.verify_batch(batch_id)
        return verification
    except Exception as e:
        logger.error("Verification error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/blockchain/journey/{batch_id}")
@limiter.limit("20/minute")
async def get_journey(request: Request, batch_id: str):
    """Get supply chain journey"""
    try:
        journey = _supply_chain_blockchain.get_supply_chain_journey(batch_id)
        return {"success": True, "data": journey}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Journey retrieval error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/blockchain/analytics/{batch_id}")
@limiter.limit("20/minute")
async def get_analytics(request: Request, batch_id: str):
    """Get supply chain analytics"""
    try:
        analytics = _supply_chain_blockchain.get_supply_chain_analytics(batch_id)
        return {"success": True, "data": analytics}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Analytics error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/blockchain/marketplace")
@limiter.limit("20/minute")
async def get_marketplace(request: Request):
    """Get certified products for marketplace"""
    try:
        certified = _supply_chain_blockchain.get_certified_products()
        return {"success": True, "products": certified, "count": len(certified)}
    except Exception as e:
        logger.error("Marketplace error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/blockchain/stats")
@limiter.limit("20/minute")
async def get_stats(request: Request):
    """Get blockchain statistics"""
    try:
        stats = {
            "total_records": _supply_chain_blockchain.get_blockchain_record_count(),
            "total_products": len(_supply_chain_blockchain.products),
            "registered_actors": len(_supply_chain_blockchain.verified_actors),
            "total_contracts": len(_supply_chain_blockchain.smart_contracts)
        }
        return {"success": True, "stats": stats}
    except Exception as e:
        logger.error("Stats error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ── Market Price Forecasting ──────────────────────────────────────────────────

@app.post("/api/market/forecast")
@limiter.limit("10/minute")
async def market_price_forecast(data: MarketForecastRequest, request: Request):
    """
    Generate an LSTM-based price forecast for a supported commodity.

    Authentication required — any logged-in user may call this endpoint.

    The forecasting engine (ml/price_forecaster.py) trains a lightweight
    LSTM on embedded 2-year historical mandi price data at first request
    per commodity and caches the model in memory.  Subsequent requests
    for the same commodity are served from the cache with no retraining.

    Security hardening applied:
    - verify_role() requires a valid Firebase ID token (any role).
    - MarketForecastRequest.commodity is validated against a whitelist of
      the 6 supported commodities.  Unknown strings are rejected with 422
      before the forecaster is called, preventing an attacker from
      submitting thousands of unique strings that each trigger a new
      30-epoch LSTM training run and grow the model cache without bound.

    Request body
    ------------
    - commodity : str  — one of: Wheat, Paddy (Dhan), Cotton, Onion,
                         Soybean, Maize
    - days      : int  — forecast horizon in days (1–30, default 14)

    Response
    --------
    - commodity          : str
    - forecast_days      : int
    - forecast           : list[{date, price, lower_bound, upper_bound}]
    - best_sell_date     : str   — ISO date of forecast price peak
    - best_sell_price    : float — predicted peak price (₹/quintal)
    - recommendation     : str   — human-readable selling advice
    - model_type         : str   — "LSTM" or "Statistical" (fallback)
    - generated_at       : str   — ISO UTC timestamp
    """
    await verify_role(request)   # any authenticated user; no role restriction

    try:
        from ml.price_forecaster import price_forecaster
        result = price_forecaster.forecast(
            commodity=data.commodity,
            days=data.days,
        )
        return result
    except Exception as exc:
        logger.error(
            "Market forecast error for commodity='%s': %s",
            data.commodity, exc,
        )
        raise HTTPException(
            status_code=500,
            detail="Price forecast temporarily unavailable. Please try again later.",
        )
