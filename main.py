"""
Main API entrypoint — modular router registration

This refactored file preserves API routes while delegating domain routes to
`backend.routers.*` modules. It intentionally keeps initialization lightweight
and non-destructive so upstream changes can be rebased safely.
"""
import os
import re
import logging
import collections
import threading
from datetime import datetime
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
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

    @validator("query")
    def sanitize_and_normalize_query(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("Query must be a non-empty string.")

        # Strip script tags entirely to prevent XSS / script injection
        v = re.sub(r'<script.*?>.*?</script>', '', v, flags=re.IGNORECASE | re.DOTALL)
        v = re.sub(r'</?script.*?>', '', v, flags=re.IGNORECASE)

        # Strip event handler attributes (onclick=, onload=, etc.)
        v = re.sub(r'on\w+\s*=', '', v, flags=re.IGNORECASE)

        # Strip dangerous URI schemes
        v = re.sub(r'javascript:', '', v, flags=re.IGNORECASE)
        v = re.sub(r'data:', '', v, flags=re.IGNORECASE)
        v = re.sub(r'vbscript:', '', v, flags=re.IGNORECASE)

        # Strip all other HTML tags entirely to prevent HTML injection in prompts or UI
        v = re.sub(r'<[^>]*>', '', v)

        # Neutralize markdown links [text](url) -> text (url)
        v = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1 (\2)', v)

        # Neutralize markdown styling syntax (bold, italic, code, headers)
        v = re.sub(r'[*_~`#]', '', v)

        # Normalize whitespace
        v = v.strip()
        v = re.sub(r'\s+', ' ', v)

        # Prompt injection mitigation: reject known instruction-override phrases
        forbidden_patterns = [
            r"ignore\s+(?:all\s+)?previous\s+instructions",
            r"ignore\s+(?:the\s+)?system\s+prompt",
            r"override\s+system\s+constraints",
            r"developer\s+mode",
            r"bypass\s+safety\s+filter",
            r"disregard\s+(?:all\s+)?prior\s+instructions",
            r"act\s+as\s+(?:a\s+)?(?:different|unrestricted|unfiltered)\s+(?:ai|model|assistant)",
            r"pretend\s+(?:you\s+are|to\s+be)\s+(?:a\s+)?(?:different|unrestricted)",
            r"jailbreak",
            r"prompt\s+injection",
        ]
        v_lower = v.lower()
        for pattern in forbidden_patterns:
            if re.search(pattern, v_lower):
                raise ValueError("Query contains disallowed phrases or prompt injection attempts.")

        # Re-enforce minimum length after sanitization has stripped content
        if len(v) < 3:
            raise ValueError("Query must be at least 3 characters long after sanitization.")

        return v

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

# Observability: tracing + metrics
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, ConsoleSpanExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
import firebase_admin
from firebase_admin import credentials, auth, firestore

# Persistence Layer
from persistence.repositories import (
    FinanceApplicationRepository,
    NotificationRepository,
    SupplyChainRepository,
)

# RBAC (Role-Based Access Control)
from rbac import (
    RBACManager,
    RBACMiddleware,
    Permission,
    require_permission,
    print_rbac_matrix,
)

# Persistence Layer
from persistence.repositories import (
    FinanceApplicationRepository,
    NotificationRepository,
    SupplyChainRepository,
)

# RBAC (Role-Based Access Control)
from rbac import (
    RBACManager,
    RBACMiddleware,
    Permission,
    require_permission,
    print_rbac_matrix,
)

# Persistence Layer
from persistence.repositories import (
    FinanceApplicationRepository,
    NotificationRepository,
    SupplyChainRepository,
)

# RBAC (Role-Based Access Control)
from rbac import (
    RBACManager,
    RBACMiddleware,
    Permission,
    require_permission,
    print_rbac_matrix,
)

# ML Governance (Drift Detection, Shadow Evaluation, Rollback Safety)
from ml.governance import (
    DriftDetector,
    ShadowEvaluator,
    ModelVersionManager,
)

# Other internal modules
from alert_rules import generate_alerts
from whatsapp_service import send_whatsapp_message, format_alert_message
from whatsapp_store import subscriber_store
from crop_quality_grading import CropQualityGrader
from blockchain_supply_chain import SupplyChainBlockchain
from farm_finance_ai import FarmFinanceAI

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import inch
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

# KMS Support
try:
    from feature_flags.routes import router as flags_router
except Exception:
    flags_router = None

# Import modular routers
from backend.routers import ml, governance, alerts, finance, quality, blockchain, reports, knowledge, community, voice_assistant

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global Firebase and Firestore references
db_firestore = None

def get_firestore_client():
    """
    Safely retrieves the Firestore client. If Firebase is already initialized
    by other modules but db_firestore is None, it attempts dynamic retrieval.
    """
    global db_firestore
    if db_firestore is not None:
        return db_firestore
    
    try:
        # If not initialized, try standard initialization
        if not firebase_admin._apps:
            cred = None
            if os.path.exists("serviceAccountKey.json"):
                cred = credentials.Certificate("serviceAccountKey.json")
            elif os.path.exists("firebase-credentials.json"):
                cred = credentials.Certificate("firebase-credentials.json")
            else:
                cred_json = os.getenv("FIREBASE_CREDENTIALS_JSON")
                if cred_json:
                    import json
                    cred_dict = json.loads(cred_json)
                    cred = credentials.Certificate(cred_dict)
            
            try:
                if cred:
                    firebase_admin.initialize_app(cred)
                else:
                    firebase_admin.initialize_app()
            except ValueError:
                # App already exists
                pass

        if firebase_admin._apps:
            db_firestore = firestore.client()
            logger.info("Firestore client initialized/retrieved successfully")
    except Exception as e:
        logger.error(f"Failed to retrieve Firestore client: {e}")
        db_firestore = None
    return db_firestore

def validate_firestore_ready():
    """
    Validates that the Firestore database is fully initialized and ready.
    If not, raises an HTTP 503 Service Unavailable exception to prevent crashes.
    """
    client = get_firestore_client()
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="Database service temporarily unavailable. Firestore was not initialized."
        )
    return client

def sanitise_log_field(value: str) -> str:
    if not isinstance(value, str):
        return str(value)
    sanitised = ''.join(c if ord(c) >= 32 or c in '\n\t' else f'\\x{ord(c):02x}' for c in value)
    return sanitised[:1000]

async def verify_role(request: Request, required_roles: list = None, require_all: bool = False):
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing auth token")
    token = auth_header.split(" ")[1]
    try:
        decoded = auth.verify_id_token(token)
        uid = decoded["uid"]
        db = validate_firestore_ready()
        user_doc = db.collection("users").document(uid).get()
        user_roles = user_doc.get("roles", []) if user_doc.exists else []
        if required_roles:
            if require_all:
                has_access = all(role in user_roles for role in required_roles)
            else:
                has_access = any(role in user_roles for role in required_roles)
            if not has_access:
                raise HTTPException(status_code=403, detail="Insufficient permissions")
        return {"uid": uid, "roles": user_roles}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected authorization verification failure: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during authorization")


class NotificationStore:
    def __init__(self, max_size=1000, ttl_hours=24):
        self.notifications = collections.deque(maxlen=max_size)
        self.ttl_hours = ttl_hours
        self.lock = threading.Lock()
        self.id_counter = 0

    def append(self, alert_type: str, message: str):
        with self.lock:
            self.id_counter += 1
            self.notifications.append({
                "id": self.id_counter,
                "alert_type": alert_type,
                "message": message,
                "timestamp": datetime.now().isoformat(),
                "ttl": datetime.now().isoformat()
            })

    def get_recent(self, count=10):
        with self.lock:
            return list(reversed([n for n in list(self.notifications)[-count:]]))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_firestore
    try:
        logger.info("Starting up: initializing Firebase and services...")
        get_firestore_client()
        if db_firestore:
            logger.info("Firebase/Firestore startup initialization verified successfully")
        else:
            logger.warning("Firebase/Firestore failed to initialize during startup diagnostics. Downstream database endpoints will return 503.")
    except Exception as e:
        logger.error(f"Firebase init failed during startup: {e}")
    yield
    logger.info("App shutting down")


app = FastAPI(title="Fasal Saathi Backend", version="2.0", lifespan=lifespan)

# ----- OpenTelemetry Tracing Setup -----
try:
    service_name = os.environ.get("OTEL_SERVICE_NAME", "fasal-saathi-backend")
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)
    otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if otlp_endpoint:
        otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
    else:
        # fallback to console exporter for local dev / CI
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    # instrument FastAPI
    FastAPIInstrumentor().instrument_app(app)
    logger.info("OpenTelemetry tracing initialized")
except Exception as e:
    logger.warning(f"OpenTelemetry init skipped: {e}")

# ----- Prometheus Metrics -----
try:
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")
    logger.info("Prometheus instrumentation enabled at /metrics")
except Exception as e:
    logger.warning(f"Prometheus instrumentation skipped: {e}")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add RBAC middleware for access logging
app.add_middleware(RBACMiddleware)

# Log RBAC matrix on startup
logger.info(print_rbac_matrix())

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
    name: str = Field(..., max_length=100, description="Full name of the farmer")
    crop: str = Field(..., max_length=50, description="Primary crop type")
    area: str = Field(..., max_length=50, description="Total farm area")
    profit: str = Field(..., max_length=50, description="Estimated season profit")
    season: str = Field(..., max_length=50, description="Farming season")

    @validator("name", "crop", "area", "profit", "season", pre=True)
    def sanitize_and_validate_input(cls, v):
        # Enforce robust input validation standards: ensure input is clean,
        # printable text and strip leading/trailing whitespace.
        # Primary security against injection is handled via canonical structured JSON
        # serialization during report signing, completely eliminating delimiter-based risks.
        if isinstance(v, str):
            v = v.strip()
            if "|" in v:
                raise ValueError("Field value must not contain the '|' character.")
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

class GeminiImageRequest(BaseModel):
    """
    Request body for the server-side Gemini image analysis proxy.
    The frontend sends the base64-encoded image and a prompt; the backend
    forwards the request to Google using the server-side GEMINI_API_KEY so
    the key is never exposed in the compiled JavaScript bundle.
    """
    image_base64: str = Field(..., min_length=10, description="Base64-encoded image data")
    mime_type: str = Field(..., pattern=r"^image/(jpeg|png|gif|webp)$", description="MIME type of the image")
    prompt: str = Field(..., min_length=10, max_length=2000, description="Analysis prompt")

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

# ML Governance Request Models
class StartShadowEvaluationRequest(BaseModel):
    production_model: str = Field(..., min_length=1, max_length=50)
    candidate_model: str = Field(..., min_length=1, max_length=50)

class RecordPredictionRequest(BaseModel):
    eval_id: str = Field(..., min_length=1)
    production_prediction: float
    candidate_prediction: float
    actual_value: float

class RegisterModelVersionRequest(BaseModel):
    model_name: str = Field(..., min_length=1, max_length=50)
    model_path: str = Field(..., min_length=1)
    rmse: float = Field(..., gt=0)
    r2_score: float = Field(default=0, ge=-1, le=1)
    metadata: Optional[Dict[str, Any]] = None

# --- ML Governance Initialization ---
drift_detector = DriftDetector(
    window_size=100,
    prediction_drift_threshold=0.2,
    input_drift_threshold=0.15,
)
shadow_evaluator = ShadowEvaluator(
    min_samples=50,
    error_improvement_threshold=0.05,
)
version_manager = ModelVersionManager(versions_dir="./model_versions")

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

notification_store = NotificationStore()

# Initialize and wire dependencies (best-effort; non-blocking)
try:
    from ml.governance import DriftDetector, ShadowEvaluator, ModelVersionManager
    from whatsapp_service import send_whatsapp_notification, format_alert_message
    from rbac import RBACManager, Permission
    from farm_finance_ai import FarmFinanceAI
    from crop_quality_grading import CropQualityGrader
    from blockchain_supply_chain import SupplyChainBlockchain

    drift_detector = DriftDetector()
    shadow_evaluator = ShadowEvaluator()
    version_manager = ModelVersionManager()
    governance.init_governance(drift_detector, shadow_evaluator, version_manager)

    def generate_alerts_fn(crop=None, irrigation_count=None, water_coverage=None, season=None):
        return []

    alerts.init_alerts(notification_store, {}, generate_alerts_fn, send_whatsapp_notification, format_alert_message, verify_role)

    farm_finance = FarmFinanceAI()
    rbac_mgr = RBACManager()
    finance.init_finance(farm_finance, rbac_mgr, Permission)

    crop_quality = CropQualityGrader()
    quality.init_quality(crop_quality, rbac_mgr, Permission)

    supply_chain = SupplyChainBlockchain()
    blockchain.init_blockchain(supply_chain)

    # Initialize Voice Assistant for Farmers
    from voice_assistant import VoiceAssistant, OfflineCacheManager
    voice_asst = VoiceAssistant(offline_mode=True)
    cache_mgr = OfflineCacheManager(cache_dir="./voice_assistant_cache")
    voice_assistant.init_voice_assistant(voice_asst, cache_mgr)
except Exception as e:
    logger.warning(f"Dependency wiring failed: {e}")

# ML Governance Request Models
class StartShadowEvaluationRequest(BaseModel):
    production_model: str = Field(..., min_length=1, max_length=50)
    candidate_model: str = Field(..., min_length=1, max_length=50)

class RecordPredictionRequest(BaseModel):
    eval_id: str = Field(..., min_length=1)
    production_prediction: float
    candidate_prediction: float
    actual_value: float

class RegisterModelVersionRequest(BaseModel):
    model_name: str = Field(..., min_length=1, max_length=50)
    model_path: str = Field(..., min_length=1)
    rmse: float = Field(..., gt=0)
    r2_score: float = Field(default=0, ge=-1, le=1)
    metadata: Optional[Dict[str, Any]] = None

# --- ML Governance Initialization ---
drift_detector = DriftDetector(
    window_size=100,
    prediction_drift_threshold=0.2,
    input_drift_threshold=0.15,
)
shadow_evaluator = ShadowEvaluator(
    min_samples=50,
    error_improvement_threshold=0.05,
)
version_manager = ModelVersionManager(versions_dir="./model_versions")

try:
    SEED_REGISTRY = {"TEST001": {"verified": True, "expiry": "2025-12-31"}}
    knowledge.init_knowledge(rag_generate_fn, rbac_mgr, Permission, SEED_REGISTRY, verify_role)

    try:
        import joblib
        import numpy as np
        model_lag = joblib.load("sklearn_yield_model.joblib")
        class ModelRouter:
            def predict(self, data, context):
                feature_array = np.array([data.get(k, 0) for k in ['CropCoveredArea', 'CHeight', 'IrriCount', 'WaterCov']])
                feature_array = feature_array.reshape(1, -1)
                return float(model_lag.predict(feature_array)[0])
        model_router_instance = ModelRouter()
        ml.init_router(model_router_instance, model_lag)
    except Exception:
        logger.warning("ML model not loaded")

except Exception as e:
    logger.warning(f"Dependency initialization skipped: {e}")


# Register routers
app.include_router(ml.router, prefix="/api/yield", tags=["ML Prediction"])
app.include_router(governance.router, prefix="/api/ml-governance", tags=["ML Governance"])
app.include_router(alerts.router, prefix="/api/notifications", tags=["Alerts"])
app.include_router(finance.router, prefix="/api/farm-finance", tags=["Finance"])
app.include_router(quality.router, prefix="/api/crop-quality", tags=["Quality"])
app.include_router(blockchain.router, prefix="/api/supply-chain", tags=["Blockchain"])
app.include_router(reports.router, prefix="/api/admin", tags=["Reports"])
app.include_router(knowledge.router, prefix="/api/knowledge", tags=["Knowledge"])
app.include_router(community.router, prefix="/api/community", tags=["Community"])
app.include_router(voice_assistant.router, prefix="/api/voice", tags=["Voice Assistant"])

# Initialize repositories for persistent storage
_finance_repository = FinanceApplicationRepository()
_notification_repository = NotificationRepository()
_supply_chain_repository = SupplyChainRepository()

# Initialize Crop Quality Grader
_crop_quality_grader = CropQualityGrader()

# Initialize Supply Chain Blockchain with persistent repository
_supply_chain_blockchain = SupplyChainBlockchain(repository=_supply_chain_repository)

# Initialize Farm Finance AI with persistent repository
_farm_finance_ai = FarmFinanceAI(repository=_finance_repository)

logger.info("Domain engines initialized with persistent repositories")

# Initialize repositories for persistent storage
_finance_repository = FinanceApplicationRepository()
_notification_repository = NotificationRepository()
_supply_chain_repository = SupplyChainRepository()

# Initialize Crop Quality Grader
_crop_quality_grader = CropQualityGrader()

# Initialize Supply Chain Blockchain with persistent repository
_supply_chain_blockchain = SupplyChainBlockchain(repository=_supply_chain_repository)

# Initialize Farm Finance AI with persistent repository
_farm_finance_ai = FarmFinanceAI(repository=_finance_repository)

logger.info("Domain engines initialized with persistent repositories")

# Initialize repositories for persistent storage
_finance_repository = FinanceApplicationRepository()
_notification_repository = NotificationRepository()
_supply_chain_repository = SupplyChainRepository()

# Initialize Crop Quality Grader
_crop_quality_grader = CropQualityGrader()

# Initialize Supply Chain Blockchain with persistent repository
_supply_chain_blockchain = SupplyChainBlockchain(repository=_supply_chain_repository)

# Initialize Farm Finance AI with persistent repository
_farm_finance_ai = FarmFinanceAI(repository=_finance_repository)

logger.info("Domain engines initialized with persistent repositories")

# Initialize repositories for persistent storage
_finance_repository = FinanceApplicationRepository()
_notification_repository = NotificationRepository()
_supply_chain_repository = SupplyChainRepository()

# Initialize Crop Quality Grader
_crop_quality_grader = CropQualityGrader()

# Initialize Supply Chain Blockchain with persistent repository
_supply_chain_blockchain = SupplyChainBlockchain(repository=_supply_chain_repository)

# Initialize Farm Finance AI with persistent repository
_farm_finance_ai = FarmFinanceAI(repository=_finance_repository)

logger.info("Domain engines initialized with persistent repositories")

# --- Routes ---

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "Fasal Saathi Backend", "version": "2.0"}

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

@app.post("/api/finance/analyze")
@limiter.limit("10/minute")
async def analyze_farm_finance(request: Request, body: FinanceAssessmentRequest):
    """Analyze farm finances and return loan recommendations."""
    # Check permission: farmer can create finance requests
    await RBACManager.raise_if_unauthorized(
        request, [Permission.FINANCE_CREATE], require_all=False
    )
    analysis = _farm_finance_ai.analyze_financial_profile(body.model_dump())
    return {"success": True, "data": analysis}


@app.post("/api/finance/applications")
@limiter.limit("5/minute")
async def create_finance_application(request: Request, body: FinanceAssessmentRequest):
    """Create a loan application from the current farm profile."""
    # Check permission: farmer can create finance applications
    await RBACManager.raise_if_unauthorized(
        request, [Permission.FINANCE_CREATE], require_all=False
    )
    application = _farm_finance_ai.create_application(body.model_dump())
    return {"success": True, "data": application}


@app.get("/api/finance/applications/{application_id}")
async def get_finance_application(application_id: str, request: Request):
    # Check permission: user can read finance applications (own or all if expert/admin)
    await RBACManager.raise_if_unauthorized(
        request, [Permission.FINANCE_READ_OWN, Permission.FINANCE_READ_ALL], require_all=False
    )
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

@app.post("/api/finance/analyze")
@limiter.limit("10/minute")
async def analyze_farm_finance(request: Request, body: FinanceAssessmentRequest):
    """Analyze farm finances and return loan recommendations."""
    # Check permission: farmer can create finance requests
    await RBACManager.raise_if_unauthorized(
        request, [Permission.FINANCE_CREATE], require_all=False
    )
    analysis = _farm_finance_ai.analyze_financial_profile(body.model_dump())
    return {"success": True, "data": analysis}


@app.post("/api/finance/applications")
@limiter.limit("5/minute")
async def create_finance_application(request: Request, body: FinanceAssessmentRequest):
    """Create a loan application from the current farm profile."""
    # Check permission: farmer can create finance applications
    await RBACManager.raise_if_unauthorized(
        request, [Permission.FINANCE_CREATE], require_all=False
    )
    application = _farm_finance_ai.create_application(body.model_dump())
    return {"success": True, "data": application}


@app.get("/api/finance/applications/{application_id}")
async def get_finance_application(application_id: str, request: Request):
    # Check permission: user can read finance applications (own or all if expert/admin)
    await RBACManager.raise_if_unauthorized(
        request, [Permission.FINANCE_READ_OWN, Permission.FINANCE_READ_ALL], require_all=False
    )
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

@app.post("/api/finance/analyze")
@limiter.limit("10/minute")
async def analyze_farm_finance(request: Request, body: FinanceAssessmentRequest):
    """Analyze farm finances and return loan recommendations."""
    # Check permission: farmer can create finance requests
    await RBACManager.raise_if_unauthorized(
        request, [Permission.FINANCE_CREATE], require_all=False
    )
    analysis = _farm_finance_ai.analyze_financial_profile(body.model_dump())
    return {"success": True, "data": analysis}


@app.post("/api/finance/applications")
@limiter.limit("5/minute")
async def create_finance_application(request: Request, body: FinanceAssessmentRequest):
    """Create a loan application from the current farm profile."""
    # Check permission: farmer can create finance applications
    await RBACManager.raise_if_unauthorized(
        request, [Permission.FINANCE_CREATE], require_all=False
    )
    application = _farm_finance_ai.create_application(body.model_dump())
    return {"success": True, "data": application}


@app.get("/api/finance/applications/{application_id}")
async def get_finance_application(application_id: str, request: Request):
    # Check permission: user can read finance applications (own or all if expert/admin)
    await RBACManager.raise_if_unauthorized(
        request, [Permission.FINANCE_READ_OWN, Permission.FINANCE_READ_ALL], require_all=False
    )
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

@app.post("/api/finance/analyze")
@limiter.limit("10/minute")
async def analyze_farm_finance(request: Request, body: FinanceAssessmentRequest):
    """Analyze farm finances and return loan recommendations."""
    # Check permission: farmer can create finance requests
    await RBACManager.raise_if_unauthorized(
        request, [Permission.FINANCE_CREATE], require_all=False
    )
    analysis = _farm_finance_ai.analyze_financial_profile(body.model_dump())
    return {"success": True, "data": analysis}


@app.post("/api/finance/applications")
@limiter.limit("5/minute")
async def create_finance_application(request: Request, body: FinanceAssessmentRequest):
    """Create a loan application from the current farm profile."""
    # Check permission: farmer can create finance applications
    await RBACManager.raise_if_unauthorized(
        request, [Permission.FINANCE_CREATE], require_all=False
    )
    application = _farm_finance_ai.create_application(body.model_dump())
    return {"success": True, "data": application}


@app.get("/api/finance/applications/{application_id}")
async def get_finance_application(application_id: str, request: Request):
    # Check permission: user can read finance applications (own or all if expert/admin)
    await RBACManager.raise_if_unauthorized(
        request, [Permission.FINANCE_READ_OWN, Permission.FINANCE_READ_ALL], require_all=False
    )
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
        
        # Data for signing: migrate from fragile delimiter-separated strings
        # to secure canonical structured JSON serialization
        signing_payload = {
            "name": data.name,
            "crop": data.crop,
            "area": data.area,
            "profit": data.profit,
            "season": data.season,
            "date": datetime.now().date().isoformat()
        }
        report_data_string = json.dumps(signing_payload, sort_keys=True)
        signature = private_key.sign(report_data_string.encode("utf-8"))
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


@app.post("/api/gemini/analyze-image")
@limiter.limit("10/minute")
async def gemini_analyze_image(request: Request, body: GeminiImageRequest):
    """
    Server-side proxy for Gemini multimodal image analysis.

    Keeps GEMINI_API_KEY on the server — it is never bundled into the
    frontend JavaScript. Frontend components (PestDetection, CropDiseaseDetection)
    send the base64 image and prompt here; this endpoint forwards the request
    to Google and returns the raw text response.

    Rate-limited to 10 requests/minute per IP to protect billing.
    Authentication is intentionally not required so unauthenticated users
    can still use the detection features, but the key stays server-side.
    """
    import httpx

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="AI analysis service is not configured"
        )

    gemini_url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.0-flash:generateContent?key={api_key}"
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": body.prompt},
                    {
                        "inline_data": {
                            "mime_type": body.mime_type,
                            "data": body.image_base64,
                        }
                    },
                ]
            }
        ]
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(gemini_url, json=payload)

        if resp.status_code != 200:
            logger.warning("Gemini API returned %s: %s", resp.status_code, resp.text[:200])
            raise HTTPException(
                status_code=502,
                detail="AI analysis service returned an error"
            )

        data = resp.json()
        text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")

        if not text:
            raise HTTPException(status_code=502, detail="Empty response from AI analysis service")

        return {"text": text}

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="AI analysis service timed out")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Gemini proxy error: %s", str(e))
        raise HTTPException(status_code=500, detail="AI analysis service unavailable")

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
    
    Requires authentication and appropriate role.
    """
    # RBAC: user can verify seeds
    await RBACManager.raise_if_unauthorized(
        request, [Permission.SEEDS_VERIFY], require_all=False
    )

    """
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
    # RBAC: user can assess crop quality
    await RBACManager.raise_if_unauthorized(
        request, [Permission.QUALITY_ASSESS], require_all=False
    )
    
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
    # RBAC: user can assess crop quality
    await RBACManager.raise_if_unauthorized(
        request, [Permission.QUALITY_ASSESS], require_all=False
    )
    
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

# ---------------------------------------------------------------------------
# Expert directory — fallback data used when Firestore is unavailable.
#
# IMPORTANT: This list must never contain real phone numbers, email addresses,
# or any other PII. Phone numbers are omitted entirely (null) so that even if
# this fallback is served, no contact details are exposed to unauthenticated
# callers. Avatar images use a local placeholder path instead of a third-party
# service (randomuser.me) that would log every request.
# ---------------------------------------------------------------------------
_FALLBACK_EXPERTS = [
    {
        "id": "exp1",
        "name": "Dr. Ramesh Kumar",
        "specialization": "crop_disease",
        "qualification": "Ph.D. in Plant Pathology",
        "location": "Madhya Pradesh",
        "phone": None,
        "rating": 4.8,
        "experience": 15,
        "is_kvk": True,
        "kvk_name": "KVK Jabalpur",
        "bio": "Specialist in crop disease diagnosis and organic treatment methods.",
        "avatar": "/assets/avatars/placeholder.svg"
    },
    {
        "id": "exp2",
        "name": "Dr. Priya Sharma",
        "specialization": "fertilizers",
        "qualification": "M.Sc. Agricultural Chemistry",
        "location": "Maharashtra",
        "phone": None,
        "rating": 4.9,
        "experience": 12,
        "is_kvk": True,
        "kvk_name": "KVK Pune",
        "bio": "Expert in nano-fertilizers and sustainable nutrient management.",
        "avatar": "/assets/avatars/placeholder.svg"
    },
    {
        "id": "exp3",
        "name": "Er. Suresh Patil",
        "specialization": "irrigation",
        "qualification": "B.Tech Agricultural Engineering",
        "location": "Karnataka",
        "phone": None,
        "rating": 4.7,
        "experience": 10,
        "is_kvk": False,
        "bio": "Drip irrigation and water management specialist.",
        "avatar": "/assets/avatars/placeholder.svg"
    },
    {
        "id": "exp4",
        "name": "Dr. Anjali Verma",
        "specialization": "pest_management",
        "qualification": "Ph.D. Entomology",
        "location": "Uttar Pradesh",
        "phone": None,
        "rating": 4.6,
        "experience": 8,
        "is_kvk": True,
        "kvk_name": "KVK Lucknow",
        "bio": "Integrated pest management and organic pest control expert.",
        "avatar": "/assets/avatars/placeholder.svg"
    },
    {
        "id": "exp5",
        "name": "Dr. Mahendra Singh",
        "specialization": "soil_health",
        "qualification": "Ph.D. Soil Science",
        "location": "Rajasthan",
        "phone": None,
        "rating": 4.9,
        "experience": 20,
        "is_kvk": True,
        "kvk_name": "KVK Jaipur",
        "bio": "Soil health assessment and reclamation specialist.",
        "avatar": "/assets/avatars/placeholder.svg"
    },
    {
        "id": "exp6",
        "name": "Dr. Kavita Desai",
        "specialization": "market_advisory",
        "qualification": "MBA Agriculture Business",
        "location": "Gujarat",
        "phone": None,
        "rating": 4.8,
        "experience": 14,
        "is_kvk": False,
        "bio": "Market intelligence and price forecasting expert.",
        "avatar": "/assets/avatars/placeholder.svg"
    },
]


@app.get("/api/experts")
@limiter.limit("20/minute")
async def get_experts(
    request: Request,
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

    Phone numbers are never included in the response — contact details are
    only accessible to authenticated users through the consultation booking
    flow, which enforces its own RBAC check.
    """
    try:
        experts = []

        db = get_firestore_client()
        if db:
            try:
                experts_ref = db.collection("experts")
                docs = experts_ref.get()
                for doc in docs:
                    expert_data = doc.to_dict()
                    expert_data["id"] = doc.id
                    # Strip phone numbers from Firestore records too — callers
                    # must go through the authenticated booking flow to get them.
                    expert_data.pop("phone", None)
                    expert_data.pop("email", None)
                    experts.append(expert_data)
            except Exception as e:
                logger.warning("Firestore experts query failed: %s", e)

        # Fall back to the static list when Firestore is unavailable or empty.
        # The static list already has phone=None so no PII is exposed.
        if not experts:
            experts = list(_FALLBACK_EXPERTS)

        # Apply filters once, after the source has been resolved
        if specialization:
            experts = [e for e in experts if e.get("specialization") == specialization]
        if kvk_only:
            experts = [e for e in experts if e.get("is_kvk")]
        if search:
            search_lower = search.lower()
            experts = [
                e for e in experts
                if search_lower in e.get("name", "").lower()
                or search_lower in e.get("location", "").lower()
            ]

        return {"experts": experts, "count": len(experts)}

    except Exception as e:
        logger.error("Error fetching experts: %s", e)
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
@limiter.limit("5/minute")
async def book_consultation(
    request: ConsultationBookRequest,
    raw_request: Request,
    authorization: Optional[str] = None
):
    """
    Book a consultation with an expert.

    Authentication required — any logged-in user may book.
    The caller's identity is derived from the verified Firebase token;
    the client-supplied user_id is never trusted.
    """
    # Idempotency check: prevent duplicate bookings on retry
    idem_key = raw_request.headers.get("X-Idempotency-Key")
    if idem_key:
        with _IDEMPOTENCY_LOCK:
            if idem_key in _IDEMPOTENCY_CACHE:
                logger.info(f"Idempotency: Returning cached consultation for key {idem_key}")
                return _IDEMPOTENCY_CACHE[idem_key]

    """
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
        db = validate_firestore_ready()

        # Require authentication — derive identity from the verified token,
        # never from the request body.  Previously authorization was Optional
        # and the entire auth block was wrapped in a try/except that silently
        # fell through, writing bookings with user_id="anonymous".
        token_data = await verify_role(raw_request)
        uid = token_data["uid"]

        # Fetch the user's display name from Firestore for the booking record.
        user_name = "Farmer"
        try:
            user_doc = db.collection("users").document(uid).get()
            if user_doc.exists:
                user_name = user_doc.to_dict().get("displayName", "Farmer") or "Farmer"
        except Exception as e:
            logger.warning("Could not fetch display name for uid=%s: %s", uid, e)

        consultation_data = {
            "expert_id": request.expert_id,
            "expert_name": request.expert_name,
            "expert_specialization": request.expert_specialization,
            "user_id": uid,
            "user_name": user_name,
            "date": request.date,
            "time": request.time,
            "notes": request.notes,
            "type": request.consultation_type,
            "status": "scheduled",
            "created_at": datetime.now().isoformat()
        }

        doc_ref = db.collection("consultations").document()
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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error booking consultation: {e}")
        raise HTTPException(status_code=500, detail="Failed to book consultation")


@app.put("/api/consultation/{consultation_id}")
async def update_consultation(
    consultation_id: str,
    request: ConsultationUpdateRequest,
    raw_request: Request,
    authorization: Optional[str] = None
):
    """
    Update consultation status.
    Only the owner of the consultation may update it.
    """
    try:
        db = validate_firestore_ready()

        # Require authentication and verify ownership.
        token_data = await verify_role(raw_request)
        uid = token_data["uid"]

        doc_ref = db.collection("consultations").document(consultation_id)
        doc = doc_ref.get()

        if not doc.exists:
            raise HTTPException(status_code=404, detail="Consultation not found")

        # Prevent one user from updating another user's consultation.
        owner_uid = doc.to_dict().get("user_id")
        if owner_uid != uid and token_data.get("role") not in ("admin", "expert"):
            raise HTTPException(
                status_code=403,
                detail="You do not have permission to update this consultation"
            )

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
    raw_request: Request,
    user_id: Optional[str] = None,
    status: Optional[str] = None,
    authorization: Optional[str] = None
):
    """
    Get the authenticated user's consultation history.
    Results are always scoped to the caller's own uid — the client-supplied
    user_id parameter is ignored to prevent IDOR enumeration.
    """
    try:
        db = validate_firestore_ready()

        # Require authentication and scope results to the caller's own uid.
        # Previously user_id was accepted from the query string with no auth
        # check, allowing any caller to read any user's consultation history.
        token_data = await verify_role(raw_request)
        uid = token_data["uid"]

        consultations_ref = db.collection("consultations")
        query = consultations_ref.where("user_id", "==", uid)

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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching consultations: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch consultations")


@app.delete("/api/consultation/{consultation_id}")
async def cancel_consultation(
    consultation_id: str,
    raw_request: Request,
    authorization: Optional[str] = None
):
    """
    Cancel a consultation.
    Only the owner of the consultation may cancel it.
    """
    try:
        db = validate_firestore_ready()

        # Require authentication and verify ownership.
        token_data = await verify_role(raw_request)
        uid = token_data["uid"]

        doc_ref = db.collection("consultations").document(consultation_id)
        doc = doc_ref.get()

        if not doc.exists:
            raise HTTPException(status_code=404, detail="Consultation not found")

        owner_uid = doc.to_dict().get("user_id")
        if owner_uid != uid and token_data.get("role") not in ("admin", "expert"):
            raise HTTPException(
                status_code=403,
                detail="You do not have permission to cancel this consultation"
            )

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
#
# Auth policy:
#   Write operations (register-actor, create-batch, add-node, create-contract,
#   execute-contract) require authentication and the appropriate supply chain
#   permission. This prevents unauthenticated callers from forging provenance
#   records or flooding the chain with garbage data.
#
#   Read operations (verify, journey, analytics, marketplace, stats) are
#   intentionally public — consumers need to scan QR codes and verify batch
#   authenticity without creating an account.
#
#   QR code generation requires SUPPLY_CHAIN_READ so only registered
#   participants can generate codes for physical packaging.

@app.post("/api/blockchain/register-actor")
@limiter.limit("10/minute")
async def register_actor(request: Request, data: RegisterActorRequest):
    """Register supply chain participant. Requires SUPPLY_CHAIN_CREATE permission."""
    await RBACManager.raise_if_unauthorized(
        request, [Permission.SUPPLY_CHAIN_CREATE], require_all=False
    )
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
    """Create product batch on blockchain. Requires SUPPLY_CHAIN_CREATE permission."""
    await RBACManager.raise_if_unauthorized(
        request, [Permission.SUPPLY_CHAIN_CREATE], require_all=False
    )
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
    """Add supply chain node. Requires SUPPLY_CHAIN_UPDATE permission."""
    await RBACManager.raise_if_unauthorized(
        request, [Permission.SUPPLY_CHAIN_UPDATE], require_all=False
    )
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
    """Create smart contract. Requires SUPPLY_CHAIN_CREATE permission."""
    await RBACManager.raise_if_unauthorized(
        request, [Permission.SUPPLY_CHAIN_CREATE], require_all=False
    )
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
    """Execute smart contract. Requires SUPPLY_CHAIN_UPDATE permission."""
    await RBACManager.raise_if_unauthorized(
        request, [Permission.SUPPLY_CHAIN_UPDATE], require_all=False
    )
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
    """Get QR code for batch. Requires SUPPLY_CHAIN_READ permission."""
    await RBACManager.raise_if_unauthorized(
        request, [Permission.SUPPLY_CHAIN_READ], require_all=False
    )
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
async def verify_batch(batch_id: str):
    """Verify batch authenticity. Public — consumers scan QR codes without an account."""
    try:
        verification = _supply_chain_blockchain.verify_batch(batch_id)
        return verification
    except Exception as e:
        logger.error("Verification error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/blockchain/journey/{batch_id}")
@limiter.limit("20/minute")
async def get_journey(batch_id: str):
    """Get supply chain journey. Public — consumers scan QR codes without an account."""
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
async def get_analytics(batch_id: str):
    """Get supply chain analytics. Public."""
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
async def get_marketplace():
    """Get certified products for marketplace. Public."""
    try:
        certified = _supply_chain_blockchain.get_certified_products()
        return {"success": True, "products": certified, "count": len(certified)}
    except Exception as e:
        logger.error("Marketplace error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/blockchain/stats")
@limiter.limit("20/minute")
async def get_stats():
    """Get blockchain statistics. Public."""
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

        @app.post("/api/ml-governance/drift/baseline")
        @limiter.limit("5/minute")
        async def set_drift_baseline(request: Request, model_name: str, predictions: list):
            """
            Set baseline statistics for drift detection on a model
    
            Query params:
                model_name: Name of the model (e.g., 'xgboost')
    
            Body:
                predictions: List of baseline prediction values
            """
            try:
                if not predictions or len(predictions) < 10:
                    raise ValueError("Need at least 10 baseline predictions")
        
                drift_detector.set_baseline(model_name, predictions)
        
                return {
                    "success": True,
                    "message": f"Baseline set for model '{model_name}' with {len(predictions)} samples",
                    "model_name": model_name
                }
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error("Drift baseline error: %s", str(e))
                raise HTTPException(status_code=500, detail="Failed to set baseline")

        @app.post("/api/ml-governance/drift/check")
        @limiter.limit("100/minute")
        async def check_drift(request: Request, model_name: str, prediction: float, actual_value: float = None):
            """
            Check if a prediction indicates drift
    
            Query params:
                model_name: Name of the model
                prediction: Current prediction value
                actual_value: Optional actual observed value
            """
            try:
                is_drift, alert = drift_detector.check_prediction_drift(
                    model_name, prediction, actual_value
                )
        
                response = {
                    "drift_detected": is_drift,
                    "model_name": model_name,
                    "prediction": prediction
                }
        
                if alert:
                    response["alert"] = alert.to_dict()
        
                return response
            except Exception as e:
                logger.error("Drift check error: %s", str(e))
                raise HTTPException(status_code=500, detail="Failed to check drift")

        @app.get("/api/ml-governance/drift/alerts")
        @limiter.limit("20/minute")
        async def get_drift_alerts(request: Request, model_name: str = None, limit: int = 10):
            """Get recent drift alerts"""
            try:
                alerts = drift_detector.get_alerts(model_name, limit)
                return {"success": True, "alerts": alerts, "count": len(alerts)}
            except Exception as e:
                logger.error("Get alerts error: %s", str(e))
                raise HTTPException(status_code=500, detail="Failed to retrieve alerts")

        @app.post("/api/ml-governance/shadow/start")
        @limiter.limit("10/minute")
        async def start_shadow_evaluation(request: Request, data: StartShadowEvaluationRequest):
            """
            Start shadow evaluation comparing production and candidate models
    
            Request body:
            {
                "production_model": "xgboost_v1",
                "candidate_model": "xgboost_v2"
            }
            """
            try:
                eval_id = shadow_evaluator.start_shadow_evaluation(
                    data.production_model,
                    data.candidate_model
                )
        
                return {
                    "success": True,
                    "eval_id": eval_id,
                    "production_model": data.production_model,
                    "candidate_model": data.candidate_model
                }
            except Exception as e:
                logger.error("Shadow eval start error: %s", str(e))
                raise HTTPException(status_code=500, detail="Failed to start shadow evaluation")

        @app.post("/api/ml-governance/shadow/record")
        @limiter.limit("200/minute")
        async def record_shadow_predictions(request: Request, data: RecordPredictionRequest):
            """
            Record predictions from both models during shadow evaluation
    
            Request body:
            {
                "eval_id": "eval_20250516_120000",
                "production_prediction": 100.5,
                "candidate_prediction": 101.2,
                "actual_value": 102.0
            }
            """
            try:
                shadow_evaluator.record_predictions(
                    data.eval_id,
                    data.production_prediction,
                    data.candidate_prediction,
                    data.actual_value
                )
        
                status = shadow_evaluator.get_evaluation_status(data.eval_id)
        
                return {
                    "success": True,
                    "eval_id": data.eval_id,
                    "status": status
                }
            except Exception as e:
                logger.error("Record predictions error: %s", str(e))
                raise HTTPException(status_code=500, detail="Failed to record predictions")

        @app.post("/api/ml-governance/shadow/evaluate")
        @limiter.limit("10/minute")
        async def evaluate_candidate_model(request: Request, eval_id: str):
            """Evaluate candidate model performance"""
            try:
                result = shadow_evaluator.evaluate_candidate(eval_id)
        
                if result is None:
                    status = shadow_evaluator.get_evaluation_status(eval_id)
                    return {
                        "success": False,
                        "message": "Not enough samples for evaluation",
                        "status": status
                    }
        
                return {
                    "success": True,
                    "evaluation": result.to_dict()
                }
            except Exception as e:
                logger.error("Evaluate candidate error: %s", str(e))
                raise HTTPException(status_code=500, detail="Failed to evaluate candidate")

        @app.get("/api/ml-governance/shadow/status/{eval_id}")
        @limiter.limit("50/minute")
        async def get_shadow_eval_status(request: Request, eval_id: str):
            """Get status of shadow evaluation"""
            try:
                status = shadow_evaluator.get_evaluation_status(eval_id)
                return {"success": True, "status": status}
            except Exception as e:
                logger.error("Get shadow status error: %s", str(e))
                raise HTTPException(status_code=500, detail="Failed to get status")

        @app.post("/api/ml-governance/versions/register")
        @limiter.limit("5/minute")
        async def register_model_version(request: Request, data: RegisterModelVersionRequest):
            """
            Register a new model version
    
            Request body:
            {
                "model_name": "xgboost",
                "model_path": "/path/to/model.joblib",
                "rmse": 0.15,
                "r2_score": 0.85,
                "metadata": {"author": "ml-team"}
            }
            """
            try:
                performance_metrics = {
                    'rmse': data.rmse,
                    'r2_score': data.r2_score,
                }
        
                version_id = version_manager.register_version(
                    data.model_name,
                    data.model_path,
                    performance_metrics,
                    data.metadata
                )
        
                return {
                    "success": True,
                    "version_id": version_id,
                    "model_name": data.model_name
                }
            except Exception as e:
                logger.error("Register version error: %s", str(e))
                raise HTTPException(status_code=500, detail="Failed to register version")

        @app.post("/api/ml-governance/versions/promote")
        @limiter.limit("5/minute")
        async def promote_model_version(request: Request, version_id: str):
            """Promote a model version to production"""
            try:
                success = version_manager.promote_version(version_id)
        
                if not success:
                    raise ValueError("Failed to promote version")
        
                prod_version = version_manager.get_production_version()
        
                return {
                    "success": True,
                    "message": f"Promoted {version_id} to production",
                    "production_version": prod_version.to_dict() if prod_version else None
                }
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error("Promote version error: %s", str(e))
                raise HTTPException(status_code=500, detail="Failed to promote version")

        @app.post("/api/ml-governance/versions/rollback")
        @limiter.limit("5/minute")
        async def rollback_model_version(request: Request, version_id: str):
            """Rollback to a previous model version (emergency only)"""
            try:
                success = version_manager.rollback_to_version(version_id)
        
                if not success:
                    raise ValueError("Rollback failed")
        
                prod_version = version_manager.get_production_version()
        
                return {
                    "success": True,
                    "message": f"Rolled back to {version_id}",
                    "production_version": prod_version.to_dict() if prod_version else None
                }
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error("Rollback version error: %s", str(e))
                raise HTTPException(status_code=500, detail="Failed to rollback")

        @app.get("/api/ml-governance/versions/production")
        @limiter.limit("50/minute")
        async def get_production_version(request: Request):
            """Get current production model version"""
            try:
                prod_version = version_manager.get_production_version()
        
                if not prod_version:
                    return {
                        "success": True,
                        "production_version": None,
                        "message": "No production version set"
                    }
        
                return {
                    "success": True,
                    "production_version": prod_version.to_dict()
                }
            except Exception as e:
                logger.error("Get production version error: %s", str(e))
                raise HTTPException(status_code=500, detail="Failed to get production version")

        @app.get("/api/ml-governance/versions/list")
        @limiter.limit("20/minute")
        async def list_model_versions(request: Request, model_name: str = None):
            """List all model versions"""
            try:
                versions = version_manager.list_versions(model_name)
        
                return {
                    "success": True,
                    "versions": versions,
                    "total": len(versions)
                }
            except Exception as e:
                logger.error("List versions error: %s", str(e))
                raise HTTPException(status_code=500, detail="Failed to list versions")

        @app.get("/api/ml-governance/versions/compare")
        @limiter.limit("20/minute")
        async def compare_model_versions(request: Request, v1: str, v2: str):
            """Compare two model versions"""
            try:
                comparison = version_manager.compare_versions(v1, v2)
        
                if 'error' in comparison:
                    raise ValueError(comparison['error'])
        
                return {
                    "success": True,
                    "comparison": comparison
                }
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error("Compare versions error: %s", str(e))
                raise HTTPException(status_code=500, detail="Failed to compare versions")

        @app.get("/api/ml-governance/status")
        @limiter.limit("20/minute")
        async def get_governance_status(request: Request):
            """Get overall ML governance status"""
            try:
                prod_version = version_manager.get_production_version()
                recent_alerts = drift_detector.get_alerts(limit=5)
                recent_evals = shadow_evaluator.get_evaluations(limit=5)
        
                return {
                    "success": True,
                    "governance_status": {
                        "drift_detection": {
                            "recent_alerts": recent_alerts,
                            "alert_count": len(drift_detector.alerts)
                        },
                        "shadow_evaluation": {
                            "active_evaluations": len(shadow_evaluator.active_evaluations),
                            "completed_evaluations": len(shadow_evaluator.evaluations),
                            "recent_evaluations": recent_evals
                        },
                        "model_versioning": {
                            "production_version": prod_version.to_dict() if prod_version else None,
                            "total_versions": len(version_manager.versions)
                        }
                    }
                }
            except Exception as e:
                logger.error("Get governance status error: %s", str(e))
                raise HTTPException(status_code=500, detail="Failed to get governance status")
        raise HTTPException(status_code=500, detail=str(e))
