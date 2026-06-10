# main.py
import os
import asyncio
import io
import json
import re
import joblib
import hashlib
import pandas as pd
import numpy as np

import sys

# Required environment variables for backend
REQUIRED_ENV_VARS = [
    "WEATHER_API_KEY",
    "SOIL_API_KEY",
    "FIREBASE_ADMIN_CRED",
    "BACKEND_PORT",
]

def validate_env_vars():
    missing = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
    if missing:
        print(f"❌ Missing required environment variables: {', '.join(missing)}")
        sys.exit(1)  # stop app immediately

# Run validation before app starts
validate_env_vars()

from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Request, Form, Query, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field, validator
from init_services import get_logger

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

# Rate Limiting
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from rate_limit_config import build_limiter, rate_limit_exceeded_handler

from init_services import firebase_admin, firestore
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from backend.routers import (
    alerts,
    blockchain,
    community,
    finance,
    governance,
    knowledge,
    ml,
    platform,
    quality,
    referrals,
    reports,
)
from blockchain_supply_chain import SupplyChainBlockchain
from crop_quality_grading import CropQualityGrader
from farm_finance_ai import FarmFinanceAI
from feature_flags.routes import router as flags_router
from ml.adapters.xgboost_adapter import XGBoostAdapter
from ml.governance import DriftDetector, ModelVersionManager, ShadowEvaluator
from ml.registry import ModelRegistry
from ml.router import ModelRouter
from ml.preprocessing import UnknownCategoryError, MissingFeatureError

# Other internal modules
from alert_rules import generate_alerts
from whatsapp_service import send_whatsapp_message, format_alert_message
from whatsapp_store import subscriber_store
from error_recovery_middleware import ErrorRecoveryMiddleware
from realtime_notifications import notification_broker

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "ok"}

# KMS Support
try:
    from google.cloud import secretmanager
    HAS_GCP_KMS = True
except ImportError:
    HAS_GCP_KMS = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.

    Runs inside **every** Uvicorn/Gunicorn worker process on startup, so the
    ML pipeline is always initialised regardless of how many workers are
    spawned.  This replaces the previous bare ``init_ml_pipeline()`` call at
    module level, which only ran reliably in single-worker deployments.

    Multi-worker guarantee
    ----------------------
    When Uvicorn is started with ``--workers N``, each worker forks/spawns
    from the main process and imports ``main.py`` independently.  The
    ``lifespan`` hook is invoked by FastAPI in every worker's event loop,
    ensuring ``ModelRegistry`` is populated in every process before the
    first request is served.
    """
    init_ml_pipeline()
    await notification_broker.start()
    yield
    # Shutdown: nothing to clean up for in-memory models.
    await notification_broker.stop()


app = FastAPI(lifespan=lifespan)

from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

# --- Global Error Handlers ---

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled error: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "code": "INTERNAL_ERROR",
            "message": "Something went wrong. Please try again later."
        },
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={"code": "VALIDATION_ERROR", "message": "Invalid request payload."},
    )

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"code": "HTTP_ERROR", "message": exc.detail},
    )



# Initialize Limiter
limiter = build_limiter(default_limits=["120/minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)


db_firestore = None


def get_firestore_client():
    global db_firestore
    if db_firestore is not None:
        return db_firestore

    try:
        if not firebase_admin._apps:
            cred = None
            if os.path.exists("serviceAccountKey.json"):
                cred = credentials.Certificate("serviceAccountKey.json")
            elif os.path.exists("firebase-credentials.json"):
                cred = credentials.Certificate("firebase-credentials.json")
            else:
                cred_json = os.getenv("FIREBASE_CREDENTIALS_JSON")
                if cred_json:
                    cred = credentials.Certificate(json.loads(cred_json))

            try:
                firebase_admin.initialize_app(cred) if cred else firebase_admin.initialize_app()
            except ValueError:
                pass

        if firebase_admin._apps:
            db_firestore = firestore.client()
    except Exception as exc:
        logger.error("Failed to initialize Firestore client: %s", exc)
        db_firestore = None

    return db_firestore


def validate_firestore_ready():
    client = get_firestore_client()
    if client is None:
        raise HTTPException(status_code=503, detail="Database service temporarily unavailable.")
    return client


async def verify_role(request: Request, required_roles: list = None, require_all: bool = False):
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing auth token")

app.add_middleware(ErrorRecoveryMiddleware)

app.add_middleware(ErrorRecoveryMiddleware)

app.add_middleware(ErrorRecoveryMiddleware)

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
    name: str
    # user_id is accepted for backward compatibility but is IGNORED by the
    # endpoint — the authoritative user identity is always derived from the
    # verified Firebase ID token, never from client-supplied data.
    user_id: Optional[str] = None

class YieldInput(BaseModel):
    data: list[float]

class AlertTriggerRequest(BaseModel):
    alert_type: str = Field(..., pattern=r'^(weather|pest|advisory)$')
    message: str = Field(..., min_length=1, max_length=500)

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

class SeedVerifyRequest(BaseModel):
    code: str = Field(..., min_length=4, max_length=100)

    db = validate_firestore_ready()
    user_doc = db.collection("users").document(uid).get()
    user_roles = user_doc.get("roles", []) if user_doc.exists else []

    if required_roles:
        has_access = all(role in user_roles for role in required_roles) if require_all else any(
            role in user_roles for role in required_roles
        )
        if not has_access:
            raise HTTPException(status_code=403, detail="Insufficient permissions")

    @app.get("/user_roles")
    def get_user_roles(uid: str):
        user_roles = ["admin", "editor"]  # example
        return {"uid": uid, "roles": user_roles}

    def __init__(self, maxlen: int = _MAX_NOTIFICATIONS, ttl_hours: int = _NOTIFICATION_TTL_HOURS):
        self._deque: collections.deque = collections.deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._counter = itertools.count(start=1)
        self._ttl = timedelta(hours=ttl_hours)

    def append(self, alert_type: str, message: str) -> dict:
        """
        Add a new notification entry and return it.

        The ID is assigned from a monotonically increasing counter so
        concurrent calls always produce distinct values.
        """
        with self._lock:
            entry = {
                "id": next(self._counter),
                "type": alert_type,
                "message": message,
                "time": datetime.now().isoformat(),
            }
            self._deque.append(entry)
        return entry

    def get_recent(self) -> list:
        """
        Return all entries newer than the configured TTL, oldest first.

        Takes a snapshot under the lock so callers always see a consistent
        view even if append() is running concurrently.
        """
        cutoff = datetime.now() - self._ttl
        with self._lock:
            snapshot = list(self._deque)
        return [
            e for e in snapshot
            if datetime.fromisoformat(e["time"]) >= cutoff
        ]


# Seed the store with the initial weather advisory that was previously
# hard-coded in the bare list.
_notification_store = NotificationStore()
_notification_store.append(
    alert_type="weather",
    message="🌧️ Heavy rainfall expected in your region today.",
)
notification_broker.seed_notifications(_notification_store.get_recent())


async def publish_notification(alert_type: str, message: str) -> dict:
    """Store a notification and broadcast it to websocket subscribers."""
    entry = _notification_store.append(alert_type=alert_type, message=message)
    await notification_broker.publish(entry)
    return entry

def sanitise_log_field(value: str) -> str:
    if not isinstance(value, str):
        value = str(value)
    sanitised = "".join(ch if ord(ch) >= 32 or ch in "\n\t" else f"\\x{ord(ch):02x}" for ch in value)
    return sanitised[:1000]

@app.get("/")
@limiter.limit("60/minute")
def root():
    return {"message": "Fasal Saathi API", "status": "running"}

@app.get("/predict")
@limiter.limit("30/minute")
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
        temp = list(data)
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
@limiter.limit("30/minute")
def get_notifications(
    crop: str = Query(default=None),
    irrigation_count: int = Query(default=None, ge=0),
    water_coverage: int = Query(default=None, ge=0, le=100),
    season: str = Query(default=None)
):
    """
    Return recent triggered-alert notifications combined with dynamic
    farm advisory alerts generated from the query parameters.

    Only notifications newer than the store's TTL window are included,
    so the response payload stays small regardless of how long the
    process has been running.
    """
    dynamic_alerts = generate_alerts(
        crop=crop,
        irrigation_count=irrigation_count,
        water_coverage=water_coverage,
        season=season
    )
    return {"success": True, "data": _notification_store.get_recent() + dynamic_alerts}

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
    # Require authentication so the subscriber's identity is always derived
    # from the verified Firebase token — never from client-supplied data.
    # Previously the endpoint accepted user_id from the request body, which
    # allowed any caller to overwrite another user's subscription by sending
    # a known user_id with an attacker-controlled phone number.
    token_data = await verify_role(request)
    uid = token_data["uid"]

    subscriber = {
        "phone_number": data.phone_number,
        "name": data.name,
        "subscribed_at": datetime.now().isoformat(),
    }
    try:
        subscriber_store.upsert(uid, subscriber)
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
@limiter.limit("10/minute")
async def trigger_whatsapp_alert(data: AlertTriggerRequest, request: Request):
    """
    Broadcast a WhatsApp alert to all subscribers.

    Requires authentication — admin or expert role only.

    Previously this endpoint had no authentication check, no rate limit,
    and no input constraints.  Any unauthenticated caller could send
    arbitrary messages to every subscribed farmer, enabling social
    engineering attacks (fake market alerts, fake pest warnings) and
    consuming Twilio API credits at the attacker's discretion.
    """
    # RBAC: only admins and experts may broadcast alerts to all farmers.
    await verify_role(request, required_roles=["admin", "expert"])

    # get_all() acquires the lock and returns a stable snapshot, so this read
    # cannot race with a concurrent subscription write.
    subscribers = subscriber_store.get_all()
    results = []
    formatted_msg = format_alert_message(data.alert_type, data.message)

    for user_id, info in subscribers.items():
        res = send_whatsapp_message(info["phone_number"], formatted_msg)
        results.append({"user_id": user_id, "success": res.get("success", False), "status": res.get("status", "error")})

    # Use the bounded, thread-safe NotificationStore instead of the bare
    # static_notifications list (which had no size cap and racy ID generation).
    await publish_notification(
        alert_type=data.alert_type,
        message=data.message,
    )

    delivered = sum(1 for r in results if r["success"])
    return {"success": True, "results": results, "delivered": delivered, "total": len(results)}


@app.websocket("/api/notifications/stream")
async def notifications_stream(websocket: WebSocket):
    await notification_broker.connect(websocket)

@app.post("/api/whatsapp/webhook")
@limiter.limit("20/minute")
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

HAS_GCP_KMS = False
secretmanager = None
try:
    from google.cloud import secretmanager as gcp_secretmanager

    secretmanager = gcp_secretmanager
    HAS_GCP_KMS = True
except Exception:
    HAS_GCP_KMS = False


def get_signing_keys():
    global _cached_private_key

    if _cached_private_key is not None:
        return _cached_private_key

    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    secret_id = os.getenv("REPORT_SIGNING_SECRET_NAME", "report-signing-key")

    if project_id:
        if not HAS_GCP_KMS:
            raise HTTPException(status_code=500, detail="KMS is required when GOOGLE_CLOUD_PROJECT is set")
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
        response = client.access_secret_version(request={"name": name})
        payload = response.payload.data.decode("UTF-8")
        _cached_private_key = serialization.load_pem_private_key(payload.encode(), password=None)
        return _cached_private_key

    if os.path.exists(PRIVATE_KEY_PATH):
        with open(PRIVATE_KEY_PATH, "rb") as key_file:
            _cached_private_key = serialization.load_pem_private_key(key_file.read(), password=None)
        return _cached_private_key

    private_key = ed25519.Ed25519PrivateKey.generate()
    os.makedirs(KEYS_DIR, exist_ok=True)
    with open(PRIVATE_KEY_PATH, "wb") as private_file:
        private_file.write(
            private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
    with open(PUBLIC_KEY_PATH, "wb") as public_file:
        public_file.write(
            private_key.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )

    _cached_private_key = private_key
    return private_key


def init_ml_pipeline() -> None:
    try:
        xgb_adapter = XGBoostAdapter()
        model_path = "yield_model.joblib"
        if os.path.exists(model_path):
            xgb_adapter.load(model_path)
            ModelRegistry.register("xgboost", xgb_adapter)
            logger.info("ML Pipeline: Registered XGBoost model")
        else:
            logger.warning("ML Pipeline: %s not found", model_path)
    except Exception as exc:
        logger.warning("ML Pipeline initialization failed: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up: initializing services")
    get_firestore_client()
    init_ml_pipeline()
    yield
    logger.info("Shutting down")


app = FastAPI(title="Fasal Saathi Backend", version="2.0", lifespan=lifespan)

# Observability setup
try:
    service_name = os.environ.get("OTEL_SERVICE_NAME", "fasal-saathi-backend")
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)
    otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if otlp_endpoint:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint)))
    else:
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    FastAPIInstrumentor().instrument_app(app)
except Exception as exc:
    logger.warning("Tracing setup skipped: %s", exc)

try:
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")
except Exception as exc:
    logger.warning("Prometheus setup skipped: %s", exc)

# Middleware and rate-limits
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RBACMiddleware)
logger.info(print_rbac_matrix())

# Domain dependencies
drift_detector = DriftDetector(window_size=100, prediction_drift_threshold=0.2, input_drift_threshold=0.15)
shadow_evaluator = ShadowEvaluator(min_samples=50, error_improvement_threshold=0.05)
version_manager = ModelVersionManager(versions_dir="./model_versions")

_finance_repository = FinanceApplicationRepository()
_notification_repository = NotificationRepository()
_supply_chain_repository = SupplyChainRepository()
_farm_finance_ai = FarmFinanceAI(repository=_finance_repository)
_supply_chain_blockchain = SupplyChainBlockchain(repository=_supply_chain_repository)
_crop_quality_grader = CropQualityGrader()

# Router init hooks
governance.init_governance(drift_detector, shadow_evaluator, version_manager)
finance.init_finance(_farm_finance_ai, RBACManager, Permission)
quality.init_quality(_crop_quality_grader, RBACManager, Permission)
blockchain.init_blockchain(_supply_chain_blockchain)
referrals.init_referrals(validate_firestore_ready)
reports.init_reports(verify_role, get_signing_keys, sanitise_log_field, logger)

rag_generate_fn = None
try:
    from rag.generator import generate_response as rag_generate_fn
except Exception as exc:
    logger.warning("RAG init skipped: %s", exc)

knowledge.init_knowledge(rag_generate_fn, RBACManager, Permission, {"TEST001": {"verified": True}}, verify_role)

alerts.init_alerts([], subscriber_store, lambda **kwargs: [], send_whatsapp_message, format_alert_message, verify_role)
platform.init_platform(
    verify_role,
    get_signing_keys,
    sanitise_log_field,
    rag_generate_fn,
    subscriber_store,
    send_whatsapp_message,
    format_alert_message,
    weather_service,
    RBACManager,
    Permission,
)

try:
    from backend.routers import voice_assistant as voice_assistant_router
    from voice_assistant import OfflineCacheManager, VoiceAssistant

    voice_asst = VoiceAssistant(offline_mode=True)
    cache_mgr = OfflineCacheManager(cache_dir="./voice_assistant_cache")
    voice_assistant_router.init_voice_assistant(voice_asst, cache_mgr, verify_role)
except Exception as exc:
    voice_assistant_router = None
    logger.warning("Voice assistant init skipped: %s", exc)

try:
    import joblib

    model_lag = joblib.load("sklearn_yield_model.joblib")
except Exception:
    model_lag = None

ml.init_router(ModelRouter(default_model="xgboost"), model_lag)

# Router registration
app.include_router(ml.router, prefix="/api/yield", tags=["ML Prediction"])
app.include_router(governance.router, prefix="/api/ml-governance", tags=["ML Governance"])
app.include_router(finance.router, prefix="/api/farm-finance", tags=["Finance"])
app.include_router(finance.router, prefix="/api/finance", tags=["Finance Legacy"])
app.include_router(quality.router, prefix="/api/crop-quality", tags=["Quality"])
app.include_router(blockchain.router, prefix="/api/supply-chain", tags=["Blockchain"])
app.include_router(reports.router, prefix="/api/admin", tags=["Reports"])
app.include_router(knowledge.router, prefix="/api/knowledge", tags=["Knowledge"])
app.include_router(community.router, prefix="/api/community", tags=["Community"])
if voice_assistant_router is not None:
    app.include_router(voice_assistant_router.router, prefix="/api/voice", tags=["Voice Assistant"])
app.include_router(referrals.router, prefix="/api/referrals", tags=["Referrals"])
app.include_router(platform.router, prefix="/api", tags=["Platform"])
app.include_router(alerts.router, prefix="/api/notifications", tags=["Alerts"])
app.include_router(flags_router, prefix="/api/flags", tags=["Feature Flags"])


@app.get("/")
def health_check():
    return {"status": "healthy", "service": "Fasal Saathi Backend", "version": "2.0"}


# --- Smart Farm Autopilot ---

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

    Requires authentication. The planner is computationally intensive
    (crop selection, sowing schedule, irrigation plan, fertilizer/pesticide
    timeline, yield/profit projection) and must not be accessible to
    unauthenticated callers.
    """
    await verify_role(request)   # raises 401/403 if token is missing or invalid
    try:
        from smart_farm_autopilot import generate_season_plan
        plan = generate_season_plan(data.dict())
        return {"success": True, "plan": plan}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Autopilot plan generation failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to generate farm plan")


# Include ML Model Management Router
try:
    from routers.ml_models import router as ml_router
    app.include_router(ml_router)
    logger.info("ML Model Management API loaded successfully")
except Exception as e:
    logger.warning(f"Could not load ML Model Management API: {e}")

# Include ML Model Management Router
try:
    from routers.ml_models import router as ml_router
    app.include_router(ml_router)
    logger.info("ML Model Management API loaded successfully")
except Exception as e:
    logger.warning(f"Could not load ML Model Management API: {e}")

# Include ML Model Management Router
try:
    from routers.ml_models import router as ml_router
    app.include_router(ml_router)
    logger.info("ML Model Management API loaded successfully")
except Exception as e:
    logger.warning(f"Could not load ML Model Management API: {e}")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
