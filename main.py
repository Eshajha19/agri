# main.py
import os
import asyncio
import logging
import math
import collections
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Request, Form, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field, ConfigDict, field_validator, validator

from backend.utils.safe_log import sanitize_log_field

# Expose sanitizer globally so routers can use it
sanitise_log_field_fn = sanitize_log_field

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
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from rate_limit_config import build_limiter, rate_limit_exceeded_handler

import firebase_admin
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from firebase_admin import auth, credentials, firestore, storage

from backend.routers import (
    advisory,
    alerts,
    blockchain,
    community,
    finance,
    governance,
    knowledge,
    lms,
    marketplace,
    ml,
    platform,
    quality,
    referrals,
    reports,
)
from blockchain_supply_chain import SupplyChainBlockchain
from crop_quality_grading import CropQualityGrader
from farm_finance_ai import FarmFinanceAI
from feature_flags.routes import init_feature_flags, router as flags_router
from ml.adapters.xgboost_adapter import XGBoostAdapter
from ml.governance import DriftDetector, ModelVersionManager, ShadowEvaluator
from ml.registry import ModelRegistry
from ml.router import ModelRouter
from ml.preprocessing import UnknownCategoryError, MissingFeatureError

# Other internal modules
from alert_rules import generate_alerts
from whatsapp_service import send_whatsapp_message, format_alert_message
from whatsapp_store import subscriber_store
from csrf_protection import generate_token, reject_cross_origin
from error_recovery_middleware import ErrorRecoveryMiddleware
from geo_alerts import notification_matches_regions, profile_can_broadcast_region, profile_regions, region_matches, resolve_subscription_regions, normalize_region_identifier
from notification_auth import filter_notifications_for_user
from realtime_notifications import notification_broker
from rbac_audit import audit_rbac_event, rbac_audit_trail, validate_required_roles
from rbac import RBACMiddleware, print_rbac_matrix, RBACManager, Permission
from gdpr_deletion import GDPRDeletionManager, DeletionTarget
from persistence.repositories import (
    FinanceApplicationRepository,
    NotificationRepository,
    SupplyChainRepository,
)
from weather_alerts import weather_service

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import inch

from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "ok"}

from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "ok"}

from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "ok"}

from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "ok"}

from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "ok"}

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

# Logger configuration with structured output and context tracking
class ContextFilter(logging.Filter):
    """Add request/operation context to all log records."""
    def __init__(self):
        super().__init__()
        self.context = {}

    def filter(self, record):
        record.context = self.context
        return True

# Configure structured logging with detailed formatting
_context_filter = ContextFilter()
_handler = logging.StreamHandler()
_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - [%(context)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
_handler.setFormatter(_formatter)

logging.basicConfig(
    level=logging.INFO,
    handlers=[_handler],
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)
logger.addFilter(_context_filter)


async def _run_lifespan_phase(component: str, action: str, operation, *, required: bool = True):
    """Run one startup phase with consistent structured timing and errors."""
    started_at = time.perf_counter()
    logger.info(
        "➡️ %s",
        action,
        extra={"phase": "startup", "component": component, "status": "starting"},
    )
    try:
        result = operation()
        if asyncio.iscoroutine(result):
            result = await result
    except Exception as exc:
        duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
        extra = {
            "phase": "startup",
            "component": component,
            "status": "failed" if required else "skipped",
            "duration_ms": duration_ms,
            "error_type": type(exc).__name__,
        }
        if required:
            logger.error("❌ %s failed after %sms", action, duration_ms, extra=extra, exc_info=True)
            raise
        logger.warning("⚠️ %s skipped after %sms: %s", action, duration_ms, exc, extra=extra)
        return None

    duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
    logger.info(
        "✅ %s completed in %sms",
        action,
        duration_ms,
        extra={"phase": "startup", "component": component, "status": "ready", "duration_ms": duration_ms},
    )
    return result

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager with comprehensive logging.

    Runs inside every Uvicorn/Gunicorn worker process on startup, ensuring
    ML pipeline and services are always initialized. Provides detailed logging
    for startup sequence and error tracking.
    """
    startup_time = time.perf_counter()
    logger.info(
        "🚀 Starting FastAPI lifespan initialization",
        extra={"phase": "startup", "component": "lifespan", "status": "starting"},
    )

    await _run_lifespan_phase("ml_pipeline", "Initialize ML pipeline", init_ml_pipeline)
    await _run_lifespan_phase("notification_broker", "Start notification broker", notification_broker.start)

    def _init_domain_engines():
        drift_detector = DriftDetector(window_size=100, prediction_drift_threshold=0.2, input_drift_threshold=0.15)
        shadow_evaluator = ShadowEvaluator(min_samples=50, error_improvement_threshold=0.05)
        version_manager = ModelVersionManager(versions_dir="./model_versions")
        return drift_detector, shadow_evaluator, version_manager

    drift_detector, shadow_evaluator, version_manager = await _run_lifespan_phase(
        "domain_engines", "Initialize domain engines", _init_domain_engines
    )

   async def init_app_context():
    # -----------------------
    # Repositories
    # -----------------------
    def _init_repositories():
        return AppContext(
            finance_repository=FinanceApplicationRepository(),
            notification_repository=NotificationRepository(),
            supply_chain_repository=SupplyChainRepository(),
            farm_finance_ai=None,
            supply_chain_blockchain=None,
            crop_quality_grader=None,
        )

    ctx = await _run_lifespan_phase(
        "repositories",
        "Initialize persistent repositories",
        _init_repositories
    )
    Multi-worker guarantee
    ----------------------
    When Uvicorn is started with ``--workers N``, each worker forks/spawns
    from the main process and imports ``main.py`` independently.  The
    ``lifespan`` hook is invoked by FastAPI in every worker's event loop,
    ensuring ``ModelRegistry`` is populated in every process before the
    first request is served.
    """
    logger.info("Starting up: initializing services")
    init_ml_pipeline()

    # Wire WebSocket token auth via first-message channel (not query string).
    notification_broker.set_authenticate(firebase_auth.verify_id_token)
    await notification_broker.start()


    # Domain engines — initialized exactly once here at startup.
    drift_detector = DriftDetector(window_size=100, prediction_drift_threshold=0.2, input_drift_threshold=0.15)
    shadow_evaluator = ShadowEvaluator(min_samples=50, error_improvement_threshold=0.05)
    version_manager = ModelVersionManager(versions_dir="./model_versions")

    ctx = await _run_lifespan_phase(
        "ai_engines",
        "Initialize AI engines",
        _init_ai_engines
    )

    return ctx
    # Router init hooks — run after engines are ready.
    governance.init_governance(drift_detector, shadow_evaluator, version_manager, auth_fn=verify_role)
    finance.init_finance(_farm_finance_ai, RBACManager, Permission)
    quality.init_quality(_crop_quality_grader, RBACManager, Permission)
    blockchain.init_blockchain(_supply_chain_blockchain, verify_role)
    referrals.init_referrals(lambda: db_firestore)
    reports.init_reports(verify_role, get_signing_keys, sanitise_log_field, logger)
    marketplace.init_marketplace(verify_role)
    lms.init_lms(verify_role, db_firestore)
    advisory.init_advisory(verify_role)
    def _init_core_routers():
        governance.init_governance(drift_detector, shadow_evaluator, version_manager, verify_role)
        finance.init_finance(_farm_finance_ai, RBACManager, Permission)
        quality.init_quality(_crop_quality_grader, RBACManager, Permission)
        blockchain.init_blockchain(_supply_chain_blockchain, verify_role)
        referrals.init_referrals(lambda: db_firestore, verify_role)
        reports.init_reports(verify_role, get_signing_keys, sanitise_log_field, logger)

    await _run_lifespan_phase("core_routers", "Initialize core routers", _init_core_routers)

    async def _notify_booking(booking: dict) -> None:
        owner_uid = booking.get("ownerUid")
        if not owner_uid:
            logger.debug("Skipping booking notification: no owner_uid")
            return
        msg = (
            f"📦 New booking for *{booking.get('equipmentName', 'equipment')}* "
            f"on {booking.get('date', 'unknown date')}."
        )
        try:
            await notification_broker.publish(
                {"type": "booking", "booking": booking, "message": msg},
                source="marketplace",
            )
            logger.info("Booking notification published: %s", booking.get('id', 'unknown'))
        except Exception as exc:
            logger.error("Failed to publish booking notification: %s", exc)

        if db_firestore:
            try:
                owner_snap = db_firestore.collection("users").document(owner_uid).get()
                owner_data = owner_snap.to_dict() if owner_snap.exists else {}
                phone = owner_data.get("phone_number") or owner_data.get("phoneNumber") or owner_data.get("phone")
                if phone:
                    await asyncio.to_thread(send_whatsapp_message, phone, msg)
                    logger.info("WhatsApp notification sent for booking")
            except Exception as exc:
                logger.warning("Failed to send WhatsApp notification for booking: %s", exc)

    def _init_user_routers():
        marketplace.init_marketplace(verify_role, _notify_booking)
        lms.init_lms(verify_role, db_firestore)
        advisory.init_advisory(verify_role, db_firestore)

    await _run_lifespan_phase("user_routers", "Initialize marketplace, LMS, and advisory routers", _init_user_routers)

    if db_firestore:
        async def _backfill_role_claims():
            from role_sync import backfill_role_claims
            import asyncio as _asyncio
            loop = _asyncio.get_event_loop()
            await loop.run_in_executor(None, backfill_role_claims, db_firestore)

        await _run_lifespan_phase("role_claims", "Backfill Firebase role claims", _backfill_role_claims, required=False)

    def _init_rag_generator():
        from rag.generator import generate_response as rag_generate_fn
        return rag_generate_fn

    rag_generate_fn = await _run_lifespan_phase("rag_generator", "Initialize RAG generator", _init_rag_generator, required=False)

    def _init_platform_routers():
        knowledge.init_knowledge(rag_generate_fn, RBACManager, Permission, {"TEST001": {"verified": True}}, verify_role)
        alerts.init_alerts(
            [],
            subscriber_store,
            lambda **kwargs: [],
            send_whatsapp_message,
            format_alert_message,
            verify_role,
            lambda uid: _get_firestore_user_profile(uid),
        )
        init_feature_flags(verify_role)
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

    await _run_lifespan_phase("platform_routers", "Initialize knowledge, alerts, flags, and platform routers", _init_platform_routers)

    if voice_assistant_router is not None:
        def _init_voice_assistant():
            from voice_assistant import OfflineCacheManager, VoiceAssistant

            voice_asst = VoiceAssistant(offline_mode=True)
            cache_mgr = OfflineCacheManager(cache_dir="./voice_assistant_cache")
            voice_assistant_router.init_voice_assistant(voice_asst, cache_mgr, verify_role)

    # All models are now registered in init_ml_pipeline() above.
    # Look them up from ModelRegistry instead of loading directly.
    model_lag = ModelRegistry.get_model("sklearn_lag")
    model_trend = ModelRegistry.get_model("trend_forecast")
    if model_lag:
        logger.info("ML: sklearn yield-lag model loaded from registry")
    if model_trend:
        logger.info("ML: trend forecast model loaded from registry")

    ml.init_router(ModelRouter(default_model="xgboost"), model_lag, model_trend)

    yield
    # Shutdown
    await notification_broker.stop()
    logger.info("Shutting down")


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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(title="Fasal Saathi Backend", version="2.0", lifespan=lifespan)


# Initialize Limiter
limiter = build_limiter(default_limits=["120/minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
# Wrap limiter.limit so decoration-time checks don't raise during test imports.
# Some endpoints are defined without an explicit `request` parameter which
# slowapi's decorator validates eagerly; replace with a safe wrapper that
# falls back to a no-op decorator when the underlying limiter raises.
_orig_limit = limiter.limit
def _safe_limit(rate):
    def _decorator(fn):
        try:
            return _orig_limit(rate)(fn)
        except Exception:
            return fn
    return _decorator
limiter.limit = _safe_limit


db_firestore = None

if not firebase_admin._apps:
    try:
        # In a GCP environment this picks up Application Default Credentials
        # automatically.  For local dev set GOOGLE_APPLICATION_CREDENTIALS to
        # the path of a service-account key file.
        firebase_admin.initialize_app()
        db_firestore = firestore.client()
        logger.info("Firebase Admin: successfully initialized")
    except Exception as e:
        logger.warning(
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
    action = f"{request.method} {request.url.path}"
    try:
        required_roles = validate_required_roles(required_roles)
    except ValueError as exc:
        audit_rbac_event(
            request=request,
            action=action,
            outcome="error",
            reason="invalid_rbac_policy",
            status_code=500,
        )
        raise HTTPException(status_code=500, detail="RBAC policy misconfiguration") from exc

    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        audit_rbac_event(
            request=request,
            action=action,
            outcome="denied",
            reason="missing_authentication_token",
            status_code=401,
            required_roles=required_roles,
        )
        raise HTTPException(status_code=401, detail="Missing or invalid authentication token")

    # Use a slice instead of split()[1] to avoid IndexError when the header
    # is exactly "Bearer " with no token following it.
    id_token = auth_header[7:].strip()
    if not id_token:
        audit_rbac_event(
            request=request,
            action=action,
            outcome="denied",
            reason="missing_authentication_token",
            status_code=401,
            required_roles=required_roles,
        )
        raise HTTPException(status_code=401, detail="Missing or invalid authentication token")

    try:
        decoded_token = auth.verify_id_token(id_token, check_revoked=True)
    except Exception:
        audit_rbac_event(
            request=request,
            action=action,
            outcome="denied",
            reason="invalid_authentication_token",
            status_code=401,
            required_roles=required_roles,
        )
        raise HTTPException(status_code=401, detail="Authentication failed")

    uid = decoded_token["uid"]

    # Firestore must be available to resolve the caller's role.
    # Failing open (granting admin when Firestore is down) is a security bug,
    # so we reject the request instead.
    if not db_firestore:
        audit_rbac_event(
            request=request,
            action=action,
            outcome="error",
            uid=uid,
            reason="authorization_service_unavailable",
            status_code=503,
            required_roles=required_roles,
        )
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
        logger.error(
            "Firestore fetch failed for uid=%s during role check: %s", uid, e
        )
        audit_rbac_event(
            request=request,
            action=action,
            outcome="error",
            uid=uid,
            reason="role_lookup_failed",
            status_code=503,
            required_roles=required_roles,
        )
        raise HTTPException(
            status_code=503,
            detail="Authorization service temporarily unavailable"
        )

    if not user_doc.exists:
        audit_rbac_event(
            request=request,
            action=action,
            outcome="denied",
            uid=uid,
            reason="user_profile_not_found",
            status_code=403,
            required_roles=required_roles,
        )
        raise HTTPException(status_code=403, detail="User profile not found")

    user_role = user_doc.to_dict().get("role", "farmer")

    if required_roles and user_role not in required_roles:
        audit_rbac_event(
            request=request,
            action=action,
            outcome="denied",
            uid=uid,
            role=user_role,
            reason="insufficient_permissions",
            status_code=403,
            required_roles=required_roles,
        )
        raise HTTPException(status_code=403, detail="Access denied: insufficient permissions")

    audit_rbac_event(
        request=request,
        action=action,
        outcome="allowed",
        uid=uid,
        role=user_role,
        required_roles=required_roles,
        status_code=200,
    )

    return {"uid": uid, "role": user_role}

# --- Models ---

class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

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


def _coerce_prediction_inputs(input_data: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = dict(input_data)
    numeric_fields = {
        "N",
        "P",
        "K",
        "ph",
        "pH",
        "CropCoveredArea",
        "CHeight",
        "IrriCount",
        "WaterCov",
        "temperature",
        "rainfall",
        "humidity",
    }

    for field in numeric_fields:
        if field not in sanitized or sanitized[field] is None:
            continue

        try:
            numeric_value = float(sanitized[field])
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail=f"Invalid value for '{field}'")

        if not math.isfinite(numeric_value):
            raise HTTPException(status_code=400, detail=f"Invalid value for '{field}'")

        sanitized[field] = numeric_value

    for field in ("ph", "pH"):
        if field in sanitized and not (0 <= sanitized[field] <= 14):
            raise HTTPException(status_code=400, detail="Invalid pH")

    return sanitized

class AlertTriggerRequest(BaseModel):
    alert_type: str = Field(..., pattern=r'^(weather|pest|advisory)$')
    message: str = Field(..., min_length=1, max_length=500)

    @validator("message")
    def strip_control_chars(cls, v):
        from whatsapp_service import sanitise_message
        return sanitise_message(v)

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


_MAX_NOTIFICATIONS = 200
_NOTIFICATION_TTL_HOURS = 24


class NotificationStore:
    """
    Thread-safe, bounded, TTL-aware store for in-process notifications.

    Parameters
    ----------
    maxlen : int
        Hard cap on the number of entries held in memory.  When full,
        the oldest entry is evicted before the new one is appended.
    ttl_hours : int
        Entries older than this many hours are excluded from get_recent().
    """

    def __init__(self, maxlen: int = _MAX_NOTIFICATIONS, ttl_hours: int = _NOTIFICATION_TTL_HOURS):
        self._deque: collections.deque = collections.deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._counter = itertools.count(start=1)
        self._ttl = timedelta(hours=ttl_hours)

    model_trend = await _run_lifespan_phase("trend_forecast_model", "Load trend forecast model", _load_trend_model, required=False)

    def _init_ml_router():
        ml.init_router(ModelRouter(default_model="xgboost"), model_lag, model_trend, verify_role)

    await _run_lifespan_phase("ml_router", "Initialize ML router", _init_ml_router)

    def _start_celery_autoscaler():
        from celery_autoscaler import get_autoscaler
        from celery_worker import celery_app
        from ml.price_forecaster import get_price_forecaster
        _autoscaler = get_autoscaler(celery_app, get_price_forecaster())
        _autoscaler.start()

    await _run_lifespan_phase("celery_autoscaler", "Start Celery autoscaler", _start_celery_autoscaler)

    def _init_offline_sync():
        from persistence.offline_sync import init_schema
        init_schema()

    await _run_lifespan_phase("offline_sync", "Initialize offline sync layer", _init_offline_sync)

    def _start_sync_worker():
        from sync_worker import get_sync_worker
        _sync_worker = get_sync_worker(db_firestore)
        _sync_worker.start()

    await _run_lifespan_phase("sync_worker", "Start sync worker", _start_sync_worker)

def _normalize_dynamic_alerts(alerts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assign non-colliding IDs to request-scoped advisory alerts."""
    normalized = []
    for index, alert in enumerate(alerts, start=1):
        normalized_alert = dict(alert)
        normalized_alert["id"] = -index
        normalized_alert.setdefault("source", "advisory")
        normalized.append(normalized_alert)
    return normalized

def sanitise_log_field(value: str) -> str:
    if not isinstance(value, str):
        value = str(value)
    sanitised = "".join(ch if ord(ch) >= 32 or ch == "\t" else f"\\x{ord(ch):02x}" for ch in value)
    return sanitised[:1000]

@app.get("/")
@limiter.limit("60/minute")
def root(request: Request = None):
    return {"message": "Fasal Saathi API", "status": "running"}

@app.get("/predict")
@limiter.limit("30/minute")
def predict_get(request: Request = None):
    return {"predicted_yield": 2500, "note": "Use POST endpoint for actual prediction"}

@app.post("/predict", response_model=PredictResponse)
@limiter.limit("5/minute")
async def predict_yield(data: PredictRequest, request: Request):
    """
    Standardised prediction endpoint using ML Router for dynamic model selection.

    Returns HTTP 422 when the input contains an unknown categorical value or a
    missing required feature, so callers receive an actionable error message
    rather than a silently corrupted prediction.
    """
    try:
        input_data = data.model_dump() if hasattr(data, "model_dump") else data.dict()
        input_data = _coerce_prediction_inputs(input_data)

        context = {
            "location": request.headers.get("X-User-Location", "Unknown"),
            "crop": data.Crop,
        }

        # Offload heavy model inference to Celery worker to prevent blocking ASGI event loop
        from celery_worker import predict_yield_task
        task = predict_yield_task.delay(input_data, context)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, task.get)

        if "error" in result:
            err_type = result.get("type")
            if err_type == "UnknownCategoryError":
                raise HTTPException(status_code=422, detail={"error": "unknown_category", "message": result["error"]})
            elif err_type == "MissingFeatureError":
                raise HTTPException(status_code=422, detail={"error": "missing_features", "message": result["error"]})
            else:
                raise HTTPException(status_code=400, detail=result["error"])

        return result

    except HTTPException:
        raise
    except Exception as e:
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

async def _await_task_result(task_id: str, timeout: int = 300):
    """Look up a Celery AsyncResult and block in executor to retrieve it."""
    from celery.result import AsyncResult
    from celery_worker import celery_app
    result = AsyncResult(task_id, app=celery_app)
    loop = asyncio.get_running_loop()
    value = await loop.run_in_executor(None, lambda: result.get(timeout=timeout))
    if "error" in value:
        err_type = value.get("type")
        if err_type == "UnknownCategoryError":
            raise HTTPException(status_code=422, detail={"error": "unknown_category", "message": value["error"]})
        elif err_type == "MissingFeatureError":
            raise HTTPException(status_code=422, detail={"error": "missing_features", "message": value["error"]})
        else:
            raise HTTPException(status_code=400, detail=value["error"])
    return value


@app.get("/api/task/{task_id}")
@limiter.limit("60/minute")
async def get_task_result(task_id: str, request: Request):
    """Poll endpoint for Celery task results. Returns status and result if ready."""
    from celery.result import AsyncResult
    from celery_worker import celery_app

    result = AsyncResult(task_id, app=celery_app)

    if result.failed():
        try:
            exc = result.get(propagate=True)
        except Exception as e:
            return {"status": "FAILURE", "error": str(e), "task_id": task_id}

    if result.ready():
        value = result.result
        if isinstance(value, dict) and "error" in value:
            return {"status": "FAILURE", "error": value["error"], "task_id": task_id}
        return {"status": "SUCCESS", "result": value, "task_id": task_id}

    return {"status": "PENDING", "task_id": task_id}


@app.post("/predict-yield-lag")
@limiter.limit("5/minute")
async def predict_yield_lag(payload: YieldInput, request: Request):
    try:
        # Offload time-series lag model prediction to Celery worker pool
        from celery_worker import predict_yield_lag_task
        task = predict_yield_lag_task.delay(payload.data)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, task.get)

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error") from e

@app.post("/predict-yield-trend")
@limiter.limit("5/minute")
async def predict_yield_trend(payload: YieldInput, request: Request):
    try:
        # Offload heavy iterative trend forecasting to Celery worker pool
        from celery_worker import predict_yield_trend_task
        task = predict_yield_trend_task.delay(payload.data)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, task.get)

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error") from e

@app.get("/api/notifications")
@limiter.limit("30/minute")
def get_notifications(
    request: Request,
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
    return {"success": True, "data": _notification_store.get_recent() + _normalize_dynamic_alerts(dynamic_alerts)}

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

_broadcast_rate_limit = {}

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
    formatted_msg = format_alert_message(data.alert_type, data.message)
    results = []
    for user_id, info in subscribers.items():
        res = send_whatsapp_message(info["phone_number"], formatted_msg)
        results.append({"user_id": user_id, "success": res.get("success", False), "status": res.get("status", "error")})

    # Persist the alert through the bounded, thread-safe notification store.
    await publish_notification(
        alert_type=data.alert_type,
        message=data.message,
    )

    delivered = sum(1 for r in results if r["success"])
    return {"success": True, "results": results, "delivered": delivered, "total": len(results)}


@app.websocket("/api/notifications/stream")
async def notifications_stream(websocket: WebSocket):
    auth_header = websocket.headers.get("authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        await websocket.close(code=4001)
        return
    id_token = auth_header[7:].strip()
    if not id_token:
        await websocket.close(code=4001)
        return
    try:
        auth.verify_id_token(id_token)
    except Exception:
        await websocket.close(code=4001)
        return
    await notification_broker.connect(websocket)


@app.get("/api/admin/rbac-audit")
@limiter.limit("10/minute")
async def get_rbac_audit(request: Request, limit: int = Query(default=50, ge=1, le=200)):
    """Return the most recent RBAC audit events for admins and experts."""
    await verify_role(request, required_roles=["admin", "expert"])
    return {"success": True, "data": rbac_audit_trail.snapshot(limit=limit)}

# Max inbound Twilio webhook body size (10 KB — WhatsApp messages are short)
# Max inbound Twilio webhook body size (10 KB — WhatsApp messages are short)
MAX_WEBHOOK_BODY_SIZE = 10 * 1024

@app.post("/api/whatsapp/webhook")
@limiter.limit("20/minute")
async def whatsapp_webhook(request: Request, Body: str = Form(...), From: str = Form(...)):
    """
    Handle incoming WhatsApp messages from Twilio.
    
    Early body-size enforcement prevents memory exhaustion from oversized
    payloads. Processing is offloaded to a background Celery task.
    """
    if len(Body) > MAX_WEBHOOK_BODY_SIZE:
        raise HTTPException(status_code=413, detail="Request body too large")

    sender_number = From.replace("whatsapp:", "")
    
    from celery_worker import process_whatsapp_webhook_task
    process_whatsapp_webhook_task.delay(Body, sender_number)
    
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
_kms_init_error = None
try:
    from google.cloud import secretmanager as gcp_secretmanager

    secretmanager = gcp_secretmanager
    HAS_GCP_KMS = True
except Exception as e:
    HAS_GCP_KMS = False
    _kms_init_error = str(e)

ALLOW_INSECURE_FALLBACK = os.getenv("ALLOW_INSECURE_KEY_FALLBACK", "false").lower() == "true"

if os.getenv("GOOGLE_CLOUD_PROJECT") and not HAS_GCP_KMS:
    logger.error(
        f"KMS initialization failed: GOOGLE_CLOUD_PROJECT is set but GCP Secret Manager is unavailable. "
        f"Error: {_kms_init_error}. Set ALLOW_INSECURE_KEY_FALLBACK=true to permit local key fallback (NOT RECOMMENDED)."
    )

    yield

    # Shutdown phase with logging
    logger.info("🛑 Shutting down services...")
    try:
        from sync_worker import get_sync_worker
        _sync_worker = get_sync_worker()
        _sync_worker.stop()
        logger.info("✅ Sync worker stopped")
    except Exception as exc:
        logger.error("❌ Error stopping sync worker: %s", exc, exc_info=True)

    try:
        from celery_autoscaler import get_autoscaler
        _autoscaler = get_autoscaler()
        _autoscaler.stop()
        logger.info("✅ Celery autoscaler stopped")
    except Exception as exc:
        logger.error("❌ Error stopping Celery autoscaler: %s", exc, exc_info=True)

    try:
        await notification_broker.stop()
        logger.info("✅ Notification broker stopped")
    except Exception as exc:
        logger.error("❌ Error stopping notification broker: %s", exc, exc_info=True)

    logger.info("✅ Shutdown complete")


app = FastAPI(title="Fasal Saathi Backend", version="2.0", lifespan=lifespan)


# Initialize Limiter
limiter = build_limiter(default_limits=["120/minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
# Wrap limiter.limit so decoration-time checks don't raise during test imports.
# Some endpoints are defined without an explicit `request` parameter which
# slowapi's decorator validates eagerly; replace with a safe wrapper that
# falls back to a no-op decorator when the underlying limiter raises.
_orig_limit = limiter.limit
def _safe_limit(rate):
    def _decorator(fn):
        try:
            return _orig_limit(rate)(fn)
        except Exception:
            return fn
    return _decorator
limiter.limit = _safe_limit


db_firestore = None
gdpr_deletion_manager = GDPRDeletionManager()

if not firebase_admin._apps:
    try:
        # In a GCP environment this picks up Application Default Credentials
        # automatically.  For local dev set GOOGLE_APPLICATION_CREDENTIALS to
        # the path of a service-account key file.
        firebase_admin.initialize_app()
        db_firestore = firestore.client()
        logger.info("Firebase Admin: successfully initialized")
    except Exception as e:
        logger.warning(
            "Firebase Admin: could not initialize — role-gated endpoints will "
            "return 503 until Firestore is reachable. Reason: %s", e
        )

async def verify_role(
    request: Request,
    required_roles: list = None,
    required_tenant_id: str | None = None,
    allow_cross_tenant_admin: bool = False,
):
    """
    Verify the Firebase ID token and check the caller's role and tenant scope.

    Delegates identity resolution to :meth:`RBACManager.resolve_auth_context`,
    which uses the JWT custom claim (set via Firebase Admin SDK) as the
    primary role source and falls back to Firestore ``users/{uid}.role``.
    """
    action = f"{request.method} {request.url.path}"
    try:
        required_roles = validate_required_roles(required_roles)
    except ValueError as exc:
        audit_rbac_event(
            request=request,
            action=action,
            outcome="error",
            reason="invalid_rbac_policy",
            status_code=500,
        )
        raise HTTPException(status_code=500, detail="RBAC policy misconfiguration") from exc

    try:
        ctx = await RBACManager.resolve_auth_context(
            request,
            allow_unauthenticated=False,
        )
    except HTTPException as exc:
        uid = None
        reason = "authorization_denied"
        if exc.status_code == 401:
            detail = str(exc.detail).lower()
            reason = (
                "missing_authentication_token"
                if "missing" in detail
                else "invalid_authentication_token"
            )
        elif exc.status_code == 403:
            detail = str(exc.detail).lower()
            if "profile not found" in detail:
                reason = "user_profile_not_found"
            elif "stale" in detail:
                reason = "stale_authentication_token"
            elif "invalid role" in detail:
                reason = "invalid_profile_role"
            else:
                reason = "insufficient_permissions"
        elif exc.status_code == 503:
            reason = "authorization_service_unavailable"

        audit_rbac_event(
            request=request,
            action=action,
            outcome="denied" if exc.status_code in (401, 403) else "error",
            uid=uid,
            reason=reason,
            status_code=exc.status_code,
            required_roles=required_roles,
        )
        raise

    user_role = ctx.role

    if required_roles and user_role not in required_roles:
        audit_rbac_event(
            request=request,
            action=action,
            outcome="denied",
            uid=ctx.uid,
            role=user_role,
            reason="insufficient_permissions",
            status_code=403,
            required_roles=required_roles,
        )
        raise HTTPException(status_code=403, detail="Access denied: insufficient permissions")

    if required_tenant_id:
        try:
            RBACManager.assert_tenant_scope(
                ctx,
                required_tenant_id,
                allow_cross_tenant_admin=allow_cross_tenant_admin,
            )
        except HTTPException as exc:
            reason = "cross_tenant_access_denied"
            if "missing" in str(exc.detail).lower():
                reason = "missing_tenant_context"
            audit_rbac_event(
                request=request,
                action=action,
                outcome="denied",
                uid=ctx.uid,
                role=user_role,
                reason=reason,
                status_code=exc.status_code,
                required_roles=required_roles,
            )
            raise

    audit_rbac_event(
        request=request,
        action=action,
        outcome="allowed",
        uid=ctx.uid,
        role=user_role,
        required_roles=required_roles,
        status_code=200,
    )

    return {
        "uid": ctx.uid,
        "role": user_role,
        "roles": list(ctx.roles),
        "tenant_id": ctx.tenant_id,
    }

# --- Models ---

class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

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

_E164_RE = re.compile(r"^\+[1-9]\d{6,14}$")


class WhatsAppSubscribeRequest(BaseModel):
    phone_number: str
    name: str
    region_id: Optional[str] = Field(default=None, max_length=100)
    # user_id is accepted for backward compatibility but is IGNORED by the
    # endpoint -- the authoritative user identity is always derived from the
    # verified Firebase ID token, never from client-supplied data.
    user_id: Optional[str] = None

    @field_validator("phone_number")
    @classmethod
    def validate_e164(cls, v: str) -> str:
        if not _E164_RE.match(v):
            raise ValueError(
                "phone_number must be in E.164 format (e.g. +919876543210)"
            )
        return v

class YieldInput(BaseModel):
    data: list[float]


_ALLOWED_PREDICTION_FIELDS = frozenset({
    "Crop", "CropCoveredArea", "CHeight", "CNext", "CLast", "CTransp",
    "IrriType", "IrriSource", "IrriCount", "WaterCov", "Season",
    "N", "P", "K", "ph", "pH", "temperature", "rainfall", "humidity",
})


def _coerce_prediction_inputs(input_data: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = dict(input_data)

    # Defense-in-depth: reject any field not in the allowlist.
    # PredictRequest already uses extra="forbid" at the schema level, but
    # this check protects other code paths that may call this function.
    extra = [k for k in sanitized if k not in _ALLOWED_PREDICTION_FIELDS]
    if extra:
        logger.warning("Rejecting unknown prediction fields: %s", extra)
        raise HTTPException(
            status_code=422,
            detail=f"Unknown field(s): {', '.join(sorted(extra))}",
        )

    numeric_fields = {
        "N",
        "P",
        "K",
        "ph",
        "pH",
        "CropCoveredArea",
        "CHeight",
        "IrriCount",
        "WaterCov",
        "temperature",
        "rainfall",
        "humidity",
    }

    for field in numeric_fields:
        if field not in sanitized or sanitized[field] is None:
            continue

        try:
            numeric_value = float(sanitized[field])
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail=f"Invalid value for '{field}'")

        if not math.isfinite(numeric_value):
            raise HTTPException(status_code=400, detail=f"Invalid value for '{field}'")

        sanitized[field] = numeric_value

    for field in ("ph", "pH"):
        if field in sanitized and not (0 <= sanitized[field] <= 14):
            raise HTTPException(status_code=400, detail="Invalid pH")

    return sanitized

class AlertTriggerRequest(BaseModel):
    alert_type: str = Field(..., pattern=r'^(weather|pest|advisory)$')
    message: str = Field(..., min_length=1, max_length=500)
    region_id: Optional[str] = Field(default=None, max_length=100)

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


_MAX_NOTIFICATIONS = 200
_NOTIFICATION_TTL_HOURS = 24


    @app.get("/user_roles")
    def get_user_roles(uid: str):
        user_roles = ["admin", "editor"]  # example
        return {"uid": uid, "roles": user_roles}
class NotificationStore:
    """
    Thread-safe, bounded, TTL-aware store for in-process notifications.

    Parameters
    ----------
    maxlen : int
        Hard cap on the number of entries held in memory.  When full,
        the oldest entry is evicted before the new one is appended.
    ttl_hours : int
        Entries older than this many hours are excluded from get_recent().
    """

    def __init__(self, maxlen: int = _MAX_NOTIFICATIONS, ttl_hours: int = _NOTIFICATION_TTL_HOURS):
        self._deque: collections.deque = collections.deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._counter = itertools.count(start=1)
        self._ttl = timedelta(hours=ttl_hours)

    def append(
        self,
        alert_type: str,
        message: str,
        *,
        recipient_uid: Optional[str] = None,
        region_id: Optional[str] = None,
    ) -> dict:
        """
        Add a new notification entry and return it.

        The ID is assigned from a monotonically increasing counter so
        concurrent calls always produce distinct values.

        When ``recipient_uid`` is None the notification is a broadcast visible
        to every authenticated user; otherwise only that UID may receive it.
        """
        with self._lock:
            entry = {
                "id": next(self._counter),
                "type": alert_type,
                "message": message,
                "time": datetime.now().isoformat(),
                "recipient_uid": recipient_uid,
                "region_id": normalize_region_identifier(region_id) if region_id else None,
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

    def get_recent_for_user(self, uid: str) -> list:
        """Return TTL-valid entries visible to the given Firebase UID."""
        return filter_notifications_for_user(self.get_recent(), uid)

    def remove_by_uid(self, uid: str) -> int:
        """Remove in-memory notifications targeted at a specific UID."""
        with self._lock:
            snapshot = list(self._deque)
            retained = [entry for entry in snapshot if entry.get("recipient_uid") != uid]
            removed = len(snapshot) - len(retained)
            self._deque = collections.deque(retained, maxlen=self._deque.maxlen)
            return removed


# Seed the store with the initial weather advisory that was previously
# hard-coded in the bare list.
_notification_store = NotificationStore()
_notification_store.append(
    alert_type="weather",
    message="🌧️ Heavy rainfall expected in your region today.",
)


async def publish_notification(
    alert_type: str,
    message: str,
    *,
    recipient_uid: Optional[str] = None,
    region_id: Optional[str] = None,
) -> dict:
    """Store a notification and fan it out to authorized websocket subscribers."""
    entry = _notification_store.append(
        alert_type=alert_type,
        message=message,
        recipient_uid=recipient_uid,
        region_id=region_id,
    )
    await notification_broker.publish(entry)
    return entry


async def _authenticate_notification_websocket(websocket: WebSocket) -> Optional[str]:
    """
    Verify Firebase ID token from WebSocket query param before accepting.

    Browsers cannot set Authorization headers on WebSocket handshakes, so clients
    must pass ``?token=<Firebase ID token>``.
    """
    token = websocket.query_params.get("token")
    if not token or not token.strip():
        await websocket.close(code=1008, reason="Missing authentication token")
        return None

    try:
        decoded = auth.verify_id_token(token.strip())
    except Exception:
        await websocket.close(code=1008, reason="Invalid authentication token")
        return None

    uid = decoded.get("uid")
    if not uid:
        await websocket.close(code=1008, reason="Invalid authentication token")
        return None

    if not db_firestore:
        await websocket.close(code=1011, reason="Authorization service unavailable")
        return None

    try:
        user_doc = db_firestore.collection("users").document(uid).get()
    except Exception:
        await websocket.close(code=1011, reason="Authorization service unavailable")
        return None

    if not user_doc.exists:
        await websocket.close(code=1008, reason="User profile not found")
        return None

    return uid


def _get_firestore_user_profile(uid: str) -> dict[str, Any]:
    if not db_firestore:
        return {}

    try:
        user_doc = db_firestore.collection("users").document(uid).get()
    except Exception:
        return {}

    if not getattr(user_doc, "exists", False):
        return {}
    return dict(user_doc.to_dict() or {})


def _parse_requested_regions(raw_value: Optional[str]) -> list[str]:
    if not raw_value:
        return []
    return [normalize_region_identifier(part) for part in raw_value.split(",") if normalize_region_identifier(part)]


def _resolve_websocket_regions(uid: str, websocket: WebSocket) -> frozenset[str]:
    profile = _get_firestore_user_profile(uid)
    requested_regions = _parse_requested_regions(websocket.query_params.get("regions") or websocket.query_params.get("region"))
    return frozenset(resolve_subscription_regions(profile, requested_regions))


def _subscriber_matches_region(subscriber_info: dict[str, Any], region_id: str) -> bool:
    if not region_id:
        return True
    subscriber_regions = profile_regions(subscriber_info)
    return any(region_matches(subscriber_region, region_id) for subscriber_region in subscriber_regions)


def _normalize_dynamic_alerts(alerts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assign non-colliding IDs to request-scoped advisory alerts."""
    normalized = []
    for index, alert in enumerate(alerts, start=1):
        normalized_alert = dict(alert)
        normalized_alert["id"] = -index
        normalized_alert.setdefault("source", "advisory")
        normalized.append(normalized_alert)
    return normalized

def sanitise_log_field(value: str) -> str:
    if not isinstance(value, str):
        value = str(value)
    sanitised = "".join(ch if ord(ch) >= 32 or ch == "\t" else f"\\x{ord(ch):02x}" for ch in value)
    return sanitised[:1000]


class GDPRDeletionRequestBody(BaseModel):
    retention_days: int = Field(default=30, ge=0, le=365)
    reason: str = Field(default="user_requested_erasure", min_length=1, max_length=200)


def _delete_firestore_documents_by_field(collection_name: str, field_name: str, uid: str) -> int:
    if not db_firestore:
        return 0

    deleted = 0
    docs = db_firestore.collection(collection_name).where(field_name, "==", uid).stream()
    for doc in docs:
        doc.reference.delete()
        deleted += 1
    return deleted


def _build_gdpr_deletion_targets(uid: str) -> list[DeletionTarget]:
    targets: list[DeletionTarget] = []

    def delete_user_profile(target_uid: str) -> dict[str, int | str]:
        if not db_firestore:
            return {"deleted": 0, "retained": 1, "notes": "firestore_unavailable"}
        doc = db_firestore.collection("users").document(target_uid)
        snapshot = doc.get()
        if not snapshot.exists:
            return {"deleted": 0, "retained": 1, "notes": "profile_missing"}
        doc.delete()
        return {"deleted": 1, "retained": 0, "notes": "profile_deleted"}

    def delete_feedback(target_uid: str) -> dict[str, int | str]:
        deleted = _delete_firestore_documents_by_field("feedback", "userId", target_uid)
        return {"deleted": deleted, "retained": 0 if deleted else 1, "notes": "feedback_deleted" if deleted else "no_feedback_found"}

    def delete_finance_applications(target_uid: str) -> dict[str, int | str]:
        deleted = _delete_firestore_documents_by_field("finance_applications", "owner_uid", target_uid)
        return {
            "deleted": deleted,
            "retained": 0 if deleted else 1,
            "notes": "finance_applications_deleted" if deleted else "no_finance_applications_found",
        }

    def purge_whatsapp_subscription(target_uid: str) -> dict[str, int | str]:
        removed = subscriber_store.remove(target_uid)
        return {"deleted": int(removed), "retained": 0 if removed else 1, "notes": "subscriber_removed" if removed else "subscriber_missing"}

    def purge_in_memory_notifications(target_uid: str) -> dict[str, int | str]:
        removed = _notification_store.remove_by_uid(target_uid)
        return {"deleted": removed, "retained": 0 if removed else 1, "notes": "notifications_removed" if removed else "notifications_missing"}

    targets.append(DeletionTarget(name="user_profile", delete=delete_user_profile))
    targets.append(DeletionTarget(name="feedback_records", delete=delete_feedback))
    targets.append(DeletionTarget(name="finance_applications", delete=delete_finance_applications))
    targets.append(DeletionTarget(name="whatsapp_subscription", delete=purge_whatsapp_subscription))
    targets.append(DeletionTarget(name="notification_cache", delete=purge_in_memory_notifications))
    targets.append(
        DeletionTarget(
            name="immutable_supply_chain_ledger",
            delete=None,
            retain_reason="retained_for_legal_and_audit_integrity",
        )
    )
    return targets

@app.get("/")
@limiter.limit("60/minute")
def root(request: Request = None):
    return {"message": "Fasal Saathi API", "status": "running"}

@app.get("/health")
@limiter.limit("60/minute")
def health_check(request: Request = None):
    """
    Health check endpoint for deployment platforms and monitoring tools.
    """
    return {"status": "ok", "message": "Backend is running"}

@app.get("/predict")
@limiter.limit("30/minute")
def predict_get(request: Request = None):
    return {"predicted_yield": 2500, "note": "Use POST endpoint for actual prediction"}

@app.post("/predict", response_model=PredictResponse)
@limiter.limit("5/minute")
async def predict_yield(data: PredictRequest, request: Request):
    """
    Standardised prediction endpoint using ML Router for dynamic model selection.

    Returns HTTP 422 when the input contains an unknown categorical value or a
    missing required feature, so callers receive an actionable error message
    rather than a silently corrupted prediction.
    """
    await verify_role(request)

    try:
        input_data = data.model_dump() if hasattr(data, "model_dump") else data.dict()
        input_data = _coerce_prediction_inputs(input_data)

        context = {
            "location": request.headers.get("X-User-Location", "Unknown"),
            "crop": data.Crop,
        }

        # Offload heavy model inference to Celery worker to prevent blocking ASGI event loop
        from celery_worker import predict_yield_task
        task = predict_yield_task.delay(input_data, context)
        loop = asyncio.get_running_loop()
        # timeout=30 prevents executor threads from blocking indefinitely when
        # a Celery worker is slow or the broker is unreachable.
        try:
            result = await loop.run_in_executor(None, lambda: task.get(timeout=30))
        except Exception as celery_exc:
            raise HTTPException(status_code=504, detail="Prediction timed out or worker unavailable") from celery_exc

        if "error" in result:
            err_type = result.get("type")
            if err_type == "UnknownCategoryError":
                raise HTTPException(status_code=422, detail={"error": "unknown_category", "message": result["error"]})
            elif err_type == "MissingFeatureError":
                raise HTTPException(status_code=422, detail={"error": "missing_features", "message": result["error"]})
            else:
                raise HTTPException(status_code=400, detail=result["error"])

        return result

    except HTTPException:
        raise
    except Exception as e:
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict-yield-lag")
@limiter.limit("5/minute")
async def predict_yield_lag(payload: YieldInput, request: Request):
    await verify_role(request)

    try:
        # Offload time-series lag model prediction to Celery worker pool
        from celery_worker import predict_yield_lag_task
        task = predict_yield_lag_task.delay(payload.data)
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(None, lambda: task.get(timeout=30))
        except Exception as celery_exc:
            raise HTTPException(status_code=504, detail="Prediction timed out or worker unavailable") from celery_exc

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error") from e

@app.post("/predict-yield-trend")
@limiter.limit("5/minute")
async def predict_yield_trend(payload: YieldInput, request: Request):
    await verify_role(request)

    try:
        # Offload heavy iterative trend forecasting to Celery worker pool
        from celery_worker import predict_yield_trend_task
        task = predict_yield_trend_task.delay(payload.data)
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(None, lambda: task.get(timeout=30))
        except Exception as celery_exc:
            raise HTTPException(status_code=504, detail="Prediction timed out or worker unavailable") from celery_exc

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error") from e

@app.get("/api/notifications")
@limiter.limit("30/minute")
async def get_notifications(
    request: Request,
    crop: str = Query(default=None),
    irrigation_count: int = Query(default=None, ge=0),
    water_coverage: int = Query(default=None, ge=0, le=100),
    season: str = Query(default=None)
):
    """
    Return recent triggered-alert notifications combined with dynamic
    farm advisory alerts generated from the query parameters.

    Requires Firebase authentication. Notifications are scoped to broadcast
    entries and entries targeted at the caller's UID.

    Only notifications newer than the store's TTL window are included,
    so the response payload stays small regardless of how long the
    process has been running.
    """
    token_data = await verify_role(request)
    uid = token_data["uid"]
    profile = _get_firestore_user_profile(uid)
    user_regions = frozenset(profile_regions(profile))
    dynamic_alerts = generate_alerts(
        crop=crop,
        irrigation_count=irrigation_count,
        water_coverage=water_coverage,
        season=season
    )
    stored = [
        notification
        for notification in _notification_store.get_recent_for_user(uid)
        if notification_matches_regions(notification, user_regions)
    ]
    return {
        "success": True,
        "data": stored + _normalize_dynamic_alerts(dynamic_alerts),
    }

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
    uid = token_data.get("uid")

    subscriber = {
        "phone_number": data.phone_number,
        "name": data.name,
        "subscribed_at": datetime.now().isoformat(),
        "region_id": normalize_region_identifier(data.region_id) or None,
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
    await asyncio.to_thread(send_whatsapp_message, data.phone_number, welcome_msg)
    return {"success": True, "message": "Successfully subscribed"}

_broadcast_rate_limit = {}

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
    token_data = await verify_role(request)
    uid = token_data["uid"]
    role = str(token_data.get("role", "")).strip().lower()
    region_id = normalize_region_identifier(data.region_id) if data.region_id else ""

    if region_id:
        if role not in {"admin", "expert"}:
            profile = _get_firestore_user_profile(uid)
            if not profile_can_broadcast_region(profile, region_id):
                raise HTTPException(status_code=403, detail="Access denied: insufficient regional authority")
    elif role not in {"admin", "expert"}:
        raise HTTPException(status_code=403, detail="Access denied: insufficient permissions")

    # get_all() acquires the lock and returns a stable snapshot, so this read
    # cannot race with a concurrent subscription write.
    subscribers = subscriber_store.get_all()
    formatted_msg = format_alert_message(data.alert_type, data.message)
    results = []
    if region_id:
        subscribers = {
            user_id: info
            for user_id, info in subscribers.items()
            if _subscriber_matches_region(info, region_id)
        }
    for user_id, info in subscribers.items():
        res = await asyncio.to_thread(send_whatsapp_message, info["phone_number"], formatted_msg)
        results.append({"user_id": user_id, "success": res.get("success", False), "status": res.get("status", "error")})

    # Persist the alert through the bounded, thread-safe notification store.
    await publish_notification(
        alert_type=data.alert_type,
        message=data.message,
        region_id=region_id or None,
    )

    delivered = sum(1 for r in results if r["success"])
    return {"success": True, "results": results, "delivered": delivered, "total": len(results)}


@app.websocket("/api/notifications/stream")
async def notifications_stream(websocket: WebSocket):
    uid = await _authenticate_notification_websocket(websocket)
    if uid is None:
        return
    await notification_broker.connect(websocket, uid, regions=_resolve_websocket_regions(uid, websocket))


@app.get("/metrics")
async def metrics_endpoint(request: Request):
    """Prometheus metrics, restricted to admin/expert roles."""
    await verify_role(request, required_roles=["admin", "expert"])
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/api/admin/rbac-audit")
@limiter.limit("10/minute")
async def get_rbac_audit(request: Request, limit: int = Query(default=50, ge=1, le=200)):
    """Return the most recent RBAC audit events for admins and experts."""
    await verify_role(request, required_roles=["admin", "expert"])
    return {"success": True, "data": rbac_audit_trail.snapshot(limit=limit)}

@app.get("/api/csrf-token")
@limiter.limit("30/minute")
async def get_csrf_token(request: Request):
    """Return a signed CSRF token tied to the authenticated user."""
    token_data = await verify_role(request)
    uid = token_data["uid"]
    token = generate_token(uid)
    return {"csrf_token": token}


@app.post("/api/privacy/deletion-requests")
@limiter.limit("5/minute")
async def request_gdpr_deletion(
    request: Request,
    body: GDPRDeletionRequestBody,
):
    """Create a retention-aware deletion request for the authenticated user."""
    token_data = await verify_role(request)
    uid = token_data["uid"]
    deletion_request = gdpr_deletion_manager.create_request(
        uid,
        requested_by=uid,
        reason=body.reason,
        retention_days=body.retention_days,
    )
    deletion_request["target_names"] = [target.name for target in _build_gdpr_deletion_targets(uid)]
    return {"success": True, "data": deletion_request}


@app.post("/api/admin/privacy/deletion-requests/process-due")
@limiter.limit("5/minute")
async def process_due_gdpr_deletions(request: Request):
    """Execute any deletion requests whose retention window has elapsed."""
    await verify_role(request, required_roles=["admin", "expert"])
    targets = _build_gdpr_deletion_targets("system")
    processed = gdpr_deletion_manager.process_due_requests(targets)
    return {"success": True, "processed": processed, "count": len(processed)}


@app.post("/api/whatsapp/webhook")
@limiter.limit("20/minute")
async def whatsapp_webhook(request: Request):
    """Handle inbound Twilio WhatsApp webhooks (signature-verified)."""
    from twilio_webhook_security import handle_inbound_whatsapp_webhook

    return await handle_inbound_whatsapp_webhook(request)

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
_kms_init_error = None
try:
    from google.cloud import secretmanager as gcp_secretmanager

    secretmanager = gcp_secretmanager
    HAS_GCP_KMS = True
except Exception as e:
    HAS_GCP_KMS = False
    _kms_init_error = str(e)

ALLOW_INSECURE_FALLBACK = os.getenv("ALLOW_INSECURE_KEY_FALLBACK", "false").lower() == "true"
REPORT_SIGNING_PRIVATE_KEY_PEM = os.getenv("REPORT_SIGNING_PRIVATE_KEY_PEM", "").strip()

if os.getenv("GOOGLE_CLOUD_PROJECT") and not HAS_GCP_KMS and not REPORT_SIGNING_PRIVATE_KEY_PEM:
    logger.error(
        f"KMS initialization failed: GOOGLE_CLOUD_PROJECT is set but GCP Secret Manager is unavailable. "
        f"Error: {_kms_init_error}. Set ALLOW_INSECURE_KEY_FALLBACK=true to permit local key fallback (NOT RECOMMENDED)."
    )
    if not ALLOW_INSECURE_FALLBACK:
        raise RuntimeError(
            "SECURITY CRITICAL: KMS is required for production use but is unavailable. "
            "Application will not start with insecure key fallback. "
            "Either configure GCP Secret Manager or explicitly set ALLOW_INSECURE_KEY_FALLBACK=true (NOT RECOMMENDED)."
        )


def get_signing_keys():
    global _cached_private_key

    if _cached_private_key is not None:
        return _cached_private_key

    if REPORT_SIGNING_PRIVATE_KEY_PEM:
        _cached_private_key = serialization.load_pem_private_key(
            REPORT_SIGNING_PRIVATE_KEY_PEM.encode("utf-8"),
            password=None,
        )
        logger.info("Successfully loaded signing key from REPORT_SIGNING_PRIVATE_KEY_PEM")
        return _cached_private_key

    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    secret_id = os.getenv("REPORT_SIGNING_SECRET_NAME", "report-signing-key")

    if project_id:
        if not HAS_GCP_KMS:
            raise RuntimeError("KMS is required when GOOGLE_CLOUD_PROJECT is set")
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
        response = client.access_secret_version(request={"name": name})
        payload = response.payload.data.decode("UTF-8")
        _cached_private_key = serialization.load_pem_private_key(payload.encode(), password=None)
        logger.info("Successfully loaded signing key from GCP KMS")
        return _cached_private_key

    if os.path.exists(PRIVATE_KEY_PATH):
        if not ALLOW_INSECURE_FALLBACK:
            logger.warning(
                "SECURITY WARNING: Falling back to local file-based signing keys. "
                "This is insecure for production use. Set GOOGLE_CLOUD_PROJECT and configure GCP KMS "
                "or set ALLOW_INSECURE_KEY_FALLBACK=true only if you understand the risks."
            )
        with open(PRIVATE_KEY_PATH, "rb") as key_file:
            _cached_private_key = serialization.load_pem_private_key(key_file.read(), password=None)
        logger.warning("Using local file-based signing key - NOT SECURE FOR PRODUCTION")
        return _cached_private_key

    if not ALLOW_INSECURE_FALLBACK:
        logger.error(
            "SECURITY CRITICAL: No secure key source available and ALLOW_INSECURE_KEY_FALLBACK is not enabled. "
            "Refusing to generate insecure keys. Set ALLOW_INSECURE_KEY_FALLBACK=true to permit key generation."
        )
        raise RuntimeError(
            "SECURITY CRITICAL: Cannot proceed without secure key management. "
            "Either configure GCP KMS, provide existing keys, or explicitly allow insecure fallback."
        )

    logger.warning(
        "SECURITY WARNING: Generating new insecure signing keys locally. "
        "This should NEVER happen in production - keys will not persist across restarts."
    )
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


# Observability setup
try:
    from opentelemetry import trace
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor

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
    from prometheus_fastapi_instrumentator import Instrumentator

    Instrumentator().instrument(app)
except Exception as exc:
    logger.warning("Prometheus setup skipped: %s", exc)

# ---------------------------------------------------------------------------
# CORS — explicit origin allowlist
#
# allow_origins=["*"] combined with allow_credentials=True is forbidden by
# the CORS specification and rejected by browsers. It would also allow any
# origin on the internet to make credentialed requests on behalf of a
# logged-in farmer if a non-browser client ever relaxed the check.
#
# Origins are built from:
#   1. Hard-coded local development origins (always included).
#   2. FRONTEND_URL env var — set this to the production deployment URL.
#   3. ADDITIONAL_ALLOWED_ORIGINS env var — comma-separated list for staging
#      or preview deployments (e.g. Vercel preview URLs).
# ---------------------------------------------------------------------------
_CORS_ORIGINS: list[str] = [
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "https://fasal-saathi.vercel.app",
    "https://fasal-saathi.xyz"
]

_frontend_url = os.getenv("FRONTEND_URL", "").strip()
if _frontend_url and _frontend_url not in _CORS_ORIGINS:
    _CORS_ORIGINS.append(_frontend_url)

_extra_origins = os.getenv("ADDITIONAL_ALLOWED_ORIGINS", "").strip()
if _extra_origins:
    for _origin in _extra_origins.split(","):
        _origin = _origin.strip()
        if _origin and _origin not in _CORS_ORIGINS:
            _CORS_ORIGINS.append(_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["Authorization", "Content-Type", "Accept", "Origin", "X-Requested-With"],
)
import csrf_protection as _csrf
_csrf.configure(_CORS_ORIGINS)
app.add_middleware(RBACMiddleware)
logger.info(print_rbac_matrix())

# Import the voice assistant router at module level so app.include_router() can
# reference it during router registration below. Its internal state is
# initialized inside lifespan (above) once all domain engines are ready.
# =============================================================================
# OPTIONAL VOICE ASSISTANT ROUTER IMPORT
# =============================================================================
#
# The voice assistant module is treated as an OPTIONAL feature.
#
# Why?
#
# In production systems some features may depend on:
#
# - extra ML libraries
# - speech models
# - external APIs
# - GPU dependencies
# - audio processing packages
#
# If any dependency is missing, importing the router could crash
# the entire FastAPI application during startup.
#
# This defensive import pattern prevents total backend failure.
#
# -----------------------------------------------------------------------------
# WHAT THIS CODE DOES
# -----------------------------------------------------------------------------
#
# 1. Initializes the variable as None
#
#       voice_assistant_router = None
#
# This guarantees the variable always exists even if import fails.
#
# -----------------------------------------------------------------------------
# 2. Attempts to import the optional router
#
#       from backend.routers import voice_assistant as voice_assistant_router
#
# If successful:
#
# - voice assistant APIs become available
# - routes can later be registered safely
#
# -----------------------------------------------------------------------------
# 3. Handles import failure gracefully
#
# If the import crashes because of:
#
# - missing package
# - syntax error
# - model loading failure
# - missing environment variable
# - incompatible dependency
#
# the backend DOES NOT crash.
#
# Instead:
#
#       logger.warning(...)
#
# records the issue in logs while allowing the rest
# of the API system to continue working normally.
#
# -----------------------------------------------------------------------------
# WHY THIS IS GOOD PRACTICE
# -----------------------------------------------------------------------------
#
# Benefits:
#
# - improves backend reliability
# - prevents startup crashes
# - supports modular architecture
# - allows optional AI features
# - safer deployments
# - easier debugging
#
# This pattern is commonly used in:
#
# - plugin systems
# - AI microservices
# - enterprise APIs
# - feature-flag architectures
#
# -----------------------------------------------------------------------------
# IMPORTANT NOTE
# -----------------------------------------------------------------------------
#
# Because the router may remain None,
# route registration MUST check:
#
#       if voice_assistant_router is not None:
#
# before calling:
#
#       app.include_router(...)
#
# Otherwise FastAPI may crash with:
#
#       AttributeError
#
# -----------------------------------------------------------------------------
# EXAMPLE FLOW
# -----------------------------------------------------------------------------
#
# SUCCESS CASE:
#
#   Import succeeds
#   -> router loads
#   -> APIs enabled
#
# FAILURE CASE:
#
#   Import fails
#   -> warning logged
#   -> backend still starts
#   -> voice APIs disabled only
#
# -----------------------------------------------------------------------------
# FINAL RESULT
# -----------------------------------------------------------------------------
#
# This is a safe and production-friendly optional import pattern.
#
# =============================================================================

voice_assistant_router = None

try:
    from backend.routers import voice_assistant as voice_assistant_router

except Exception as exc:
    logger.warning(
        "Voice assistant router import skipped: %s",
        exc
    )

# Router registration
app.include_router(ml.router, prefix="/api/yield", tags=["ML Prediction"])
app.include_router(governance.router, prefix="/api/ml-governance", tags=["ML Governance"])
app.include_router(finance.router, prefix="/api/farm-finance", tags=["Finance"])
app.include_router(finance.router, prefix="/api/finance", tags=["Finance Legacy"])
app.include_router(quality.router, prefix="/api/crop-quality", tags=["Quality"])
app.include_router(blockchain.router, prefix="/api/supply-chain", tags=["Blockchain"])
app.include_router(reports.router, prefix="/api/admin", tags=["Reports"])
app.include_router(marketplace.router, prefix="/api/marketplace", tags=["Marketplace"])
app.include_router(knowledge.router, prefix="/api/knowledge", tags=["Knowledge"])
app.include_router(community.router, prefix="/api/community", tags=["Community"])
if voice_assistant_router is not None:
    app.include_router(voice_assistant_router.router, prefix="/api/voice", tags=["Voice Assistant"])
app.include_router(referrals.router, prefix="/api/referrals", tags=["Referrals"])
app.include_router(platform.router, prefix="/api", tags=["Platform"])
app.include_router(advisory.router, prefix="/api", tags=["Advisory"])
app.include_router(alerts.router, prefix="/api/notifications", tags=["Alerts"])
app.include_router(flags_router, tags=["Feature Flags"])
app.include_router(lms.router, prefix="/api", tags=["LMS"])


from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Optional, Literal


class SeasonPlanRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    farm_name: str = Field(default="My Farm", max_length=100)
    state: str = Field(..., min_length=2, max_length=50)
    district: str = Field(default="", max_length=100)

    area_acres: float = Field(..., gt=0, le=10000)

    soil_type: str = Field(..., min_length=2, max_length=50)

    season: Literal["Kharif", "Rabi", "Zaid"]

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
        plan = generate_season_plan(data.model_dump() if hasattr(data, "model_dump") else data.dict())
        return {"success": True, "plan": plan}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Autopilot plan generation failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to generate farm plan")


# Include ML Model Management Router (registered once)
try:
    from routers.ml_models import router as ml_router
    app.include_router(ml_router)
    logger.info("ML Model Management API loaded successfully")
except Exception as e:
    logger.warning(f"Could not load ML Model Management API: {e}")

def load_router(router, name: str):
    try:
        app.include_router(router)
        logger.info(f"{name} API loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load {name} API: {e}")


try:
    from routers.retraining_pipeline import router as retraining_router
    load_router(retraining_router, "Retraining Pipeline")
except Exception as e:
    logger.warning(f"Could not import Retraining Pipeline API: {e}")


try:
    from routers.feature_drift import (
        router as feature_drift_router,
        init_auth as init_drift_auth
    )

    init_drift_auth(verify_role)
    load_router(feature_drift_router, "Feature Drift Detection")

except Exception as e:
    logger.warning(f"Could not import Feature Drift Detection API: {e}")


try:
    from routers.crop_recommendation import router as crop_router
    load_router(crop_router, "Crop Recommendation")
except Exception as e:
    logger.warning(f"Could not import Crop Recommendation API: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
