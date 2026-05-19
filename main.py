"""Main API entrypoint.

This module is intentionally small: it builds the FastAPI app, initializes
shared services, and registers routers. Domain endpoints live in
`backend.routers.*` modules.
"""

import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Tuple

import firebase_admin
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from firebase_admin import auth, credentials, firestore, storage
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
from persistence.repositories import (
    FinanceApplicationRepository,
    NotificationRepository,
    SupplyChainRepository,
)
from rbac import Permission, RBACManager, RBACMiddleware, print_rbac_matrix
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from weather_alerts import weather_service
from whatsapp_service import format_alert_message, send_whatsapp_message
from whatsapp_store import subscriber_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IdempotencyCache:
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 86400):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Tuple[Any, float]] = {}

    def _evict_expired(self) -> None:
        now = time.time()
        expired_keys = [key for key, (_, ts) in self.cache.items() if now - ts >= self.ttl_seconds]
        for key in expired_keys:
            del self.cache[key]

    def __contains__(self, key: str) -> bool:
        self._evict_expired()
        if key in self.cache:
            _, ts = self.cache[key]
            if time.time() - ts < self.ttl_seconds:
                return True
            del self.cache[key]
        return False

    def __getitem__(self, key: str) -> Any:
        self._evict_expired()
        if key in self.cache:
            value, ts = self.cache[key]
            if time.time() - ts < self.ttl_seconds:
                return value
            del self.cache[key]
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        self._evict_expired()
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        self.cache[key] = (value, time.time())


_IDEMPOTENCY_LOCK = threading.Lock()
_IDEMPOTENCY_CACHE = IdempotencyCache(max_size=1000, ttl_seconds=86400)


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

    token = auth_header.split(" ")[1]
    decoded = auth.verify_id_token(token)
    uid = decoded["uid"]

    db = validate_firestore_ready()
    user_doc = db.collection("users").document(uid).get()
    user_roles = user_doc.get("roles", []) if user_doc.exists else []

    if required_roles:
        has_access = all(role in user_roles for role in required_roles) if require_all else any(
            role in user_roles for role in required_roles
        )
        if not has_access:
            raise HTTPException(status_code=403, detail="Insufficient permissions")

    return {"uid": uid, "roles": user_roles}


def sanitise_log_field(value: str) -> str:
    if not isinstance(value, str):
        value = str(value)
    sanitised = "".join(ch if ord(ch) >= 32 or ch in "\n\t" else f"\\x{ord(ch):02x}" for ch in value)
    return sanitised[:1000]


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
    voice_assistant_router.init_voice_assistant(voice_asst, cache_mgr)
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
