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

# Import resilience layer
from firestore_resilience import get_resilient_client

# keep upstream feature-flags router if present
try:
    from feature_flags.routes import router as flags_router
except Exception:
    flags_router = None

# Import modular routers
from backend.routers import ml, governance, alerts, finance, quality, blockchain, reports, knowledge

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        
        # Use resilient client for Firestore access
        resilient_db = get_resilient_client()
        user_data = resilient_db.get("users", uid)
        user_roles = user_data.get("roles", []) if user_data else []
        
        if required_roles:
            if require_all:
                has_access = all(role in user_roles for role in required_roles)
            else:
                has_access = any(role in user_roles for role in required_roles)
            if not has_access:
                raise HTTPException(status_code=403, detail="Insufficient permissions")
        return {"uid": uid, "roles": user_roles}
    except Exception as e:
        logger.error(f"Auth error: {e}")
        raise HTTPException(status_code=401, detail="Invalid token")


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
    try:
        if os.path.exists("serviceAccountKey.json"):
            cred = credentials.Certificate("serviceAccountKey.json")
            firebase_admin.initialize_app(cred)
        logger.info("Firebase initialized")
    except Exception as e:
        logger.error(f"Firebase init failed: {e}")
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

# include flags router if available
if flags_router is not None:
    app.include_router(flags_router)

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

    reports.init_reports(verify_role, lambda: "mock_key", sanitise_log_field, logger)

    def rag_generate_fn(query: str, top_k: int = 3):
        return [{"relevance": 0.9, "text": f"RAG result for {query}"}]

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


@app.get("/")
def health_check():
    return {"status": "healthy", "service": "Fasal Saathi Backend", "version": "2.0"}


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/health/detailed")
def detailed_health():
    """Comprehensive health check including Firestore resilience status"""
    resilient_db = get_resilient_client()
    health_status = resilient_db.get_health_status()
    
    return {
        "service": "Fasal Saathi Backend",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "firestore": health_status,
        "queue_status": {
            "pending_writes": len(resilient_db.write_queue),
            "circuit_breaker_state": resilient_db.circuit_breaker.state.value
        }
    }


@app.post("/health/flush-queue")
def flush_queue():
    """Manually flush offline write queue"""
    resilient_db = get_resilient_client()
    flushed = resilient_db.flush_queue()
    
    return {
        "flushed": flushed,
        "pending": len(resilient_db.write_queue),
        "timestamp": datetime.now().isoformat()
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return {"success": False, "error": exc.detail, "status_code": exc.status_code}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
