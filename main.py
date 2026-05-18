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
from backend.routers import ml, governance, alerts, finance, quality, blockchain, reports, knowledge, community, voice_assistant

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
app.include_router(community.router, prefix="/api/community", tags=["Community"])
app.include_router(voice_assistant.router, prefix="/api/voice", tags=["Voice Assistant"])


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


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return {"success": False, "error": exc.detail, "status_code": exc.status_code}


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
