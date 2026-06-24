import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Request, Response
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field

import firebase_admin
from firebase_admin import auth as firebase_auth, firestore
from starlette.middleware.base import BaseHTTPMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -----------------------
# CORS Configuration for Vercel
# -----------------------
import re

def _is_vercel_origin(origin: str) -> bool:
    """Check if origin is a Vercel deployment URL."""
    if not origin:
        return False
    return bool(re.match(r"^https://[a-z0-9-]+\.vercel\.app$", origin.lower()))

# Static allowed origins
STATIC_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "https://fasal-saathi.vercel.app",
    "https://fasal-saathi.xyz",
]

# Add FRONTEND_URL from environment
_frontend_url = os.getenv("FRONTEND_URL", "").strip()
if _frontend_url and _frontend_url not in STATIC_ORIGINS:
    STATIC_ORIGINS.append(_frontend_url)

# Add additional origins from environment
_extra_origins = os.getenv("ADDITIONAL_ALLOWED_ORIGINS", "").strip()
if _extra_origins:
    for origin in _extra_origins.split(","):
        origin = origin.strip()
        if origin and origin not in STATIC_ORIGINS:
            STATIC_ORIGINS.append(origin)


class VercelCORSMiddleware(BaseHTTPMiddleware):
    """CORS middleware supporting dynamic Vercel preview URLs."""
    
    async def dispatch(self, request, call_next):
        origin = request.headers.get("origin")
        response = await call_next(request)
        
        if origin:
            # Allow static origins or Vercel preview URLs
            if origin in STATIC_ORIGINS or _is_vercel_origin(origin):
                response.headers["Access-Control-Allow-Origin"] = origin
                response.headers["Access-Control-Allow-Credentials"] = "true"
                
                if request.method == "OPTIONS":
                    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
                    response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type, Accept, Origin, X-Requested-With"
                    response.headers["Access-Control-Max-Age"] = "3600"
                    return Response(status_code=204)
        
        return response


# -----------------------
# Firebase Initialization (lazy)
# -----------------------
# Render deployments may not have Firebase credentials at import time.
# We initialize Firebase inside FastAPI lifespan so the process can start
# reliably; auth endpoints will return 503 until Firebase is ready.

db_firestore: Optional[firestore.Client] = None
firebase_ready: bool = False


def _init_firebase() -> None:
    global db_firestore, firebase_ready
    if firebase_ready and db_firestore is not None:
        return

    if firebase_admin._apps:
        # App already initialized elsewhere in the process.
        try:
            db_firestore = firestore.client()
            firebase_ready = True
            return
        except Exception:
            firebase_ready = False

    try:
        firebase_admin.initialize_app()
        db_firestore = firestore.client()
        firebase_ready = True
        logger.info("Firebase Admin: successfully initialized")
    except Exception as e:
        firebase_ready = False
        logger.warning("Firebase Admin: could not initialize: %s", e)




# -----------------------
# Models
# -----------------------
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
    user_id: Optional[str] = None

class YieldInput(BaseModel):
    data: list[float]

class AlertTriggerRequest(BaseModel):
    alert_type: str = Field(..., pattern=r'^(weather|pest|advisory)$')
    message: str = Field(..., min_length=1, max_length=500)

class SeedVerifyRequest(BaseModel):
    code: str = Field(..., min_length=4, max_length=100)


# -----------------------
# Request Validators
# -----------------------
_ALLOWED_PREDICTION_FIELDS = frozenset({
    "Crop", "CropCoveredArea", "CHeight", "CNext", "CLast", "CTransp",
    "IrriType", "IrriSource", "IrriCount", "WaterCov", "Season",
    "N", "P", "K", "ph", "pH", "temperature", "rainfall", "humidity",
})

def _coerce_prediction_inputs(input_data: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = dict(input_data)
    
    extra = [k for k in sanitized if k not in _ALLOWED_PREDICTION_FIELDS]
    if extra:
        raise HTTPException(status_code=422, detail=f"Unknown field(s): {', '.join(sorted(extra))}")
    
    numeric_fields = {"N", "P", "K", "ph", "pH", "CropCoveredArea", "CHeight", "IrriCount", "WaterCov", "temperature", "rainfall", "humidity"}
    
    for field in numeric_fields:
        if field not in sanitized or sanitized[field] is None:
            continue
        try:
            sanitized[field] = float(sanitized[field])
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail=f"Invalid value for '{field}'")
    
    if any(field in sanitized and not (0 <= sanitized[field] <= 14) for field in ("ph", "pH") if field in sanitized):
        raise HTTPException(status_code=400, detail="Invalid pH")
    
    return sanitized


# -----------------------
# Notification Store
# -----------------------
import collections
_NOTIFICATION_STORE: collections.deque = collections.deque(maxlen=200)
_NOTIFICATION_TTL = timedelta(hours=24)

def _add_notification(alert_type: str, message: str) -> dict:
    entry = {
        "id": len(_NOTIFICATION_STORE) + 1,
        "type": alert_type,
        "message": message,
        "time": datetime.now().isoformat(),
    }
    _NOTIFICATION_STORE.append(entry)
    return entry


# -----------------------
# Lifespan
# -----------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up")

    # Firebase init during startup (not during import)
    _init_firebase()

    # Best-effort ML warmup so models are functional immediately after deploy.
    # This avoids production issues where the working directory differs.
    global ml_ready
    ml_ready = False
    try:
        from ml.registry import ModelRegistry  # type: ignore
        from ml.adapters.xgboost_adapter import XGBoostAdapter  # type: ignore
        from ml.router import ModelRouter  # type: ignore

        # Resolve model path relative to this file (repo root)
        model_path = os.path.join(os.path.dirname(__file__), "yield_model.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ML model not found at {model_path}")

        adapter = XGBoostAdapter()
        adapter.load(model_path)
        ModelRegistry.register("xgboost", adapter)

        # Touch router to ensure predict pipeline can be constructed
        _ = ModelRouter(default_model="xgboost")
        ml_ready = True
        logger.info("ML warmup completed (xgboost)")
    except Exception:
        # Keep service up, but mark as not ready.
        logger.exception("ML warmup failed")

    yield
    logger.info("Shutting down")




# -----------------------
# FastAPI App
# -----------------------
app = FastAPI(title="Fasal Saathi Backend", version="1.0", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(VercelCORSMiddleware)

# Add CSP middleware
try:
    from csp import build_csp_policy

    class CSPMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            response = await call_next(request)
            response.headers["Content-Security-Policy"] = build_csp_policy()
            return response
    
    app.add_middleware(CSPMiddleware)
except Exception:
    logger.warning("CSP middleware skipped")


# -----------------------
# Auth Helper
# -----------------------
async def verify_role(request: Request, required_roles: list = None):
    """Verify Firebase ID token and check role."""
    auth_header = request.headers.get("Authorization")
    
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authentication token")
    
    id_token = auth_header[7:].strip()
    if not id_token:
        raise HTTPException(status_code=401, detail="Missing or invalid authentication token")
    
    try:
        decoded_token = firebase_auth.verify_id_token(id_token, check_revoked=True)
    except Exception:
        raise HTTPException(status_code=401, detail="Authentication failed")
    
    uid = decoded_token.get("uid")
    
    if db_firestore is None:
        raise HTTPException(status_code=503, detail="Authorization service unavailable")
    
    try:
        user_doc = db_firestore.collection("users").document(uid).get()
    except Exception:
        raise HTTPException(status_code=503, detail="Authorization service unavailable")
    
    if not user_doc.exists:
        raise HTTPException(status_code=403, detail="User profile not found")
    
    user_data = user_doc.to_dict() or {}
    user_role = user_data.get("role", "farmer")
    
    if required_roles and user_role not in required_roles:
        raise HTTPException(status_code=403, detail="Access denied: insufficient permissions")
    
    return {"uid": uid, "role": user_role}


# -----------------------
# Routes
# -----------------------
@app.get("/")
async def root():
    return {"message": "Fasal Saathi API", "status": "running"}


@app.get("/health")
async def health():
    # ml_ready is set by lifespan warmup (best-effort)
    ml_status = globals().get("ml_ready", False)
    return {
        "status": "ok",
        "message": "Backend is running",
        "firebase_ready": firebase_ready,
        "ml_ready": bool(ml_status),
    }




@app.get("/predict")
async def predict_get():
    return {"predicted_yield": 2500, "note": "Use POST endpoint for actual prediction"}


@app.post("/predict", response_model=PredictResponse)
async def predict_yield(data: PredictRequest, request: Request):
    # Role-gated; if Firebase/Firestore isn’t ready, fail clearly.
    await verify_role(request)

    try:
        input_data = data.model_dump() if hasattr(data, "model_dump") else data.dict()
        input_data = _coerce_prediction_inputs(input_data)

        # Import ML router lazily. If ML is missing/unloadable, return 503.
        try:
            from ml.router import ModelRouter  # type: ignore
        except Exception as e:
            logger.exception("ML router import failed")
            raise HTTPException(status_code=503, detail=f"ML pipeline unavailable: {e}")

        # Best-effort model registration.
        # The ML registry must contain at least the default model before
        # ModelRouter.predict() is called.
        try:
            from ml.registry import ModelRegistry  # type: ignore
            if not getattr(ModelRegistry, "_models", None):
                from ml.adapters.xgboost_adapter import XGBoostAdapter  # type: ignore

                model_path = "yield_model.joblib"
                if os.path.exists(model_path):
                    adapter = XGBoostAdapter()
                    adapter.load(model_path)
                    ModelRegistry.register("xgboost", adapter)
        except Exception:
            logger.exception("ML model registration best-effort failed")

        ml_router = ModelRouter(default_model="xgboost")
        try:
            result = ml_router.predict(input_data)
        except Exception as e:
            logger.exception("ML prediction failed")
            raise HTTPException(status_code=503, detail=f"ML prediction unavailable: {e}")

        return result


    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



@app.post("/api/whatsapp/subscribe")
async def subscribe_whatsapp(data: WhatsAppSubscribeRequest, request: Request):
    token_data = await verify_role(request)
    uid = token_data.get("uid")
    
    # Simple in-memory storage for demo
    logger.info(f"WhatsApp subscription for user {uid}: {data.phone_number}")
    return {"success": True, "message": "Successfully subscribed"}


@app.post("/api/whatsapp/trigger-alert")
async def trigger_whatsapp_alert(data: AlertTriggerRequest, request: Request):
    token_data = await verify_role(request, required_roles=["admin", "expert"])
    _add_notification(data.alert_type, data.message)
    return {"success": True, "delivered": 1, "total": 1}


@app.get("/api/notifications")
async def get_notifications(request: Request):
    token_data = await verify_role(request)
    return {"success": True, "data": list(_NOTIFICATION_STORE)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)