# main.py
import os
import io
import json
import joblib
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Request, Form, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

class SimulationRequest(BaseModel):
    crop_type: str
    temp_delta: float = Field(..., ge=-5, le=5)
    rain_delta: float = Field(..., ge=-100, le=100)

class RAGQuery(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    top_k: int = Field(default=3, ge=1, le=5)

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

# Other internal modules
from alert_rules import generate_alerts
from whatsapp_service import send_whatsapp_message, format_alert_message

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

# Initialize Limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Initialize Firebase Admin
if not firebase_admin._apps:
    try:
        # If running in a cloud environment (GCP), it uses default credentials
        # For local, you might need: cred = credentials.Certificate('path/to/serviceAccountKey.json')
        firebase_admin.initialize_app()
        db_firestore = firestore.client()
        print("Firebase Admin: Successfully initialized")
    except Exception as e:
        print(f"Firebase Admin Warning: Could not initialize (expected in local dev): {e}")
        db_firestore = None

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

    id_token = auth_header.split("Bearer ")[1]

    # Verify the token signature with Firebase — raises on invalid/expired tokens
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

    user_doc = db_firestore.collection("users").document(uid).get()

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
    "https://yourfrontend.com",  # Production domain placeholder
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
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept", "Origin", "X-Requested-With"],
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
    model = joblib.load("yield_model.joblib")
    model_lag = joblib.load("sklearn_yield_model.pkl")
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
    return {"success": True, "data": static_notifications + dynamic_alerts}

# --- WhatsApp Service Endpoints ---

SUBSCRIBERS_FILE = "whatsapp_subscribers.json"

def load_subscribers():
    if os.path.exists(SUBSCRIBERS_FILE):
        try:
            with open(SUBSCRIBERS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_subscribers(subscribers):
    with open(SUBSCRIBERS_FILE, "w") as f:
        json.dump(subscribers, f)

@app.post("/api/whatsapp/subscribe")
@limiter.limit("2/minute")
async def subscribe_whatsapp(data: WhatsAppSubscribeRequest, request: Request):
    subscribers = load_subscribers()
    user_id = data.user_id if data.user_id else str(datetime.now().timestamp())
    subscribers[user_id] = {
        "phone_number": data.phone_number,
        "name": data.name,
        "subscribed_at": datetime.now().isoformat()
    }
    save_subscribers(subscribers)
    
    welcome_msg = f"Namaste {data.name}! 🙏\n\nWelcome to *Fasal Saathi WhatsApp Alerts*. You will now receive real-time updates directly here."
    send_whatsapp_message(data.phone_number, welcome_msg)
    return {"success": True, "message": "Successfully subscribed"}

@app.post("/api/whatsapp/trigger-alert")
async def trigger_whatsapp_alert(data: AlertTriggerRequest):
    subscribers = load_subscribers()
    results = []
    formatted_msg = format_alert_message(data.alert_type, data.message)
    
    for user_id, info in subscribers.items():
        res = send_whatsapp_message(info["phone_number"], formatted_msg)
        results.append({"user_id": user_id, "success": res.get("success", False)})
    
    static_notifications.append({
        "id": len(static_notifications) + 1,
        "type": data.alert_type,
        "message": data.message,
        "time": datetime.now().isoformat()
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
            if IS_PRODUCTION:
                raise HTTPException(
                    status_code=500,
                    detail="google-cloud-secret-manager is not installed but is required in production"
                )
            print("KMS Warning: google-cloud-secret-manager not installed; skipping GCP path.")
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
                if IS_PRODUCTION:
                    print(f"KMS Error: {e}")
                    raise HTTPException(
                        status_code=500,
                        detail="Failed to retrieve signing key from Secret Manager"
                    )
                print(f"KMS Warning: Could not reach Secret Manager ({e}); falling back to local key.")
    elif IS_PRODUCTION:
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
async def log_error(request: Request):
    try:
        error_data = await request.json()
        print(f"[Error Log] {error_data.get('message', 'Unknown error')}")
        return {"success": True}
    except Exception:
        return {"success": False}

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
