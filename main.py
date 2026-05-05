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

# ML Ops Imports
from ml.registry import ModelRegistry
from ml.adapters.xgboost_adapter import XGBoostAdapter
from ml.router import ModelRouter

# Other internal modules
from alert_rules import generate_alerts
from whatsapp_service import send_whatsapp_message, format_alert_message

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import inch
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    # FIX: Add max_length to prevent OOM exceptions
    # A malicious user can send a 5GB string in the Crop field, causing an Out of Memory (OOM) exception on the server.
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
    # FIX: Add max_length to prevent OOM exceptions
    # A malicious user can send a 5GB string in the Season field, causing an Out of Memory (OOM) exception on the server.
    Season: str = Field(..., max_length=50)

class PredictResponse(BaseModel):
    predicted_ExpYield: float

class WhatsAppSubscribeRequest(BaseModel):
    phone_number: str
    user_id: str
    name: str

class AlertTriggerRequest(BaseModel):
    alert_type: str  # 'weather', 'pest', 'advisory'
    message: str


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

# Store notifications
@app.get("/api/notifications")
def get_notifications(
    crop: str = Query(default=None),
    irrigation_count: int = Query(default=None, ge=0),
    water_coverage: int = Query(default=None, ge=0, le=100),
    season: str = Query(default=None)
):
    """
    Generate dynamic farm advisory alerts.
    
    Query params (all optional):
    - crop: rice / wheat / maize
    - irrigation_count: number of irrigations done
    - water_coverage: 0-100 (% of field covered)
    - season: kharif / rabi / zaid (auto-detected if not passed)
    """
    alerts = generate_alerts(
        crop=crop,
        irrigation_count=irrigation_count,
        water_coverage=water_coverage,
        season=season
    )
@app.get("/")
def root():
    return {"message": "Fasal Saathi Yield Prediction API", "status": "running"}

@app.post("/predict", response_model=PredictResponse)
def predict_yield(data: PredictRequest, request: Request):
    """
    Standardized prediction endpoint using ML Router for dynamic model selection.
    """
    try:
        input_data = data.model_dump() if hasattr(data, 'model_dump') else data.dict()
        
        context = {
            "location": request.headers.get("X-User-Location", "Unknown"),
            "crop": data.Crop
        }
        
        predicted_yield = router.predict(input_data, context)
        return {"predicted_ExpYield": float(predicted_yield)}
    except Exception as e:
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/notifications")
def get_notifications(
    crop: str = Query(default=None),
    irrigation_count: int = Query(default=None, ge=0),
    water_coverage: int = Query(default=None, ge=0, le=100),
    season: str = Query(default=None)
):
    """Generate dynamic farm advisory alerts + static ones."""
    static_notifications = [
        {
            "id": 1,
            "type": "weather",
            "message": "🌧️ Heavy rainfall expected in your region today.",
            "time": datetime.now().isoformat()
        }
    ]
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
        with open(SUBSCRIBERS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_subscribers(subscribers):
    with open(SUBSCRIBERS_FILE, "w") as f:
        json.dump(subscribers, f)

@app.post("/api/whatsapp/subscribe")
async def subscribe_whatsapp(data: WhatsAppSubscribeRequest):
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
KEYS_DIR = "keys"
PRIVATE_KEY_PATH = os.path.join(KEYS_DIR, "report_signing.key")
PUBLIC_KEY_PATH = os.path.join(KEYS_DIR, "report_signing.pub")

def get_signing_keys():
    if not os.path.exists(KEYS_DIR): os.makedirs(KEYS_DIR)
    if os.path.exists(PRIVATE_KEY_PATH):
        with open(PRIVATE_KEY_PATH, "rb") as f:
            return serialization.load_pem_private_key(f.read(), password=None)
    
    private_key = ed25519.Ed25519PrivateKey.generate()
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
    return private_key

class ReportRequest(BaseModel):
    name: str = Field(..., max_length=100)
    crop: str = Field(..., max_length=50)
    area: str = Field(..., max_length=50)
    profit: str = Field(..., max_length=50)
    season: str = Field(..., max_length=50)

@app.post("/api/reports/generate")
async def generate_signed_report(data: ReportRequest):
    try:
        private_key = get_signing_keys()
        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter

        p.setFont("Helvetica-Bold", 24)
        p.setFillColor(colors.green)
        p.drawCentredString(width/2, height - 1*inch, "FASAL SAATHI")
        
        p.setFont("Helvetica-Bold", 18)
        p.setFillColor(colors.black)
        p.drawCentredString(width/2, height - 1.5*inch, "CERTIFIED FINANCIAL FARM REPORT")
        
        p.setFont("Helvetica", 14)
        y = height - 2.5*inch
        details = [
            ("Farmer Name:", data.name), ("Crop Type:", data.crop),
            ("Farm Area:", data.area), ("Season Profit:", f"Rs. {data.profit}"),
            ("Season:", data.season), ("Report Date:", datetime.now().strftime("%d %B, %Y")),
        ]

        for label, value in details:
            p.drawString(1.5*inch, y, label)
            p.drawString(3.5*inch, y, value)
            y -= 0.4*inch

        report_data_string = f"{data.name}|{data.crop}|{data.area}|{data.profit}|{datetime.now().date()}"
        signature = private_key.sign(report_data_string.encode())
        sig_id = hashlib.sha256(signature).hexdigest()[:8].upper()

        p.rect(1*inch, y - 1.5*inch, width - 2*inch, 1.8*inch)
        p.drawString(1.2*inch, y - 0.3*inch, f"Signature ID: {sig_id}")
        p.drawString(1.2*inch, y - 0.7*inch, "Status: VERIFIED ✔")

        p.showPage()
        p.save()
        pdf_content = buffer.getvalue()
        buffer.close()

        return Response(
            content=pdf_content,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=FasalSaathi_Report_{sig_id}.pdf"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/log-error")
async def log_error(request: Request):
    try:
        error_data = await request.json()
        print(f"[Error Log] {error_data.get('message', 'Unknown error')}")
        return {"success": True}
    except Exception:
        return {"success": False}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
