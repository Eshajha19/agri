from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import os

# Load trained model (ensure this path is correct)
model = joblib.load("yield_model.joblib")

# Create FastAPI app
app = FastAPI()

# --- Secure CORS Configuration ---
frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173")
trusted_origins = [
    "http://localhost:5173",     # Local development
    "http://127.0.0.1:5173",     # Local development alternative
    "https://yourfrontend.com",  # Production domain placeholder
]

if frontend_url and frontend_url not in trusted_origins:
    trusted_origins.append(frontend_url)

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

@app.get("/predict")
def predict():
    # Dummy input matching training features
    input_df = pd.DataFrame([{
        "NDVI": 4800,
        "Rainfall": 25.0,
        "SoilMoisture": 4.5,
        "Crop-wise_Rice": 1  # Example crop, adjust if needed
    }])

    # Ensure all model features exist
    for col in model.get_booster().feature_names:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[model.get_booster().feature_names]

    # Make prediction
    prediction = model.predict(input_df)[0]
    return {"predicted_yield": float(prediction)}
