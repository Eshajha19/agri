from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd

# Load trained model (ensure this path is correct)
model = joblib.load("yield_model.joblib")

# Create FastAPI app
app = FastAPI()

# Allow CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"],
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
