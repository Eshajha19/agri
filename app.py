from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import joblib
import pandas as pd
import numpy as np
from typing import List
from pydantic import BaseModel

# Load trained models (ensure this path is correct)
try:
    model = joblib.load("yield_model.joblib")
    model_lag = joblib.load("sklearn_yield_model.pkl")
    print("✅ Models loaded successfully")
except Exception as e:
    print(f"Warning: Could not load models: {e}")
    model = None
    model_lag = None

# Create FastAPI app
app = FastAPI()
class YieldInput(BaseModel):
    data: List[float]

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------
# FIX: Inject standard security headers
# ----------------------------------------------------------------------
# Leaving out headers makes the frontend vulnerable to clickjacking and 
# data exfiltration. This middleware injects CSP, X-Frame-Options, 
# and HSTS headers.
# ----------------------------------------------------------------------
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        # Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"
        # Enforce HTTP Strict Transport Security
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        # Content Security Policy to mitigate XSS and data exfiltration
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        return response

app.add_middleware(SecurityHeadersMiddleware)



@app.get("/predict")
def predict():
    if model is None:
        return {"error": "Model not available", "predicted_yield": None}
    
    # Dummy input matching training features
    input_df = pd.DataFrame([{
        "NDVI": 4800,
        "Rainfall": 25.0,
        "SoilMoisture": 4.5,
        "Crop-wise_Rice": 1
    }])

    for col in model.get_booster().feature_names:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[model.get_booster().feature_names]

    prediction = model.predict(input_df)[0]
    return {"predicted_yield": float(prediction)}

@app.post("/predict-yield-lag")
async def predict_yield_lag(payload: YieldInput):
    if model_lag is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    try:
        # ----------------------------------------------------------------------
        # EXPLANATION OF FIX: Using payload.data instead of input.data
        # ----------------------------------------------------------------------
        # Previously, the code was attempting to read from `input.data`. Since
        # `input` is a built-in Python function, this resulted in an
        # `AttributeError`, causing the endpoint to fail. The fix involves
        # using the properly injected `payload` parameter from FastAPI.
        # When a client sends a POST request, FastAPI parses the JSON body,
        # validates it against our `YieldInput` Pydantic model, and passes it
        # to the route handler as the `payload` argument. Accessing 
        # `payload.data` correctly retrieves the list of floats required
        # for our prediction logic, avoiding any namespace collision with 
        # Python's built-in functions and ensuring reliable data extraction.
        # ----------------------------------------------------------------------
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
async def predict_yield_trend(payload: YieldInput):
    if model_lag is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    try:
        # ----------------------------------------------------------------------
        # EXPLANATION OF FIX: Using payload.data instead of input.data
        # ----------------------------------------------------------------------
        # Just as in the lag endpoint, we must avoid shadowing or referencing
        # Python's built-in `input` function. Referencing `input.data` would
        # trigger an `AttributeError` and crash this trend prediction route.
        # By utilizing `payload.data`, we securely access the validated request
        # body provided by FastAPI's dependency injection system. The `payload`
        # is an instance of the `YieldInput` model, guaranteeing that `.data` 
        # is a safely structured list of numbers. This ensures seamless 
        # data flow into our time-series forecasting loop below without
        # encountering unexpected runtime exceptions due to built-ins.
        # ----------------------------------------------------------------------
        data = payload.data

        if len(data) != 5:
            raise ValueError("Exactly 5 values are required")

        temp = data[::-1]  # reverse once
        trend = []

        for _ in range(5):
            features = temp[:5]  # latest 5 values (correct order)

            pred = model_lag.predict([features])[0]
            pred_value = round(float(pred), 2)

            trend.append(pred_value)
            

            temp = [pred_value] + temp  # maintain correct lag order

        # ✅ return AFTER loop (inside try)
        return {
            "trend": trend,
            "prediction": trend[-1],
            "model": "RandomForest Trend Forecast (Lag Features)"
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error") from e