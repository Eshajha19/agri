import pandas as pd
import numpy as np
import logging
import os
import joblib
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List
from sklearn.preprocessing import MinMaxScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Global variables for model and scaler caching
model = None
scaler = None

MODEL_PATH = "lstm_yield_model.keras"
SCALER_PATH = "lstm_scaler.pkl"

class PredictionRequest(BaseModel):
    # Expecting sequential data for LSTM
    # e.g., list of past 'seq_length' values
    features: List[List[float]]

class PredictionResponse(BaseModel):
    prediction: float

def train_and_save_model():
    """Original script functionality: Train and save the model."""
    global scaler
    logger.info("Starting model training process...")
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        
        scaler = MinMaxScaler()

        # Load data
        df = pd.read_csv("Train.csv")

        df['SDate'] = pd.to_datetime(df['SDate'], errors='coerce', dayfirst=True)
        df = df.dropna(subset=['SDate'])
        df = df.sort_values('SDate')

        df = df.groupby('SDate')['ExpYield'].mean().reset_index()
        df.set_index('SDate', inplace=True)

        logger.info(f"Data after grouping:\n{df.head()}")

        # Scaling
        scaled_data = scaler.fit_transform(df)

        def create_sequences(data, seq_length=5):
            X, y = [], []
            for i in range(len(data) - seq_length):
                X.append(data[i:i+seq_length])
                y.append(data[i+seq_length])
            return np.array(X), np.array(y)

        X, y = create_sequences(scaled_data)

        logger.info(f"X shape: {X.shape}")
        logger.info(f"y shape: {y.shape}")

        model_seq = Sequential([
            LSTM(64, activation='relu', input_shape=(X.shape[1], 1)),
            Dense(1)
        ])

        model_seq.compile(optimizer='adam', loss='mse')
        model_seq.fit(X, y, epochs=20, batch_size=16)
        model_seq.save(MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        logger.info("✅ LSTM model trained and saved successfully.")
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic: Load the model into memory ONCE during application startup
    global model, scaler
    logger.info("Starting up FastAPI application...")
    
    # We delay keras import to avoid slow startup if not needed
    try:
        from tensorflow.keras.models import load_model
        if not os.path.exists(MODEL_PATH):
            logger.warning(f"Model file {MODEL_PATH} not found. Attempting to train...")
            train_and_save_model()
            
        logger.info(f"Loading model from {MODEL_PATH}...")
        model = load_model(MODEL_PATH)
        logger.info("✅ Model loaded into memory successfully.")

        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            logger.info("✅ Scaler loaded from disk successfully.")
        else:
            logger.error(f"Scaler file {SCALER_PATH} not found. Predictions will be in normalized scale.")
            scaler = None
    except Exception as e:
        logger.error(f"Failed to load model on startup: {str(e)}")
        # If model is None, endpoints will handle it gracefully.
        
    yield
    
    # Shutdown logic
    logger.info("Shutting down FastAPI application... Cleaning up resources.")
    model = None
    scaler = None


# Initialize FastAPI application with lifespan event
app = FastAPI(
    title="LSTM Yield Inference API",
    description="Dedicated inference server for LSTM yield model, loading model once on startup to avoid latency.",
    version="1.0.0",
    lifespan=lifespan
)

_bearer = HTTPBearer(auto_error=False)


async def _require_firebase_auth(request: Request) -> str:
    """Verify a Firebase ID token from the Authorization: Bearer header.

    Returns the verified UID on success.  Raises HTTP 401 on any failure so
    unauthenticated callers cannot reach the inference endpoint.
    """
    credentials: HTTPAuthorizationCredentials | None = await _bearer(request)
    if credentials is None or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Missing authentication token")

    token = credentials.credentials.strip()
    try:
        import firebase_admin
        from firebase_admin import auth as firebase_auth

        # Initialize the Firebase app lazily if it hasn't been done yet.
        # In production the app is typically initialized by the parent process;
        # this guard makes the standalone server self-contained.
        if not firebase_admin._apps:
            firebase_admin.initialize_app()

        decoded = firebase_auth.verify_id_token(token)
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired authentication token") from exc

    uid = decoded.get("uid")
    if not uid:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return uid


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: Request, body: PredictionRequest):
    """
    Inference endpoint.
    Expects input features matching the sequence length used during training.

    Requires a valid Firebase ID token in the Authorization: Bearer header.
    Unauthenticated requests are rejected with HTTP 401 to prevent quota
    exhaustion and automated scraping of model behaviour.
    """
    await _require_firebase_auth(request)

    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Cannot serve predictions.")
    if scaler is None:
        raise HTTPException(status_code=503, detail="Scaler is not loaded. Cannot serve predictions.")
    
    try:
        # Convert request data to numpy array
        # Reshape to match the expected input shape: (batch_size, sequence_length, num_features)
        input_data = np.array(body.features)
        
        if len(input_data.shape) == 2:
            input_data = np.expand_dims(input_data, axis=0)
            
        logger.info(f"Received prediction request with input shape: {input_data.shape}")
        
        # The model is cached in memory, so prediction is fast and doesn't hit disk
        prediction_scaled = model.predict(input_data)
        
        # Inverse transform to convert from normalized [0,1] back to actual yield units
        prediction_unscaled = scaler.inverse_transform(prediction_scaled.reshape(-1, 1))
        pred_value = float(prediction_unscaled[0][0])
        
        return PredictionResponse(prediction=pred_value)
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint to verify model status."""
    status = "healthy" if model is not None else "degraded"
    return {"status": status, "model_loaded": model is not None}

if __name__ == "__main__":
    # When run as a script, we start the inference server locally
    import uvicorn
    # Note: Run it on a specific port for the dedicated inference server
    uvicorn.run("lstm_yield_model:app", host="0.0.0.0", port=8001, reload=False)
