import os
from typing import List

import numpy as np
import pandas as pd
import joblib
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.security import HTTPBearer
from pydantic import BaseModel
from sklearn.preprocessing import MinMaxScaler

MODEL_PATH = "lstm_yield_model.keras"
SCALER_PATH = "lstm_yield_model.scaler.npz"
SEQ_LENGTH = 5
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


def create_sequences(values: np.ndarray, seq_length: int = SEQ_LENGTH):
    return (
        np.array([values[i : i + seq_length] for i in range(len(values) - seq_length)]),
        values[seq_length :],
    )


def save_scaler(scaler: MinMaxScaler, path: str = SCALER_PATH):
    np.savez_compressed(
        path,
        scale=scaler.scale_,
        min=scaler.min_,
        data_min=scaler.data_min_,
        data_max=scaler.data_max_,
        n_samples_seen=scaler.n_samples_seen_,
    )


def load_scaler(path: str = SCALER_PATH) -> MinMaxScaler:
    data = np.load(path)
    scaler = MinMaxScaler()
    scaler.scale_ = data["scale"]
    scaler.min_ = data["min"]
    scaler.data_min_ = data["data_min"]
    scaler.data_max_ = data["data_max"]
    scaler.n_samples_seen_ = data["n_samples_seen"]
    return scaler


def train_and_save_model(
    data_path: str = "Train.csv",
    model_path: str = MODEL_PATH,
    scaler_path: str = SCALER_PATH,
):
    from tensorflow.keras.layers import Dense, LSTM
    from tensorflow.keras.models import Sequential

    df = pd.read_csv(data_path, parse_dates=["SDate"], dayfirst=True)
    df = (
        df.dropna(subset=["SDate", "ExpYield"])
        .sort_values("SDate")
        .groupby("SDate", as_index=False)["ExpYield"]
        .mean()
    )

    values = df[["ExpYield"]].to_numpy()
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(values)
    X, y = create_sequences(scaled_values)

    model = Sequential(
        [
            LSTM(64, activation="relu", input_shape=(X.shape[1], X.shape[2])),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=20, batch_size=16, verbose=0)
    model.save(model_path)
    save_scaler(scaler, scaler_path)
    return model, scaler


def load_model_and_scaler(
    model_path: str = MODEL_PATH, scaler_path: str = SCALER_PATH
):
    from tensorflow.keras.models import load_model

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return train_and_save_model()

    model = load_model(model_path)
    scaler = load_scaler(scaler_path)
    return model, scaler


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
)


@app.on_event("startup")
def startup_event():
    app.state.model, app.state.scaler = load_model_and_scaler()


def normalize_input(features: np.ndarray) -> np.ndarray:
    if features.ndim == 2:
        features = features[np.newaxis, ...]

    if features.ndim != 3 or features.shape[2] != 1:
        raise ValueError("Input must be a 2D or 3D array with a single feature per time step.")

    flattened = features.reshape(-1, 1)
    scaled = app.state.scaler.transform(flattened)
    return scaled.reshape(features.shape)


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if not hasattr(app.state, "model") or app.state.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    raw_input = np.asarray(request.features, dtype=float)
    try:
        input_data = normalize_input(raw_input)
        prediction = app.state.model.predict(input_data, verbose=0).squeeze()
        return PredictionResponse(prediction=float(prediction))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

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
def health_check():
    return {
        "status": "healthy" if getattr(app.state, "model", None) is not None else "unhealthy",
        "model_loaded": getattr(app.state, "model", None) is not None,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
