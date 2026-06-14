import pandas as pd
import numpy as np
import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from sklearn.preprocessing import MinMaxScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Global variables for model and scaler caching
model = None
scaler = MinMaxScaler()

MODEL_PATH = "lstm_yield_model.keras"

class PredictionRequest(BaseModel):
    # Expecting sequential data for LSTM
    # e.g., list of past 'seq_length' values
    features: List[List[float]]

class PredictionResponse(BaseModel):
    prediction: float

def train_and_save_model():
    """Original script functionality: Train and save the model."""
    logger.info("Starting model training process...")
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        
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
        import joblib
        joblib.dump(scaler, "lstm_yield_scaler.joblib")
        logger.info("✅ Scaler saved successfully.")

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
        
        import joblib
        if os.path.exists("lstm_yield_scaler.joblib"):
            scaler = joblib.load("lstm_yield_scaler.joblib")
            logger.info("✅ Scaler loaded into memory successfully.")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {str(e)}")
        # If model is None, endpoints will handle it gracefully.
        
    yield
    
    # Shutdown logic
    logger.info("Shutting down FastAPI application... Cleaning up resources.")
    model = None


# Initialize FastAPI application with lifespan event
app = FastAPI(
    title="LSTM Yield Inference API",
    description="Dedicated inference server for LSTM yield model, loading model once on startup to avoid latency.",
    version="1.0.0",
    lifespan=lifespan
)


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Inference endpoint.
    Expects input features matching the sequence length used during training.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Cannot serve predictions.")
    
    try:
        # Convert request data to numpy array
        # Reshape to match the expected input shape: (batch_size, sequence_length, num_features)
        input_data = np.array(request.features)
        
        if len(input_data.shape) == 2:
            input_data = np.expand_dims(input_data, axis=0)
            
        logger.info(f"Received prediction request with input shape: {input_data.shape}")
        
        # The model is cached in memory, so prediction is fast and doesn't hit disk
        prediction_scaled = model.predict(input_data)
        
        # Extract the float prediction
        pred_value = float(prediction_scaled[0][0])
        
        # Perform inverse transformation to restore actual unit scale if fitted
        if hasattr(scaler, "scale_"):
            dummy = np.zeros((1, 1))
            dummy[0, 0] = pred_value
            pred_actual = float(scaler.inverse_transform(dummy)[0, 0])
        else:
            pred_actual = pred_value
            
        return PredictionResponse(prediction=pred_actual)
    
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