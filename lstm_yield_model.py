import os
from typing import List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.preprocessing import MinMaxScaler

MODEL_PATH = "lstm_yield_model.keras"
SCALER_PATH = "lstm_yield_model.scaler.npz"
SEQ_LENGTH = 5

class PredictionRequest(BaseModel):
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


app = FastAPI(
    title="LSTM Yield Inference API",
    description="Simple LSTM inference service for yield forecasting.",
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


@app.get("/health")
def health_check():
    return {
        "status": "healthy" if getattr(app.state, "model", None) is not None else "unhealthy",
        "model_loaded": getattr(app.state, "model", None) is not None,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("lstm_yield_model:app", host="0.0.0.0", port=8001, reload=False)
