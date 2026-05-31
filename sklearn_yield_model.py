import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
import hmac
import hashlib


def load_model(path: str = "sklearn_yield_model.pkl"):
    """Load a trained sklearn yield model from disk."""
    return joblib.load(path)


def _write_signature(artifact_path: str, signing_key: str) -> None:
    """Write an HMAC-SHA256 signature file for *artifact_path*."""
    with open(artifact_path, "rb") as f:
        data = f.read()
    sig = hmac.new(signing_key.encode("utf-8"), data, hashlib.sha256).hexdigest()
    sig_path = artifact_path + ".sig"
    with open(sig_path, "w", encoding="utf-8") as sf:
        sf.write(sig)
    print(f"Wrote signature to {sig_path}")


if __name__ == "__main__":
    # Load data
    df = pd.read_csv("Train.csv")

    # Convert date
    df['SDate'] = pd.to_datetime(df['SDate'], errors='coerce', dayfirst=True)
    df = df.dropna(subset=['SDate'])
    df = df.sort_values('SDate')

    # Group by date
    df = df.groupby('SDate')['ExpYield'].mean().reset_index()

    # 🔥 CREATE LAG FEATURES (THIS REPLACES LSTM)
    for i in range(1, 6):
        df[f'lag_{i}'] = df['ExpYield'].shift(i)

    # Drop missing rows
    df = df.dropna()

    # Features and target
    X = df[[f'lag_{i}' for i in range(1, 6)]]
    y = df['ExpYield']

    # ------------------------------------------------------------------ #
    # 1. Lag model — single-step yield prediction                         #
    # ------------------------------------------------------------------ #
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Save model
    joblib.dump(model, "sklearn_yield_model.pkl")

    # If a model signing key is available, create a signature file to allow
    # verification when the model is loaded in production.
    signing_key = os.getenv("MODEL_SIGNING_KEY")
    if signing_key:
        _write_signature("sklearn_yield_model.pkl", signing_key)
    else:
        print("MODEL_SIGNING_KEY not set; no signature file written for lag model")

    print("✅ Sklearn time-series (lag) model trained and saved")

    # ------------------------------------------------------------------ #
    # 2. Trend forecast model — multi-step recursive forecasting          #
    #                                                                     #
    # Uses deeper trees and more estimators so that autoregressive error  #
    # accumulation is minimised over multiple prediction steps.           #
    # Saved as 'trend_forecast_model.joblib' — the artifact path that    #
    # celery_worker._get_trend_model() and main.py both look for.        #
    # Without this file on disk both callers returned None immediately    #
    # and every trend prediction request failed with RuntimeError.        #
    # ------------------------------------------------------------------ #
    trend_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
    )
    trend_model.fit(X, y)

    trend_model_path = "trend_forecast_model.joblib"
    joblib.dump(trend_model, trend_model_path)

    if signing_key:
        _write_signature(trend_model_path, signing_key)
    else:
        print("MODEL_SIGNING_KEY not set; no signature file written for trend model")

    print("✅ Trend forecast model trained and saved as trend_forecast_model.joblib")