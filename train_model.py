import argparse
import json
import os
import random
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np
import os
import hmac
import hashlib
import json
from datetime import datetime
from pathlib import Path

# ── Retraining pipeline helpers ──────────────────────────────────────────────

_CAT_COLS = ['Crop', 'CNext', 'CLast', 'CTransp', 'IrriType', 'IrriSource', 'Season']
_DROP_COLS = ["FarmID", "category", "State", "District", "Sub-District",
              "SDate", "HDate", "ExpYield", "geometry"]


def save_feature_baseline(X_raw, output_path="feature_baseline.json"):
    """Persist training feature statistics for drift detection."""
    numeric_features = {}
    categorical_features = {}

    for col in X_raw.columns:
        if col in _CAT_COLS:
            vc = X_raw[col].dropna().value_counts(normalize=True)
            categorical_features[col] = {
                "categories": vc.index.tolist(),
                "value_counts": vc.to_dict(),
            }
        else:
            series = X_raw[col].dropna().astype(float)
            sample = series.sample(min(500, len(series)), random_state=42).tolist()
            numeric_features[col] = {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
                "sample_values": sample,
            }

    baseline = {
        "generated_at": datetime.utcnow().isoformat(),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }
    tmp = Path(output_path).with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2)
    os.replace(tmp, output_path)
    return baseline


def train_yield_model(
    csv_path="Train.csv",
    model_output="yield_model.joblib",
    baseline_output="feature_baseline.json",
):
    """
    Callable training entry point used by the retraining pipeline Celery task.
    Returns dict: rmse, model_path, baseline_path, trained_at.
    Mirrors the script body exactly — single source of truth.
    """
    df = pd.read_csv(csv_path)
    df['SDate'] = pd.to_datetime(df['SDate'], errors='coerce')
    df = df.dropna(subset=['SDate'])
    df = df.sort_values('SDate')

    X = df.drop(columns=[c for c in _DROP_COLS if c in df.columns], errors='ignore')
    y = df["ExpYield"]

    # Save baseline from raw X (before get_dummies)
    save_feature_baseline(X, baseline_output)

    X = pd.get_dummies(X, columns=_CAT_COLS, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = xgb.XGBRegressor(n_estimators=200, max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))

    # Atomic save — prevents serving a half-written file
    tmp_path = model_output + ".tmp"
    joblib.dump(model, tmp_path)
    os.replace(tmp_path, model_output)

    signing_key = os.getenv("MODEL_SIGNING_KEY")
    if signing_key:
        with open(model_output, "rb") as f:
            raw = f.read()
        sig = hmac.new(signing_key.encode("utf-8"), raw, hashlib.sha256).hexdigest()
        with open(model_output + ".sig", "w", encoding="utf-8") as sf:
            sf.write(sig)

    return {
        "rmse": rmse,
        "model_path": model_output,
        "baseline_path": baseline_output,
        "trained_at": datetime.utcnow().isoformat(),
    }


# ── Script body (unchanged) ───────────────────────────────────────────────────
df = pd.read_csv("Train.csv")
# Convert SDate to datetime
df['SDate'] = pd.to_datetime(df['SDate'], errors='coerce')
df = df.dropna(subset=['SDate'])
df = df.sort_values('SDate')
print(df[['SDate', 'ExpYield']].head())


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:
        pass


def train_from_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    seed = config.get("seed", 42)
    set_seeds(seed)

    dataset_path = config.get("dataset", "Train.csv")
    dry_run = config.get("dry_run", False)

    # create run manifest (data provenance)
    manifest = create_run_manifest([dataset_path], config)
    print("Run manifest created with id:", manifest["run_id"])

    # light-weight schema validation (CI-safe)
    try:
        validate_csv_schema(dataset_path)
    except Exception as e:
        print("Dataset schema validation failed:", e)
        raise

    if dry_run:
        print("Dry run requested; skipping training.")
        return manifest

    df = pd.read_csv(dataset_path)
    # Convert SDate to datetime
    df["SDate"] = pd.to_datetime(df["SDate"], errors="coerce")
    df = df.dropna(subset=["SDate"]) 
    df = df.sort_values("SDate")

    X = df.drop(
        columns=["FarmID", "category", "State", "District", "Sub-District", "SDate", "HDate", "ExpYield", "geometry"],
        errors="ignore",
    )
    y = df["ExpYield"]

    categorical_cols = config.get(
        "categorical_cols", ["Crop", "CNext", "CLast", "CTransp", "IrriType", "IrriSource", "Season"]
    )
    X = pd.get_dummies(X, columns=[c for c in categorical_cols if c in X.columns], drop_first=True)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.get("test_size", 0.2), random_state=seed)

    # Train model
    model = xgb.XGBRegressor(n_estimators=config.get("n_estimators", 200), max_depth=config.get("max_depth", 6), random_state=seed)
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print("✅ Model trained successfully")
    print("📊 RMSE:", rmse)

    out_path = config.get("output_model", "yield_model.joblib")
    joblib.dump(model, out_path)

    # create model manifest and register
    model_name = config.get("model_name", "yield_model")
    version = manifest.get("run_id")
    manifest_meta = create_manifest(out_path, model_name, version, created_by=config.get("created_by", "ci"))

    # optional signing
    signing_key = os.getenv("MODEL_SIGNING_KEY")
    if signing_key:
        sig_hex = sign_file_hmac(out_path, signing_key)
        manifest_meta["signature_hmac_sha256"] = sig_hex
        sig_path = out_path + ".sig"
        with open(sig_path, "w", encoding="utf-8") as sf:
            sf.write(sig_hex)
        print(f"Wrote signature to {sig_path}")
    else:
        print("MODEL_SIGNING_KEY not set; no signature written")

    # register model in in-memory registry (for CI/tests)
    registry = get_model_registry()
    registry.register_model(model_name=model_name, version=version, model_path=out_path, created_by=config.get("created_by", "ci"), description=config.get("description"), metrics={"rmse": float(rmse)})

    return manifest_meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True, help="Path to JSON config file")
    args = parser.parse_args()

    train_from_config(args.config)


if __name__ == "__main__":
    main()
