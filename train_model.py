import argparse
import json
import math
import os
import random
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from ml.repro import create_run_manifest
from ml.dp_privacy import (
    approximate_epsilon,
    build_privacy_budget,
)
from ml.model_manifest import create_manifest
from ml_model_registry import get_model_registry
from ml.ci_pipeline import (
    validate_csv_schema,
    sign_file_hmac,
)
import joblib
import numpy as np
import os
import hmac
import hashlib
import json
from datetime import datetime
from pathlib import Path


# Default hyperparameters for XGBoost yield model
DEFAULT_XGB_CONFIG = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "gamma": 0,
    "min_child_weight": 1,
    "seed": 42,
}

DEFAULT_DP_CONFIG = {
    "dp_batch_size": 64,
    "dp_epochs": 8,
    "dp_max_grad_norm": 1.0,
    "dp_learning_rate": 0.05,
    "epsilon": 3.0,
    "delta": 1e-5,
    "dp_hidden_size": 64,
    "seed": 42,
}


def resolve_config(config: dict | None, defaults: dict) -> dict:
    """Merge user config with defaults so all keys are defined."""
    config = resolve_config(config, DEFAULT_XGB_CONFIG)
    return {**defaults, **config}


config = resolve_config(config, DEFAULT_XGB_CONFIG)

model = xgb.XGBRegressor(
    n_estimators=config["n_estimators"],
    max_depth=config["max_depth"],
    learning_rate=config["learning_rate"],
    subsample=config["subsample"],
    colsample_bytree=config["colsample_bytree"],
    gamma=config["gamma"],
    min_child_weight=config["min_child_weight"],
    random_state=config["seed"],
)

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
    config: dict | None = None,
):
    """
    Callable training entry point used by the retraining pipeline Celery task.
    Returns dict: rmse, model_path, baseline_path, trained_at.
    Mirrors the script body exactly — single source of truth.
    """
    config = resolve_config(config, DEFAULT_XGB_CONFIG)
 

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

    model = xgb.XGBRegressor(
        n_estimators=config.get("n_estimators", 200),
        max_depth=config.get("max_depth", 6),
        learning_rate=config.get("learning_rate", 0.1),
        subsample=config.get("subsample", 1.0),
        colsample_bytree=config.get("colsample_bytree", 1.0),
        gamma=config.get("gamma", 0),
        min_child_weight=config.get("min_child_weight", 1),
        random_state=42,
    )
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


def _train_baseline_xgboost(config: Dict, X_train, y_train, X_test):
    config = resolve_config(config, DEFAULT_XGB_CONFIG)
    model = xgb.XGBRegressor(
        n_estimators=config.get("n_estimators", 200),
        max_depth=config.get("max_depth", 6),
        learning_rate=config.get("learning_rate", 0.1),
        subsample=config.get("subsample", 1.0),
        colsample_bytree=config.get("colsample_bytree", 1.0),
        gamma=config.get("gamma", 0),
        min_child_weight=config.get("min_child_weight", 1),
        random_state=config.get("seed", 42),
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, preds


def _train_dp_sgd_regressor(config: Dict, X_train, y_train, X_test):
    """Train a small DP-SGD regressor using optional torch + opacus."""
    config = resolve_config(config, DEFAULT_DP_CONFIG)

    seed = int(config["seed"])
    torch.manual_seed(seed)

    batch_size = int(config["dp_batch_size"])
    epochs = int(config["dp_epochs"])
    max_grad_norm = float(config["dp_max_grad_norm"])
    learning_rate = float(config["dp_learning_rate"])
    epsilon = float(config["epsilon"])
    delta = float(config["delta"])
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
        from opacus import PrivacyEngine
    except Exception as exc:
        raise RuntimeError(
            "DP mode requires optional dependencies. Install with: pip install torch opacus"
        ) from exc

    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)

    X_train_np = np.asarray(X_train, dtype=np.float32)
    y_train_np = np.asarray(y_train, dtype=np.float32).reshape(-1, 1)
    X_test_np = np.asarray(X_test, dtype=np.float32)

    batch_size = int(config.get("dp_batch_size", 64))
    epochs = int(config.get("dp_epochs", 8))
    max_grad_norm = float(config.get("dp_max_grad_norm", 1.0))
    learning_rate = float(config.get("dp_learning_rate", 0.05))
    epsilon = float(config.get("epsilon", 3.0))
    delta = float(config.get("delta", 1e-5))

    train_ds = TensorDataset(
        torch.tensor(X_train_np),
        torch.tensor(y_train_np),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    sample_rate = min(1.0, batch_size / max(1, len(train_ds)))
    steps_per_epoch = math.ceil(len(train_ds) / max(1, batch_size))
    steps = max(1, epochs * steps_per_epoch)

    budget = build_privacy_budget(
        epsilon=epsilon,
        delta=delta,
        sample_rate=sample_rate,
        steps=steps,
    )

    print(
        "DP config -> "
        f"epsilon={budget.epsilon:.4f}, delta={budget.delta:.2e}, "
        f"noise_multiplier={budget.noise_multiplier:.6f}, sample_rate={budget.sample_rate:.6f}, "
        f"steps={budget.steps}"
    )

    input_dim = X_train_np.shape[1]
    model = nn.Sequential(
        nn.Linear(input_dim, int(config.get("dp_hidden_size", 64))),
        nn.ReLU(),
        nn.Linear(int(config.get("dp_hidden_size", 64)), 1),
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    privacy_engine = PrivacyEngine()

    try:
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            epochs=epochs,
            target_epsilon=epsilon,
            target_delta=delta,
            max_grad_norm=max_grad_norm,
        )
    except AttributeError:
        # Fallback for older Opacus versions.
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=budget.noise_multiplier,
            max_grad_norm=max_grad_norm,
        )

    model.train()
    for _ in range(epochs):
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test_np)).cpu().numpy().reshape(-1)

    accountant_epsilon = float(privacy_engine.accountant.get_epsilon(delta=delta))
    used_noise_multiplier = float(getattr(optimizer, "noise_multiplier", budget.noise_multiplier))
    approx_eps = approximate_epsilon(
        noise_multiplier=used_noise_multiplier,
        sample_rate=budget.sample_rate,
        steps=budget.steps,
        delta=budget.delta,
    )

    print(
        "DP accounting -> "
        f"target_epsilon={epsilon:.4f}, spent_epsilon={accountant_epsilon:.4f}, "
        f"approx_epsilon={approx_eps:.4f}, delta={delta:.2e}"
    )

    return model, preds, {
        "target_epsilon": epsilon,
        "spent_epsilon": accountant_epsilon,
        "approx_epsilon": approx_eps,
        "delta": delta,
        "noise_multiplier": used_noise_multiplier,
        "sample_rate": budget.sample_rate,
        "steps": budget.steps,
        "batch_size": batch_size,
        "epochs": epochs,
        "max_grad_norm": max_grad_norm,
        "learning_rate": learning_rate,
        "input_dim": input_dim,
        "hidden_size": int(config.get("dp_hidden_size", 64)),
    }


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

    training_mode = (config.get("training_mode") or "baseline").strip().lower()
    dp_metrics = {}

    # Train model
    if training_mode == "dp_sgd":
        model, preds, dp_metrics = _train_dp_sgd_regressor(
            config=config,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
        )
    elif training_mode == "baseline":
        model, preds = _train_baseline_xgboost(
            config=config,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
        )
    else:
        raise ValueError(f"Unsupported training_mode: {training_mode}")

    # Evaluate
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print("✅ Model trained successfully")
    print("📊 RMSE:", rmse)

    out_path = config.get(
        "output_model",
        "yield_model_dp.pt" if training_mode == "dp_sgd" else "yield_model.joblib",
    )
    if training_mode == "dp_sgd":
        import torch

        torch.save(
            {
                "state_dict": model.state_dict(),
                "training_mode": "dp_sgd",
                "dp_metrics": dp_metrics,
            },
            out_path,
        )
    else:
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
    metrics_payload = {"rmse": float(rmse), "training_mode": training_mode}
    metrics_payload.update(dp_metrics)
    registry.register_model(model_name=model_name, version=version, model_path=out_path, created_by=config.get("created_by", "ci"), description=config.get("description"), metrics=metrics_payload)

    manifest_meta["training_mode"] = training_mode
    if dp_metrics:
        manifest_meta["privacy"] = dp_metrics

    return manifest_meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True, help="Path to JSON config file")
    args = parser.parse_args()

    train_from_config(args.config)

# Split data — temporal ordering preserved; no shuffle to avoid future leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

if __name__ == "__main__":
    main()
