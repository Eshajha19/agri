import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import argparse
import json
import os
import random

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np
import hmac
import hashlib

from ml.repro import create_run_manifest


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass


def train_from_config(config_path: str):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    seed = config.get('seed', 42)
    set_seeds(seed)

    dataset_path = config.get('dataset', 'Train.csv')
    dry_run = config.get('dry_run', False)

    # create run manifest (data provenance)
    manifest = create_run_manifest([dataset_path], config)
    print('Run manifest created with id:', manifest['run_id'])

    if dry_run:
        print('Dry run requested; skipping training.')
        return manifest

    df = pd.read_csv(dataset_path)
    # Convert SDate to datetime
    df['SDate'] = pd.to_datetime(df['SDate'], errors='coerce')
    df = df.dropna(subset=['SDate'])
    df = df.sort_values('SDate')

    X = df.drop(columns=["FarmID", "category", "State", "District", "Sub-District", "SDate", "HDate", "ExpYield", "geometry"], errors='ignore')
    y = df["ExpYield"]

    categorical_cols = config.get('categorical_cols', ['Crop', 'CNext', 'CLast', 'CTransp', 'IrriType', 'IrriSource', 'Season'])
    # Capture category vocabulary BEFORE one-hot encoding for inference validation
    category_vocab = {}
    for col in categorical_cols:
        if col in X.columns:
            category_vocab[col] = sorted(X[col].astype(str).unique().tolist())
    X = pd.get_dummies(X, columns=[c for c in categorical_cols if c in X.columns], drop_first=True)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.get('test_size', 0.2), random_state=seed
    )

    # Train model
    model = xgb.XGBRegressor(n_estimators=config.get('n_estimators', 200), max_depth=config.get('max_depth', 6), random_state=seed)
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print("✅ Model trained successfully")
    print("📊 RMSE:", rmse)

    out_path = config.get('output_model', 'yield_model.joblib')
    joblib.dump(model, out_path)

    # Persist category vocabulary alongside the model for inference validation
    voc_path = os.path.splitext(out_path)[0] + '_vocab.json'
    with open(voc_path, 'w', encoding='utf-8') as vf:
        json.dump(category_vocab, vf)

    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', required=True, help='Path to JSON config file')
    args = parser.parse_args()

    train_from_config(args.config)


if __name__ == '__main__':
    main()

    # Optionally sign the model if a signing key is available in environment
    signing_key = os.getenv("MODEL_SIGNING_KEY")
    if signing_key:
        with open("yield_model.joblib", "rb") as f:
            data = f.read()
        sig = hmac.new(signing_key.encode("utf-8"), data, hashlib.sha256).hexdigest()
        with open("yield_model.joblib.sig", "w", encoding="utf-8") as sf:
            sf.write(sig)
        print("Wrote signature to yield_model.joblib.sig")
    else:
        print("MODEL_SIGNING_KEY not set; no signature file written for yield_model.joblib")
